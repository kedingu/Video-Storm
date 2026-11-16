import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model

try:
    from natten.functional import na2d_av
    _NATTEN_AVAILABLE = True
except Exception:
    na2d_av = None
    _NATTEN_AVAILABLE = False

try:
    from natten.functional import na1d_qk as _na1d_qk, na1d_av as _na1d_av
    _NATTEN1D_AVAILABLE = True
except Exception:
    _na1d_qk = None
    _na1d_av = None
    _NATTEN1D_AVAILABLE = False

try:
    from einops import rearrange
    from einops import einsum as einops_einsum
    _EINOPS_AVAILABLE = True
except Exception:
    rearrange = None
    einops_einsum = None
    _EINOPS_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

import os, sys

_IGEMM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'SLaK', 'cutlass', 'examples',
    '19_large_depthwise_conv2d_torch_extension')
if os.path.isdir(_IGEMM_PATH) and _IGEMM_PATH not in sys.path:
    sys.path.insert(0, os.path.normpath(_IGEMM_PATH))

_IGEMM_CHECKED = False
_IGEMM_AVAILABLE = False


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding,
               dilation, groups, bias, attempt_use_lk_impl=True):
    global _IGEMM_CHECKED, _IGEMM_AVAILABLE
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2))
    if attempt_use_lk_impl and need_large_impl:
        if not _IGEMM_CHECKED:
            try:
                from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
                _IGEMM_AVAILABLE = True
                print('[VideoSTORM] iGEMM large-kernel implementation found.')
            except Exception:
                _IGEMM_AVAILABLE = False
                print('[VideoSTORM] iGEMM not found, using standard Conv2d.')
            _IGEMM_CHECKED = True
        if _IGEMM_AVAILABLE and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=True):
    return nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    return torch.cat([F.conv_transpose2d(kernel[:, i:i+1], identity_kernel,
                                         stride=dilate_rate)
                      for i in range(kernel.size(1))], dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_kernel.size(2) - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


# ---------------------------------------------------------------------------
# 1D reparameterization helpers
# ---------------------------------------------------------------------------

def fuse_bn_1d(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1),
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)


def convert_dilated_to_nondilated_1d(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose1d(kernel, identity_kernel, stride=dilate_rate)
    return torch.cat([F.conv_transpose1d(kernel[:, i:i+1], identity_kernel,
                                         stride=dilate_rate)
                      for i in range(kernel.size(1))], dim=1)


def merge_dilated_into_large_kernel_1d(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_kernel.size(2) - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated_1d(dilated_kernel, dilated_r)
    pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [pad, pad])


def _temporal_kernel_size(num_frames):
    k = (num_frames - 1) // 2 * 2 + 1
    return max(min(k, 31), 3)


def _joint_attn_flags(policy, n):
    """[FIX 11] Resolve a joint-attention policy string into per-block flags.

    'none'      : all blocks use per-frame spatial attention (B*T, HW).
    'alternate' : every other block (odd indices) uses joint spatio-temporal
                  attention (B, T*HW); even indices stay per-frame. The first
                  block of a sub-stage stays per-frame (it handles the context
                  blending), in the spirit of UniRepLKNet's alternating 13/3
                  kernel pattern.
    'all'       : all blocks use joint spatio-temporal attention.
    """
    if policy == 'all':
        return [True] * n
    if policy == 'alternate':
        return [(i % 2 == 1) for i in range(n)]
    if policy == 'none':
        return [False] * n
    raise ValueError(f"joint attention policy must be 'none' | 'alternate' | "
                     f"'all', got {policy!r}")


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size,
                         padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


# ---------------------------------------------------------------------------
# DWConv3d / ResDWConv3d
# [FIX 10: joint local spatio-temporal mixing in STORMBlocks]
# ---------------------------------------------------------------------------

class DWConv3d(nn.Module):
    """[FIX 10] Depthwise 3x3x3 convolution in the (B, C, T, H, W) view.

    UniFormer-CBlock lesson: local mixing must be spatio-temporally JOINT,
    not factorized. Every temporal path of VideoSTORM (TTB, TOB SDPA, lepe,
    PTGU) operates per pixel in (B*H*W, C, T) and is therefore blind to
    cross-location motion; UniFormer instead uses depthwise 3D convs
    (its pos_embed conv_3x3x3 and local-attention conv_5x5x5) so that an
    object moving by one cell per frame is captured directly by the kernel.

    This module replaces the 2D depthwise 3x3 convs inside STORMBlocks
    (dwconv_ctx, fusion DW, dwconv2, MLP DW): input (B*T, C, H, W) is
    reshaped to (B, C, T, H, W), convolved depthwise over (t±1, i±1, j±1),
    and reshaped back. Acts as a dynamic spatio-temporal positional encoding
    (zero padding encodes absolute position, UniFormer's DPE argument),
    which also feeds the joint attention of FIX 11.

    These modules are new (randomly initialized, inside the sub-blocks):
    no pretrained weight is affected, standard trunc_normal_ init applies
    (active path -- same regime as the PTGU). Cost: 27 vs 9 weights per
    channel on depthwise convs, negligible.

    Frame-count convention: T = self.num_frames, consistent with
    TemporalTransitionBranch / TemporalOverarchingBranch.
    """

    def __init__(self, dim, num_frames, bias=False):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1,
                              groups=dim, bias=bias)

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv(x)
        return x.permute(0, 2, 1, 3, 4).contiguous().view(BT, C, H, W)

    def extra_repr(self):
        return f'dim={self.dim}, num_frames={self.num_frames}'


class ResDWConv3d(DWConv3d):
    """[FIX 10] Residual variant of DWConv3d (3D analogue of ResDWConv)."""

    def __init__(self, dim, num_frames, bias=True):
        super().__init__(dim, num_frames, bias=bias)

    def forward(self, x):
        return x + super().forward(x)


class GRN(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return (self.gamma * Nx + 1) * x + self.beta if self.use_bias \
               else (self.gamma * Nx + 1) * x


class LayerScale(nn.Module):
    """Per-channel multiplicative scale (Touvron et al., CaiT).

    [FIX] 1D weights of shape (dim,) instead of (dim,1,1,1).
    """
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * init_value,
                                   requires_grad=True)

    def forward(self, x):
        shape = [1] * x.dim()
        shape[1] = -1
        return x * self.weight.view(*shape)


class GRNwithNHWC(nn.Module):
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x): return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x): return x.permute(0, 3, 1, 2)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert data_format in ("channels_last", "channels_first")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        return F.layer_norm(
            x.permute(0, 2, 3, 1).contiguous(),
            self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2).contiguous()


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        return inputs * torch.sigmoid(x).view(-1, self.input_channels, 1, 1)


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        return x * self.proj(x)


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=True,
                 attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(
            channels, channels, kernel_size, stride=1,
            padding=kernel_size // 2, dilation=1,
            groups=channels, bias=deploy,
            attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]; self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]; self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]; self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]; self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]; self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]; self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]; self.dilates = [1, 2]
        else:
            raise ValueError('DilatedReparamBlock requires kernel_size >= 5')
        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                    nn.Conv2d(channels, channels, kernel_size=k, stride=1,
                              padding=(r * (k - 1) + 1) // 2, dilation=r,
                              groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', get_bn(channels, use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            out = out + self.__getattr__(f'dil_bn_k{k}_{r}')(
                            self.__getattr__(f'dil_conv_k{k}_{r}')(x))
        return out

    def merge_dilated_branches(self):
        if not hasattr(self, 'origin_bn'):
            return
        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            branch_k, branch_b = fuse_bn(
                self.__getattr__(f'dil_conv_k{k}_{r}'),
                self.__getattr__(f'dil_bn_k{k}_{r}'))
            origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b
        merged = get_conv2d(
            origin_k.size(0), origin_k.size(0), origin_k.size(2),
            stride=1, padding=origin_k.size(2) // 2, dilation=1,
            groups=origin_k.size(0), bias=True,
            attempt_use_lk_impl=self.attempt_use_lk_impl)
        merged.weight.data = origin_k
        merged.bias.data = origin_b
        self.lk_origin = merged
        self.__delattr__('origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{k}_{r}')
            self.__delattr__(f'dil_bn_k{k}_{r}')


# ---------------------------------------------------------------------------
# TemporalReceptiveBlock
# ---------------------------------------------------------------------------

class TemporalReceptiveBlock(nn.Module):
    _BRANCH_CFG = {
        5:  ([3, 3],          [1, 2]),
        7:  ([5, 3, 3],       [1, 2, 3]),
        9:  ([5, 5, 3, 3],    [1, 2, 3, 4]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
        17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
        19: ([5, 9, 3, 3, 3, 3],   [1, 2, 4, 5, 7, 9]),
        21: ([5, 9, 3, 3, 3, 3],   [1, 2, 4, 6, 8, 10]),
        23: ([5, 9, 3, 3, 3, 3],   [1, 2, 4, 6, 9, 11]),
        25: ([5, 9, 3, 3, 3, 3],   [1, 2, 4, 7, 9, 12]),
        27: ([5, 9, 3, 3, 3, 3],   [1, 3, 4, 7, 10, 13]),
        29: ([5, 9, 3, 3, 3, 3],   [1, 3, 5, 7, 10, 14]),
        31: ([5, 9, 3, 3, 3, 3],   [1, 3, 5, 7, 10, 15]),
    }

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=True):
        super().__init__()
        if kernel_size not in self._BRANCH_CFG:
            raise ValueError(f'TemporalReceptiveBlock: unsupported kernel_size={kernel_size}')
        self.kernel_sizes, self.dilates = self._BRANCH_CFG[kernel_size]
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d
        self.lk_origin = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size // 2, groups=channels,
                                    bias=deploy)
        if not deploy:
            self.origin_bn = BN1d(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                    nn.Conv1d(channels, channels, kernel_size=k, stride=1,
                              padding=(r * (k - 1) + 1) // 2, dilation=r,
                              groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', BN1d(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            out = out + self.__getattr__(f'dil_bn_k{k}_{r}')(
                            self.__getattr__(f'dil_conv_k{k}_{r}')(x))
        return out

    def merge_temporal_branches(self):
        if not hasattr(self, 'origin_bn'):
            return
        origin_k, origin_b = fuse_bn_1d(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            branch_k, branch_b = fuse_bn_1d(
                self.__getattr__(f'dil_conv_k{k}_{r}'),
                self.__getattr__(f'dil_bn_k{k}_{r}'))
            origin_k = merge_dilated_into_large_kernel_1d(origin_k, branch_k, r)
            origin_b = origin_b + branch_b
        merged = nn.Conv1d(origin_k.size(0), origin_k.size(0), origin_k.size(2),
                           stride=1, padding=origin_k.size(2) // 2,
                           groups=origin_k.size(0), bias=True)
        merged.weight.data = origin_k
        merged.bias.data = origin_b
        self.lk_origin = merged
        self.__delattr__('origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{k}_{r}')
            self.__delattr__(f'dil_bn_k{k}_{r}')


# ---------------------------------------------------------------------------
# TemporalTransitionBranch
# [FIX 8: zero-gamma identity init -- TTB is an exact identity at step 0]
# ---------------------------------------------------------------------------

class TemporalTransitionBranch(nn.Module):
    """Per-pixel temporal depthwise convolution branch for STLK blocks.

    [FIX 8] norm_1d (the final BN of the branch) is zero-gamma initialized.
    Rationale: the depthwise temporal convs receive trunc_normal_ init and
    the final BN renormalizes their output to unit variance per channel.
    Without this fix, every pretrained backbone block receives v_s + v_t
    where v_t is unit-variance temporal NOISE at step 0 -- the network spends
    the early epochs repairing the ImageNet-21K features instead of learning
    temporal dynamics. With gamma = 0, the whole branch outputs exactly zero
    at step 0 (true identity, consistent with the zero-init policy of
    pe_q / pe_k / temporal_embed / rel_pos_bias_table). gamma receives a
    non-zero gradient from the first step and opens the branch progressively
    (standard zero-gamma adapter practice: TSM / TAM / AIM).

    The _zero_gamma_init flag protects this init from the global BN init in
    VideoSTORM._init_weights (which sets gamma = 1 everywhere else).

    Checkpoint compatibility: unchanged (TTB params are absent from the 2D
    checkpoint either way).
    """

    def __init__(self, dim, kernel_size, num_frames,
                 deploy=False, use_sync_bn=True):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        k_t = _temporal_kernel_size(num_frames)
        self.k_t = k_t

        if k_t >= 5:
            self.dwconv_1d = TemporalReceptiveBlock(dim, k_t, deploy=deploy,
                                                    use_sync_bn=use_sync_bn)
        else:
            self.dwconv_1d = nn.Conv1d(dim, dim, kernel_size=k_t,
                                       padding=k_t // 2, groups=dim,
                                       bias=deploy)

        self.norm_1d = nn.Identity() if deploy else BN1d(dim)

        # [FIX 8] Zero-gamma init on the final branch BN -> exact identity
        # at step 0. The flag exempts this BN from the global gamma=1 init.
        if not deploy:
            nn.init.zeros_(self.norm_1d.weight)
            nn.init.zeros_(self.norm_1d.bias)
            self.norm_1d._zero_gamma_init = True

    @staticmethod
    def _to_1d(x, B, T, H, W):
        return (x.view(B, T, x.size(1), H, W)
                  .permute(0, 3, 4, 2, 1).contiguous()
                  .view(B * H * W, x.size(1), T))

    @staticmethod
    def _from_1d(x_1d, B, T, H, W, C):
        return (x_1d.view(B, H, W, C, T)
                    .permute(0, 4, 3, 1, 2).contiguous()
                    .view(B * T, C, H, W))

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T
        x_1d = self._to_1d(x, B, T, H, W)
        out = self.norm_1d(self.dwconv_1d(x_1d))
        return self._from_1d(out, B, T, H, W, C)

    def reparameterize(self):
        if isinstance(self.dwconv_1d, TemporalReceptiveBlock):
            self.dwconv_1d.merge_temporal_branches()
        if isinstance(self.norm_1d, (nn.BatchNorm1d, nn.SyncBatchNorm)):
            conv = (self.dwconv_1d.lk_origin
                    if isinstance(self.dwconv_1d, TemporalReceptiveBlock)
                    else self.dwconv_1d)
            w, b = fuse_bn_1d(conv, self.norm_1d)
            fused = nn.Conv1d(w.size(0), w.size(0), w.size(2),
                              stride=1, padding=w.size(2) // 2,
                              groups=w.size(0), bias=True)
            fused.weight.data = w
            fused.bias.data = b
            if isinstance(self.dwconv_1d, TemporalReceptiveBlock):
                self.dwconv_1d.lk_origin = fused
            else:
                self.dwconv_1d = fused
            self.norm_1d = nn.Identity()


# ---------------------------------------------------------------------------
# PosTemporalGatingUnit
# [FIX 9: signed, position-based temporal mixing with multiplicative gating]
# ---------------------------------------------------------------------------

class PosTemporalGatingUnit(nn.Module):
    """Positional temporal gating unit, adapted from PosMLP-Video's PoSGUT.

    [FIX 9] Structural complement to the softmax temporal SDPA. Softmax
    attention computes a convex combination over frames (non-negative weights
    summing to 1): it can average over time but can NEVER express a signed
    operator such as a temporal difference or an antisymmetric order
    detector -- precisely the operator class required by direction-sensitive
    benchmarks (SSv2, Diving48). This module provides it:

    1. A learnable relative-position table (2T-1, gamma) is unfolded into a
       full T x T mixing matrix per channel group, with FREE-SIGNED weights:
       it can directly implement frame differencing (+1/-1 on adjacent
       offsets), directional filters, or any per-group temporal FIR.
    2. The mixed half-features are multiplied elementwise with the gating
       half (u * v): a second-order interaction between "what changes" and
       "what is present" -- the motion-feature mechanism that drives both
       PosMLP-Video's PoSGUT and VideoMogaNet's MOTM.
    3. The position table is trunc_normal_-initialized (NOT zero): this path
       lives inside the randomly-initialized STORMBlocks, so there is no
       pretrained weight to protect, and the temporal ordering signal must be
       active from the very first step (zero-init here would reproduce the
       weak-gradient pathology this fix addresses).

    Layout: input/output (B*H*W, dim, T), identical to the other TOB paths.
    Cost: fc1 (dim -> 2*dim) + T x T mixing per group + fc2 (dim -> dim);
    negligible next to the spatial SDPA at T = 8-16.

    Checkpoint compatibility: new module, absent from the 2D checkpoint,
    handled by the strict=False load. pos_table / pos_bias are exempt from
    weight decay (see VideoSTORM.no_weight_decay_keywords).
    """

    def __init__(self, dim, num_frames, gamma=8):
        super().__init__()
        assert dim % gamma == 0, \
            f'PosTemporalGatingUnit: dim={dim} not divisible by gamma={gamma}'
        self.dim = dim
        self.num_frames = num_frames
        self.gamma = gamma

        # fc1 expands to 2*dim, then chunked into u (mixed) and v (gate).
        self.fc1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(dim, dim, kernel_size=1)

        # Signed relative-position mixing table: one T x T map per group.
        # trunc_normal_ init (active at step 0), matching PosMLP-Video.
        self.pos_table = nn.Parameter(torch.zeros(2 * num_frames - 1, gamma))
        trunc_normal_(self.pos_table, std=.02)

        # Per-output-position bias (PosMLP's token_proj_n_bias), zero-init.
        self.pos_bias = nn.Parameter(torch.zeros(num_frames))

    def _get_pos_map(self, T, dtype):
        """Return the (gamma, T, T) signed mixing matrix; W[g, m, n] is the
        weight from input frame n to output frame m, indexed by the relative
        offset m - n. Interpolated when T != num_frames (test-time dense
        sampling)."""
        table = self.pos_table                                # (2*nf-1, g)
        if T != self.num_frames:
            table = F.interpolate(
                table.t().unsqueeze(0),                       # (1, g, 2*nf-1)
                size=2 * T - 1, mode='linear', align_corners=False
            ).squeeze(0).t()                                  # (2*T-1, g)
        idx = torch.arange(T, device=table.device)
        rel = idx[:, None] - idx[None, :] + (T - 1)           # (T, T) in [0, 2T-2]
        pos_map = table[rel]                                  # (T, T, g)
        return pos_map.permute(2, 0, 1).to(dtype)             # (g, T, T)

    def _get_pos_bias(self, T, dtype):
        bias = self.pos_bias
        if T != self.num_frames:
            bias = F.interpolate(bias.view(1, 1, -1), size=T,
                                 mode='linear', align_corners=False).view(-1)
        return bias.to(dtype)

    def forward(self, x_1d):
        BHW, C, T = x_1d.shape
        u, v = self.act(self.fc1(x_1d)).chunk(2, dim=1)       # (BHW, dim, T) each

        pos_map = self._get_pos_map(T, u.dtype)               # (g, T, T)
        u = u.view(BHW, self.gamma, C // self.gamma, T)
        u = torch.einsum('gmn,bgcn->bgcm', pos_map, u)
        u = u + self._get_pos_bias(T, u.dtype).view(1, 1, 1, T)
        u = u.reshape(BHW, C, T)

        return self.fc2(u * v)

    def extra_repr(self):
        return f'dim={self.dim}, num_frames={self.num_frames}, gamma={self.gamma}'


# ---------------------------------------------------------------------------
# TemporalOverarchingBranch
# [FIX 1: temporal positional encoding on Q and K]
# [FIX 6: temporal relative position bias on attention logits]
# [FIX 9: PosTemporalGatingUnit -- signed positional temporal mixing path]
# ---------------------------------------------------------------------------

class TemporalOverarchingBranch(nn.Module):
    """1D temporal mirror of STORMBlock's spatial attention path.

    [FIX 1] pe_q / pe_k : learnable absolute positional encodings injected
    into Q and K before SDPA. V remains unmodified (content semantics).
    Both PEs are zero-initialized; gradients flow through weight_q/k_1d.
    Interpolation handles T != num_frames at inference.

    [FIX 5 cancelled] The key source is used UNMODIFIED. Keys come from
    the context encoder, whose features guide the sub-stages (STORM
    modules); the former key_dw spatialization altered this guidance
    signal and caused a large performance loss. No spatial operation is
    applied to keys (nor to V).

    [FIX 6] rel_pos_bias_table : learnable temporal relative position bias
    (2T-1, num_heads), zero-initialized, added to attention logits via the
    attn_mask argument of F.scaled_dot_product_attention. Stronger ordering
    signal than absolute PEs alone (directly biases who-attends-to-whom by
    temporal offset). May fall back from the Flash kernel; negligible at
    T = 8-16 for this per-pixel attention. Interpolated when T != num_frames.

    [FIX 9] ptgu : PosTemporalGatingUnit, a third output path parallel to the
    SDPA and lepe paths. It contributes the signed, position-based temporal
    operators (differencing, order detection) and the multiplicative gating
    that the convex softmax attention cannot express. Output composition:
    x_out = dyconv_proj(SDPA) + lepe + ptgu. DISABLED by default
    (use_ptgu=False): ablations showed no performance gain; set
    use_ptgu=True to restore the path. See PosTemporalGatingUnit for
    the full rationale and init policy.
    """

    def __init__(self, dim, ctx_ch, num_heads, num_frames, spatial_kernel_size,
                 deploy=False, use_sync_bn=True, ptgu_gamma=8,
                 use_ptgu=False):
        super().__init__()

        self.num_frames = num_frames
        self.dim = dim
        self.ctx_ch = ctx_ch
        self.num_heads = num_heads

        self.head_dim_qk = dim // num_heads
        self.head_dim_v  = dim // num_heads

        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        k_t = _temporal_kernel_size(num_frames)
        self.k_t = k_t

        if k_t >= 5:
            lepe_conv = TemporalReceptiveBlock(dim, k_t, deploy=deploy,
                                               use_sync_bn=use_sync_bn)
        else:
            lepe_conv = nn.Conv1d(dim, dim, kernel_size=k_t,
                                  padding=k_t // 2, groups=dim, bias=deploy)
        self.lepe_1d = nn.Sequential(lepe_conv, BN1d(dim))

        self.weight_q_1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            BN1d(dim))

        self.weight_k_1d = nn.Sequential(
            nn.Conv1d(ctx_ch, dim, kernel_size=1, bias=False),
            BN1d(dim))

        self.dyconv_proj_1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            BN1d(dim))

        # [FIX 1] Learnable temporal positional encodings.
        # pe_q : added to x_1d  before weight_q_1d  -> Q is position-aware.
        # pe_k : added to h_1d  before weight_k_1d  -> K is position-aware.
        # V    = x_1d (no PE)   -> content semantics preserved.
        # Zero-init: at step 0, SDPA is position-agnostic (safe for pre-training
        # compatibility). Gradients flow through weight_q/k_1d which are non-zero.
        self.pe_q = nn.Parameter(torch.zeros(1, dim, num_frames))
        self.pe_k = nn.Parameter(torch.zeros(1, ctx_ch, num_frames))

        # [FIX 6] Temporal relative position bias table, zero-initialized
        # (identity at step 0). Exempt from weight decay
        # (see VideoSTORM.no_weight_decay_keywords).
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros(2 * num_frames - 1, num_heads))

        # [FIX 9] Signed positional temporal mixing + multiplicative gating.
        # trunc_normal_ position table (active at step 0): this path carries
        # the temporal-difference / order-detection operators that the convex
        # softmax SDPA cannot represent.
        # Optional (use_ptgu, default False: ablations showed no gain):
        # None removes the third output path and all its parameters.
        self.ptgu = (PosTemporalGatingUnit(dim, num_frames,
                                           gamma=ptgu_gamma)
                     if use_ptgu else None)

    @staticmethod
    def _to_1d(x, B, T, H, W):
        return (x.view(B, T, x.size(1), H, W)
                  .permute(0, 3, 4, 2, 1).contiguous()
                  .view(B * H * W, x.size(1), T))

    @staticmethod
    def _from_1d(x_1d, B, T, H, W, C):
        return (x_1d.view(B, H, W, C, T)
                    .permute(0, 4, 3, 1, 2).contiguous()
                    .view(B * T, C, H, W))

    def _get_pe(self, T, device, dtype):
        """Return pe_q, pe_k interpolated to the current T if needed."""
        pe_q = self.pe_q.to(dtype)
        pe_k = self.pe_k.to(dtype)
        if pe_q.shape[-1] != T:
            pe_q = F.interpolate(pe_q, size=T, mode='linear', align_corners=False)
            pe_k = F.interpolate(pe_k, size=T, mode='linear', align_corners=False)
        return pe_q, pe_k

    def _get_rel_pos_bias(self, T, dtype):
        """[FIX 6] Return additive bias (1, num_heads, T, T) for SDPA."""
        table = self.rel_pos_bias_table                       # (2*nf-1, heads)
        if T != self.num_frames:
            table = F.interpolate(
                table.t().unsqueeze(0),                       # (1, heads, 2*nf-1)
                size=2 * T - 1, mode='linear', align_corners=False
            ).squeeze(0).t()                                  # (2*T-1, heads)
        idx = torch.arange(T, device=table.device)
        rel = idx[None, :] - idx[:, None] + (T - 1)           # (T, T) in [0, 2T-2]
        bias = table[rel]                                     # (T, T, heads)
        return bias.permute(2, 0, 1).unsqueeze(0).to(dtype)   # (1, heads, T, T)

    def _temporal_attention(self, x_1d, h_1d):
        BHW, C, T = x_1d.shape

        # [FIX 1] Inject positional encodings into Q and K; V is unmodified.
        pe_q, pe_k = self._get_pe(T, x_1d.device, x_1d.dtype)
        Q = self.weight_q_1d(x_1d + pe_q)
        K = self.weight_k_1d(h_1d + pe_k)
        V = x_1d   # content-only, no PE

        Q = Q.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()
        K = K.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()
        V = V.view(BHW, self.num_heads, self.head_dim_v,  T).permute(0, 1, 3, 2).contiguous()

        # [FIX 6] Additive relative position bias on attention logits.
        rel_bias = self._get_rel_pos_bias(T, Q.dtype)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=rel_bias)
        return out.permute(0, 1, 3, 2).contiguous().view(BHW, C, T)

    def forward(self, x_cat):
        BT, out_dim, H, W = x_cat.shape
        T = self.num_frames
        B = BT // T

        # [FIX 5 cancelled] The key source (context channels) is used
        # UNMODIFIED: the context guides the sub-stages and must reach
        # the temporal attention intact.
        x_2d, h_2d = torch.split(x_cat, [self.dim, self.ctx_ch], dim=1)

        x_1d = self._to_1d(x_2d, B, T, H, W)
        h_1d = self._to_1d(h_2d, B, T, H, W)

        lepe   = self.lepe_1d(x_1d)
        x_attn = self._temporal_attention(x_1d, h_1d)
        x_attn = self.dyconv_proj_1d(x_attn)

        out = x_attn + lepe
        if self.ptgu is not None:
            # [FIX 9] Signed positional gating path, parallel to SDPA
            # and lepe.
            out = out + self.ptgu(x_1d)

        return self._from_1d(out, B, T, H, W, self.dim)

    def reparameterize(self):
        if isinstance(self.lepe_1d[0], TemporalReceptiveBlock):
            self.lepe_1d[0].merge_temporal_branches()


# ---------------------------------------------------------------------------
# SpatioTemporalLargeKernelBlock  (TSM ablation variant)
# [FIX 8: TTB zero-gamma identity init -- see TemporalTransitionBranch]
# ---------------------------------------------------------------------------

class SpatioTemporalLargeKernelBlock(nn.Module):
    """UniRepLKNet block extended with temporal transition branch.

    TSM ablation variant: the temporal channel shift (former FIX 4) is
    removed; the spatial DWconv sees the unshifted input. With FIX 8, this
    block is therefore an EXACT identity to the pretrained UniRepLKNet
    block at step 0.

    [FIX 8] The TTB now opens from an exact zero (zero-gamma final BN): at
    step 0 this block computes exactly the pretrained UniRepLKNet function
    (up to the TSM shift), instead of pretrained + unit-variance temporal
    noise. See TemporalTransitionBranch.

    Checkpoint compatibility: the spatial path parameters
    (dwconv, norm, se, pwconv*, gamma) are structurally unchanged. The checkpoint loads identically.
    """

    def __init__(self, dim, kernel_size, drop_path=0.,
                 layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=True, ffn_factor=4, num_frames=0):
        super().__init__()
        self.with_cp = with_cp
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(
                dim, kernel_size, deploy=deploy,
                use_sync_bn=use_sync_bn,
                attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in (3, 5)
            self.dwconv = get_conv2d(
                dim, dim, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2, dilation=1,
                groups=dim, bias=deploy,
                attempt_use_lk_impl=attempt_use_lk_impl)
        self.norm = (nn.Identity() if (deploy or kernel_size == 0)
                     else get_bn(dim, use_sync_bn))
        self.se = SEBlock(dim, dim // 4)
        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(nn.Linear(ffn_dim, dim), NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn))
        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
                      if (not deploy) and layer_scale_init_value is not None
                         and layer_scale_init_value > 0
                      else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # TSM ablation variant: no temporal channel shift is applied.
        self.tsm = None

        self.temporal_branch = None
        if num_frames > 0 and kernel_size != 0 and not deploy:
            self.temporal_branch = TemporalTransitionBranch(
                dim=dim, kernel_size=kernel_size,
                num_frames=num_frames,
                deploy=False, use_sync_bn=use_sync_bn)

    def compute_residual(self, x):
        # Parallel design: pretrained DWconv and TTB both receive x-derived
        # inputs without cross-feeding. Cascade via DWconv(x + v_t) was
        # reverted: applying a large spatial kernel (k=13) to temporal
        # features (v_t) causes spatial smearing of temporal signals,
        # compounding destructively over 27 blocks in stage 2 and corrupting
        # the pretrained backbone features before the STORMBlocks. Temporal
        # cascade is handled exclusively in STORMBlocks (Fix 2) where modules
        # are randomly initialized and safe to modify.
        x_s = x
        v_s = self.norm(self.dwconv(x_s))
        if self.temporal_branch is not None:
            v_t = self.temporal_branch(x)
            y = self.se(v_s + v_t)
        else:
            y = self.se(v_s)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def _f(x): return x + self.compute_residual(x)
        return cp.checkpoint(_f, inputs) if self.with_cp and inputs.requires_grad \
               else _f(inputs)

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= \
                    (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = (
                    self.norm.bias
                    + (self.dwconv.lk_origin.bias - self.norm.running_mean)
                    * self.norm.weight / std)
            else:
                conv = nn.Conv2d(
                    self.dwconv.in_channels, self.dwconv.out_channels,
                    self.dwconv.kernel_size, padding=self.dwconv.padding,
                    groups=self.dwconv.groups, bias=True)
                conv.weight.data = (self.dwconv.weight
                                    * (self.norm.weight / std).view(-1, 1, 1, 1))
                conv.bias.data = (self.norm.bias
                                    - self.norm.running_mean * self.norm.weight / std)
                self.dwconv = conv
            self.norm = nn.Identity()
        if self.temporal_branch is not None:
            self.temporal_branch.reparameterize()
        final_scale = self.gamma.data if self.gamma is not None else 1
        self.gamma = None
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = (
                linear.weight * (bn.weight / std * final_scale).view(-1, 1))
            lb = (0 if linear.bias is None else linear.bias.data) + grn_bias_projected
            new_linear.bias.data = (
                (bn.bias + (lb - bn.running_mean) * bn.weight / std) * final_scale)
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


# ---------------------------------------------------------------------------
# RetroActiveBridge
# ---------------------------------------------------------------------------

class RetroActiveBridge(nn.Module):
    def __init__(self, dim, h_dim, use_sync_bn=True):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            get_bn(h_dim, use_sync_bn))
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2,
                      padding=1, bias=False),
            get_bn(h_dim // 4, use_sync_bn))

    def forward(self, x, ctx):
        return self.x_proj(x), self.h_proj(ctx)


# ---------------------------------------------------------------------------
# SpatioTemporalOverarchingRetroactiveModulationBlock
# [FIX 2: T-to-S cascade, with toggle]
# [FIX 10: 3D depthwise local convs (UniFormer-CBlock lesson)]
# [FIX 11: joint spatio-temporal SDPA (UniFormer-SABlock lesson)]
# [FIX 12: temporally-informed gate]
# ---------------------------------------------------------------------------

class SpatioTemporalOverarchingRetroactiveModulationBlock(nn.Module):
    """STORMBlock.

    [FIX 2] Temporal -> spatial cascade, controlled by `t2s_cascade`:
    - t2s_cascade=True (default): x_t is computed before the spatial SDPA
      and the spatial query source is enriched with it
      (Q = weight_q(query_src + x_t)). The spatial attention thus queries
      from a temporally-informed perspective. x_t ALSO contributes as a
      direct residual to x_agg (dual contribution, mirroring PosMLP's
      sequential design). No new parameters; zero additional compute.
    - t2s_cascade=False: strictly parallel composition -- x_t contributes
      only as a residual to x_agg, the spatial Q is computed from the
      unmodified query_src. This reproduces the previous (pre-fix)
      behaviour, kept available for the cascade-vs-parallel ablation.

    NOTE: previous revisions of this file DOCUMENTED the cascade but did
    not implement it (query_src was passed unmodified to the spatial SDPA).
    Any earlier ablation labelled "FIX 2" actually measured the parallel
    configuration (equivalent to t2s_cascade=False).

    [FIX 10] All depthwise 3x3 local convs of the block (dwconv_ctx, the
    fusion DW, dwconv2 and the MLP DW) become depthwise 3x3x3 convs in the
    (B, C, T, H, W) view (DWConv3d / ResDWConv3d) when num_frames > 0.
    UniFormer-CBlock lesson: joint local spatio-temporal mixing in every
    block -- the displacement of an object by one cell per frame becomes
    visible everywhere, not only on the TOB keys. Also serves as a dynamic
    spatio-temporal positional encoding (zero padding encodes position),
    feeding the joint attention of FIX 11.

    [FIX 11] `joint_st_attn`: when True, the spatial SDPA operates over the
    JOINT token set (B, T*H*W) instead of per-frame (B*T, H*W). UniFormer-
    SABlock lesson: this is the operation that lets attention answer "where
    did this object go in the other frames" -- the cross-location temporal
    reasoning that neither the per-frame spatial SDPA nor the per-pixel
    temporal SDPA of the TOB can express. The TOB remains in place as the
    specialist for fine-grained per-position temporal reasoning. The policy
    (which blocks are joint) is set per sub-stage in VideoSTORM via
    joint_attn_s2 / joint_attn_s3.

    [FIX 12] `temporal_gate`: the gate is computed from x + x_t instead of
    x alone (when x_t exists). The gate multiplicatively modulates x_agg,
    which CONTAINS the temporal contribution x_t; a gate blind to time can
    silently suppress the temporal signal. Zero extra parameters.

    All prior fixes preserved:
    [FIX #2]  ls1 / ls2 on x_mixed.
    [FIX #4]  Temporal branch after norm_ctx.
    [FIX #5]  head_dim_qk = dim // num_heads.
    [FIX #10] All BN2d via get_bn(dim, use_sync_bn).
    [MODIF Prof] ls_temporal removed; sub_ls_init=1.0.

    TOB capabilities (see TemporalOverarchingBranch):
    [FIX 6] temporal relative position bias.
    [FIX 9] PosTemporalGatingUnit (signed positional temporal mixing).
    """

    def __init__(self, dim, kernel_size, ctx_ch, smk_size=5, num_heads=4,
                 drop_path=0., layer_scale_init_value=1.0, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False, use_sync_bn=True,
                 ffn_factor=2, is_first=False, is_last=False, num_frames=0,
                 res_scale=True, t2s_cascade=True, ptgu_gamma=8,
                 joint_st_attn=False, temporal_gate=True,
                 use_ptgu=False):
        super().__init__()
        assert _EINOPS_AVAILABLE, '[VideoSTORM] einops required for STORMBlock.'
        self.kernel_size = kernel_size
        self.num_heads = num_heads * 2
        self.is_first = is_first
        self.is_last = is_last
        self.res_scale = res_scale
        self.with_cp = with_cp
        self.dim = dim
        self.ctx_ch = ctx_ch
        self.num_frames = num_frames
        # [FIX 2] Cascade toggle (True = T-to-S cascade, False = parallel sum).
        self.t2s_cascade = t2s_cascade
        # [FIX 11] Joint spatio-temporal attention flag for this block.
        self.joint_st_attn = joint_st_attn and num_frames > 0
        # [FIX 12] Temporally-informed gate toggle.
        self.temporal_gate = temporal_gate
        out_dim = dim + ctx_ch

        self.head_dim_qk = dim // self.num_heads
        self.head_dim_v  = dim // self.num_heads

        if not is_first:
            self.x_scale = LayerScale(ctx_ch, init_value=1)
            self.h_scale = LayerScale(ctx_ch, init_value=1)

        # [FIX 10] 3D depthwise local convs when the block is temporal;
        # 2D fallback preserves image-mode operation (num_frames == 0).
        if num_frames > 0:
            self.dwconv_ctx = ResDWConv3d(out_dim, num_frames)
        else:
            self.dwconv_ctx = ResDWConv(out_dim, kernel_size=3)
        self.norm_ctx = LayerNorm2d(out_dim)

        fusion_dw = (DWConv3d(out_dim, num_frames, bias=False)
                     if num_frames > 0 else
                     nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,
                               groups=out_dim, bias=False))
        self.fusion = nn.Sequential(
            fusion_dw,
            get_bn(out_dim, use_sync_bn),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim))

        self.weight_q = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            get_bn(dim, use_sync_bn))

        self.weight_k = nn.Sequential(
            nn.Conv2d(ctx_ch, dim, kernel_size=1, bias=False),
            get_bn(dim, use_sync_bn))

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            get_bn(dim, use_sync_bn))

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                use_sync_bn=use_sync_bn,
                                attempt_use_lk_impl=attempt_use_lk_impl),
            get_bn(dim, use_sync_bn))

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            get_bn(dim, use_sync_bn),
            nn.SiLU())

        self.se = SEModule(dim)

        self.temporal_branch = None
        if num_frames > 0 and not deploy:
            self.temporal_branch = TemporalOverarchingBranch(
                dim=dim, ctx_ch=ctx_ch,
                num_heads=self.num_heads,
                num_frames=num_frames,
                spatial_kernel_size=kernel_size,
                deploy=False, use_sync_bn=use_sync_bn,
                ptgu_gamma=ptgu_gamma, use_ptgu=use_ptgu)

        self.proj = nn.Sequential(
            get_bn(dim, use_sync_bn),
            nn.Conv2d(dim, out_dim, kernel_size=1))

        self.ls1 = LayerScale(out_dim, init_value=layer_scale_init_value) \
                   if layer_scale_init_value and layer_scale_init_value > 0 \
                   else nn.Identity()

        if num_frames > 0:
            self.dwconv2 = ResDWConv3d(out_dim, num_frames)
        else:
            self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = LayerNorm2d(out_dim)

        ffn_dim = int(ffn_factor * dim)
        mlp_dw = (ResDWConv3d(ffn_dim, num_frames)
                  if num_frames > 0 else ResDWConv(ffn_dim, kernel_size=3))
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, ffn_dim, kernel_size=1),
            nn.GELU(),
            mlp_dw,
            GRN(ffn_dim),
            nn.Conv2d(ffn_dim, out_dim, kernel_size=1))

        self.ls2 = LayerScale(out_dim, init_value=layer_scale_init_value) \
                   if layer_scale_init_value and layer_scale_init_value > 0 \
                   else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _sdpa_spatial(self, query_src, key_src, x, H, W):
        Q = self.weight_q(query_src)
        K = self.weight_k(key_src)
        V = x

        if self.joint_st_attn:
            # [FIX 11] Joint spatio-temporal attention: tokens are the
            # T*H*W positions of the whole clip. Attention can directly
            # relate position (i, j) at frame t to position (i', j') at
            # frame t' -- object tracking by attention (UniFormer SABlock).
            # Temporal position information is carried by the features
            # (temporal_embed of FIX 3, TOB injections) and by the zero-
            # padded 3D DW convs of FIX 10 (DPE argument).
            T = self.num_frames
            Q_s = rearrange(Q, '(b t) (g c) h w -> b g (t h w) c',
                            t=T, g=self.num_heads)
            K_s = rearrange(K, '(b t) (g c) h w -> b g (t h w) c',
                            t=T, g=self.num_heads)
            V_s = rearrange(V, '(b t) (g c) h w -> b g (t h w) c',
                            t=T, g=self.num_heads)
            x_attn = F.scaled_dot_product_attention(Q_s, K_s, V_s)
            return rearrange(x_attn, 'b g (t h w) c -> (b t) (g c) h w',
                             t=T, h=H, w=W)

        Q_s = rearrange(Q, 'b (g c) h w -> b g (h w) c', g=self.num_heads)
        K_s = rearrange(K, 'b (g c) h w -> b g (h w) c', g=self.num_heads)
        V_s = rearrange(V, 'b (g c) h w -> b g (h w) c', g=self.num_heads)

        x_attn = F.scaled_dot_product_attention(Q_s, K_s, V_s)
        return rearrange(x_attn, 'b g (h w) c -> b (g c) h w', h=H, w=W)

    def _forward_inner(self, x, h_x, h_r):
        B, C, H, W = x.shape
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_cat = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv_ctx(x_cat)
        identity = x_f

        # norm_ctx before temporal branch (FIX #4)
        x_f = self.norm_ctx(x_f)

        # [FIX 2] x_t computed FIRST -- before spatial SDPA -- so the spatial
        # branch can be enriched with temporal context.
        x_t = self.temporal_branch(x_f) if self.temporal_branch is not None else None

        query_src, key_src = torch.split(x_f, [C, self.ctx_ch], dim=1)

        x = self.fusion(x_f)

        # [FIX 12] Temporally-informed gate: the gate modulates x_agg, which
        # contains x_t; computing it from x + x_t prevents a time-blind gate
        # from suppressing the temporal signal.
        if x_t is not None and self.temporal_gate:
            gate = self.gate(x + x_t)
        else:
            gate = self.gate(x)
        lepe = self.lepe(x)

        # [FIX 2] T-to-S cascade: the spatial query source is enriched with
        # the temporal features (Q = weight_q(query_src + x_t)) so the
        # spatial attention queries from a temporally-informed perspective.
        # With t2s_cascade=False, the previous strictly-parallel behaviour is
        # reproduced (x_t contributes only as a residual to x_agg below).
        if x_t is not None and self.t2s_cascade:
            x_attn = self._sdpa_spatial(query_src + x_t, key_src, x, H, W)
        else:
            x_attn = self._sdpa_spatial(query_src, key_src, x, H, W)
        x_attn = self.dyconv_proj(x_attn)

        x_agg = x_attn + lepe
        if x_t is not None:
            x_agg = x_agg + x_t

        x_mixed = gate * self.se(x_agg)
        x_mixed = self.proj(x_mixed)

        # ls1 on x_mixed; init=1.0 (MODIF Prof)
        x = identity + self.drop_path(self.ls1(x_mixed))

        x = self.dwconv2(x)

        # ls2 on MLP output; init=1.0 (MODIF Prof)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return x, None
        l_x, h_x_new = torch.split(x, [self.dim, self.ctx_ch], dim=1)
        return l_x, h_x_new

    def forward(self, x, h_x, h_r):
        if self.with_cp and x.requires_grad:
            return cp.checkpoint(self._forward_inner, x, h_x, h_r,
                                 use_reentrant=False)
        return self._forward_inner(x, h_x, h_r)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

default_UniRepLKNet_A_F_P_kernel_sizes = ((3,3),(13,13),(13,13,13,13,13,13),(13,13))
default_UniRepLKNet_N_kernel_sizes = ((3,3),(13,13),(13,13,13,13,13,13,13,13),(13,13))
default_UniRepLKNet_T_kernel_sizes = (
    (3,3,3),(13,13,13),
    (13,3,13,3,13,3,13,3,13,3,13,3,13,3,13,3,13,3),(13,13,13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = (
    (3,3,3),(13,13,13),
    (13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3),(13,13,13))
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)
_default_ks = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes,
}


# ---------------------------------------------------------------------------
# VideoSTORM  [FIX 3: input temporal positional embedding]
# ---------------------------------------------------------------------------

class VideoSTORM(nn.Module):
    """VideoSTORM with all temporal reasoning fixes applied.

    [FIX 2] t2s_cascade (default True): the temporal->spatial cascade in
    STORMBlocks is now actually implemented (previous revisions documented
    it without implementing it). Set t2s_cascade=False to recover the
    strictly parallel composition for ablation.

    [FIX 3] temporal_embed : learnable absolute positional embedding of shape
    (num_frames, dims[0]), zero-initialized, injected into the feature map
    after the first downsample layer and before stage 0. This propagates
    temporal position information into all subsequent stages (including the
    TTB convolutions in STLKBlocks and the TOB attention in STORMBlocks).
    Interpolation handles T != num_frames at inference.

    [FIX 4 removed] TSM ablation variant: no temporal channel shift in
    STLKBlocks. Combined with FIX 8, the whole backbone is an exact
    identity to the pretrained 2D UniRepLKNet at step 0.

    [FIX 5 cancelled] The TOB attention keys are the UNMODIFIED context
    features: the context encoder guides the sub-stages, and altering it
    (former key_dw spatialization) caused a large performance loss.

    [FIX 6] Temporal relative position bias in TOB attention logits:
    zero-initialized (2T-1, heads) table, direct ordering signal stronger
    than absolute PEs alone.

    [FIX 8] TTB zero-gamma identity init (see TemporalTransitionBranch):
    the final BN of every TTB starts at gamma = 0, so the pretrained
    backbone function is exactly preserved at step 0 instead of being
    perturbed by unit-variance temporal noise in all 36 blocks.

    [FIX 9] PosTemporalGatingUnit in TOB (see PosTemporalGatingUnit):
    signed, position-based T x T temporal mixing with multiplicative gating,
    trunc_normal_-initialized (active from step 0). Provides the temporal
    differencing / order-detection operator class that the convex softmax
    SDPA cannot express. DISABLED by default (use_ptgu=False): the
    ablation showed no performance gain, plausibly because the signed
    paths (lepe, FIX 10 3D convs) and the multiplicative paths (gate,
    SE) already cover the operator class through depth composition.
    Set use_ptgu=True to restore the path.

    [FIX 10] 3D depthwise local convs in STORMBlocks (UniFormer-CBlock
    lesson, see DWConv3d): joint local spatio-temporal mixing in every
    sub-block, replacing the strictly spatial 2D DW 3x3 convs. Backbone
    untouched.

    [FIX 11] Joint spatio-temporal SDPA in STORMBlocks (UniFormer-SABlock
    lesson): attention over the joint T*H*W token set. Policies:
    joint_attn_s2 ('all' by default: every block of sub_blocks_s2) and
    joint_attn_s3 ('all' by default: every block of sub_blocks_s3, cheap
    at 7x7 resolution). Values: 'none' | 'alternate' | 'all'. Cost note:
    a joint block at stage-2 resolution costs ~T times the per-frame
    attention on that op; use 'alternate' to bound this if memory
    requires it.

    [FIX 12] Temporally-informed gate in STORMBlocks (temporal_gate=True):
    the multiplicative gate sees x + x_t instead of x alone, so it cannot
    silently suppress the temporal contribution it modulates.

    Checkpoint compatibility: temporal_embed is absent from the ImageNet-21K
    UniRepLKNet checkpoint and is initialized to zeros (identity additive
    operation at step 0). The strict=False load in load_pretrained_2d handles
    the missing key gracefully without modifying any pretrained spatial weight.
    rel_pos_bias_table and the PTGU parameters live in TOB (new
    modules, also absent from the checkpoint). The FIX 10 3D convs and the
    FIX 11/12 changes live entirely in the sub-blocks (new modules / no new
    parameters). The TTB zero-gamma init
    touches no pretrained key.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., kernel_sizes=None,
                 deploy=False, with_cp=False, attempt_use_lk_impl=True,
                 use_sync_bn=True, num_frames=8,
                 sub_depth=(9, 3), sub_num_heads=(4, 8), smk_size=5,
                 sub_drop_path_rate=0., sub_ffn_factor=2, projection=2048,
                 t2s_cascade=True, ptgu_gamma=8, use_ptgu=False,
                 joint_attn_s2='all', joint_attn_s3='all',
                 temporal_gate=True,
                 **kwargs):
        super().__init__()
        assert _EINOPS_AVAILABLE, '[VideoSTORM] einops required.'
        self.num_classes = num_classes
        self.num_frames = num_frames
        depths = tuple(depths)
        sub_depth = tuple(sub_depth)
        if kernel_sizes is None:
            if depths in _default_ks:
                kernel_sizes = _default_ks[depths]
            else:
                raise ValueError('Provide kernel_sizes for non-standard depths.')
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i]
        dp_rates = [x.item() for x in
                    torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format='channels_first'),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i+1], eps=1e-6, data_format='channels_first')))

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                SpatioTemporalLargeKernelBlock(
                    dim=dims[i], kernel_size=kernel_sizes[i][j],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                    with_cp=with_cp, use_sync_bn=use_sync_bn,
                    num_frames=num_frames)
                for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        ctx_ch = dims[3] // 4
        self.context_encoder = nn.Conv2d(dims[3], ctx_ch, kernel_size=1)
        self.patch_embedx = RetroActiveBridge(dims[2], dims[3],
                                               use_sync_bn=use_sync_bn)

        sub_ks_s2 = kernel_sizes[2][0]
        sub_dpr = [x.item() for x in
                     torch.linspace(0, sub_drop_path_rate, sum(sub_depth))]

        sub_ls_init = 1.0

        # [FIX 11] Per-block joint-attention flags from the per-stage policy.
        joint_s2 = _joint_attn_flags(joint_attn_s2, sub_depth[0])
        joint_s3 = _joint_attn_flags(joint_attn_s3, sub_depth[1])

        self.sub_blocks_s2 = nn.ModuleList([
            SpatioTemporalOverarchingRetroactiveModulationBlock(
                dim=dims[2], kernel_size=sub_ks_s2,
                ctx_ch=ctx_ch, smk_size=smk_size,
                num_heads=sub_num_heads[0], drop_path=sub_dpr[i],
                layer_scale_init_value=sub_ls_init,
                deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                with_cp=with_cp, use_sync_bn=use_sync_bn,
                ffn_factor=sub_ffn_factor,
                is_first=(i == 0), is_last=False, num_frames=num_frames,
                res_scale=True,
                t2s_cascade=t2s_cascade, ptgu_gamma=ptgu_gamma,
                use_ptgu=use_ptgu,
                joint_st_attn=joint_s2[i], temporal_gate=temporal_gate)
            for i in range(sub_depth[0])])

        sub_ks_s3 = kernel_sizes[3][0]
        n_s3 = sub_depth[1]
        self.sub_blocks_s3 = nn.ModuleList([
            SpatioTemporalOverarchingRetroactiveModulationBlock(
                dim=dims[3], kernel_size=sub_ks_s3,
                ctx_ch=ctx_ch, smk_size=smk_size,
                num_heads=sub_num_heads[1], drop_path=sub_dpr[sub_depth[0] + i],
                layer_scale_init_value=sub_ls_init,
                deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                with_cp=with_cp, use_sync_bn=use_sync_bn,
                ffn_factor=sub_ffn_factor,
                is_first=False, is_last=(i == n_s3 - 1), num_frames=num_frames,
                res_scale=True,
                t2s_cascade=t2s_cascade, ptgu_gamma=ptgu_gamma,
                use_ptgu=use_ptgu,
                joint_st_attn=joint_s3[i], temporal_gate=temporal_gate)
            for i in range(n_s3)])

        head_in_dim = dims[3] + ctx_ch
        self.video_head = nn.Sequential(
            nn.Conv2d(head_in_dim, projection, kernel_size=1, bias=False),
            get_bn(projection, use_sync_bn),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(projection, num_classes))

        self.aux_head = nn.Sequential(
            get_bn(dims[3], use_sync_bn),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dims[3], num_classes))

        # [FIX 3] Learnable input temporal positional embedding.
        # Shape (num_frames, dims[0]); zero-initialized so the block behaves
        # identically to the pretrained 2D model at step 0.
        # Absent from the UniRepLKNet checkpoint -> loaded gracefully via strict=False.
        self.temporal_embed = nn.Parameter(torch.zeros(num_frames, dims[0]))

        # [FIX 7] Set by load_pretrained_2d: names of the model parameters
        # that were actually present in the 2D checkpoint. Ground truth for
        # the differential-LR param grouping (see pretrained_param_names()).
        self._pretrained_param_names = None

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # [FIX 10] nn.Conv3d added: the 3D DW convs of the sub-blocks follow
        # the same trunc_normal_ init as every other conv (active path,
        # randomly-initialized modules, no pretrained weight to protect).
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            # [FIX 8] BNs flagged _zero_gamma_init (TTB final norm) start at
            # gamma = 0 so the branch is an exact identity at step 0.
            if getattr(m, '_zero_gamma_init', False):
                nn.init.constant_(m.weight, 0.0)
            else:
                nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @staticmethod
    def no_weight_decay_keywords():
        """Parameters exempted from weight decay.

        Exemptions for 4-D affine params not covered by the 1-D rule:
        - ls1.weight, ls2.weight : residual scales in STORMBlock
        - x_scale.weight, h_scale.weight : context blending in BCR
        - .gamma, .beta : GRN / GRNwithNHWC affine params

        [FIX 1] pe_q, pe_k : temporal positional encodings in TOB
        [FIX 3] temporal_embed : input-level temporal PE in VideoSTORM
        [FIX 6] rel_pos_bias_table : temporal relative position bias in TOB
        [FIX 9] pos_table, pos_bias : PTGU positional mixing table and bias
        """
        return {
            'ls1.weight',
            'ls2.weight',
            'x_scale.weight',
            'h_scale.weight',
            '.gamma',
            '.beta',
            'pe_q',
            'pe_k',
            'temporal_embed',
            'rel_pos_bias_table',
            'pos_table',
            'pos_bias',
        }

    def pretrained_param_names(self):
        """[FIX 7] Names of parameters genuinely loaded from the 2D checkpoint.

        Ground truth for the differential-LR grouping: a parameter deserves
        LR * LR_BACKBONE_RATIO if and only if it was present in the IN-21K
        checkpoint. Prefix-based grouping (e.g. 'stages.*' -> backbone) is
        WRONG here: the randomly-initialized TTBs live inside stages.* and
        would be silently trained at 0.1x LR, crippling temporal learning.

        Primary path: the exact key set recorded by load_pretrained_2d.
        Fallback (no 2D checkpoint loaded): name-marker heuristic listing
        every module that does not exist in UniRepLKNet. The PTGU [FIX 9]
        and the 3D DW convs [FIX 10] live inside sub_blocks_s2 /
        sub_blocks_s3, already covered by the markers below.
        """
        if self._pretrained_param_names is not None:
            return self._pretrained_param_names
        new_markers = ('temporal_branch',
                       'sub_blocks_s2', 'sub_blocks_s3',
                       'context_encoder', 'patch_embedx',
                       'video_head', 'aux_head', 'temporal_embed')
        return {n for n, _ in self.named_parameters()
                if not any(m in n for m in new_markers)}

    def new_param_names(self):
        """[FIX 7] Names of randomly-initialized parameters (full LR)."""
        all_names = {n for n, _ in self.named_parameters()}
        return all_names - self.pretrained_param_names()

    def reparameterize_backbone(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()
            if isinstance(m, TemporalReceptiveBlock):
                m.merge_temporal_branches()

    def _forward_features_2d(self, x, B, T):
        """2D backbone forward with input temporal positional embedding.

        [FIX 3] temporal_embed is injected between downsample_layers[0] and
        stages[0]. Frames are reshaped to (B, T, C, H, W), the PE is broadcast-
        added, then collapsed back to (B*T, C, H, W) for 2D stage processing.
        Interpolation handles T != self.num_frames (test-time dense sampling).
        """
        # Stage 0 : downsample -> temporal PE -> blocks
        x = self.downsample_layers[0](x)           # (B*T, dims[0], H0, W0)
        _, C0, H0, W0 = x.shape

        # [FIX 3] Inject temporal PE
        pe = self.temporal_embed                    # (num_frames, dims[0])
        if pe.shape[0] != T:
            # Interpolate along the temporal axis for non-standard T
            pe = F.interpolate(
                pe.t().unsqueeze(0),                # (1, dims[0], num_frames)
                size=T, mode='linear', align_corners=False
            ).squeeze(0).t()                        # (T, dims[0])
        x = (x.view(B, T, C0, H0, W0)
               + pe.to(dtype=x.dtype).view(1, T, C0, 1, 1)
             ).view(B * T, C0, H0, W0)

        x = self.stages[0](x)

        # Stage 1
        x = self.stages[1](self.downsample_layers[1](x))

        # Stage 2 (backbone)
        x_s2 = self.stages[2](self.downsample_layers[2](x))

        # Stage 3 (backbone)
        x_s3 = self.stages[3](self.downsample_layers[3](x_s2))

        # STORMBlocks
        ctx_ori = self.context_encoder(x_s3)
        ctx_up = F.interpolate(ctx_ori, size=x_s2.shape[2:],
                                mode='bilinear', align_corners=False)
        x, ctx = x_s2, ctx_up
        for blk in self.sub_blocks_s2:
            x, ctx = blk(x, ctx, ctx_up)
        x, ctx = self.patch_embedx(x, ctx)
        for blk in self.sub_blocks_s3:
            x, ctx = blk(x, ctx, ctx_ori)
        return x, x_s3

    @staticmethod
    def _ensure_bcthw(x):
        if x.dim() != 5:
            raise ValueError(f'VideoSTORM expects 5-D input, got {x.dim()}-D.')
        if x.size(1) in (1, 3): return x
        if x.size(2) in (1, 3): return x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def forward_features(self, x):
        x = self._ensure_bcthw(x)
        B, C, T, H, W = x.shape
        # Pass B and T to _forward_features_2d for temporal PE injection
        feat_bt, ctx_bt = self._forward_features_2d(
            x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W), B, T)
        return feat_bt, ctx_bt, B, T

    def forward(self, x):
        feat_bt, ctx_bt, B, T = self.forward_features(x)

        main_logits = self.video_head(feat_bt).view(B, T, -1).mean(1)

        if self.training:
            aux_logits = self.aux_head(ctx_bt).view(B, T, -1).mean(1)
            return dict(main=main_logits, aux=aux_logits)

        return main_logits
        
    def load_pretrained_2d(self, ckpt, strict=False, skip_head=True,
                            map_location='cpu'):
        def _load(p):
            if isinstance(p, str) and p.startswith('http'):
                return torch.hub.load_state_dict_from_url(
                    p, map_location=map_location, check_hash=False)
            try:
                from yacs.config import CfgNode
                torch.serialization.add_safe_globals([CfgNode])
                return torch.load(p, map_location=map_location, weights_only=True)
            except Exception:
                return torch.load(p, map_location=map_location, weights_only=False)

        def _unwrap(raw):
            if not isinstance(raw, dict): return raw
            if 'model' in raw and isinstance(raw['model'], dict):
                return raw['model']
            if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
                return raw['state_dict']
            return raw

        def _strip(sd):
            return {k[len('module.'):]: v for k, v in sd.items()} \
                   if any(k.startswith('module.') for k in sd) else sd

        # Markers identifying temporal/new modules that must always train at full LR,
        # regardless of whether their weights happened to be shape-compatible with
        # the checkpoint. These modules did not exist in the ImageNet-21K backbone.
        TEMPORAL_MARKERS = (
            'temporal_branch',
            'temporal_embed',
            'sub_blocks_s2',
            'sub_blocks_s3',
            'context_encoder',
            'patch_embedx',
            'video_head',
            'aux_head',
        )

        src = ckpt if isinstance(ckpt, str) else '<dict>'
        print('=' * 80); print(f'[LOADING] {src}'); print('=' * 80)
        try:
            raw   = ckpt if isinstance(ckpt, dict) else _load(ckpt)
            state = _strip(_unwrap(raw))
            if not isinstance(state, dict):
                raise ValueError('Unsupported checkpoint format.')

            # Filter legacy head/norm keys (UniRepLKNet 2D classifier head)
            if skip_head:
                skip_keys = [k for k in state
                             if k.startswith('head.') or k.startswith('norm.')]
                if skip_keys:
                    print(f'[INFO] Filtering {len(skip_keys)} legacy head/norm keys.')
                state = {k: v for k, v in state.items()
                         if not (k.startswith('head.') or k.startswith('norm.'))}

            # Filter shape-mismatched keys before calling load_state_dict.
            # load_state_dict raises RuntimeError on any shape mismatch even with
            # strict=False; we must remove incompatible keys ourselves first.
            model_state   = self.state_dict()
            filtered      = {}
            skipped_head  = []
            skipped_shape = []

            for k, v in state.items():
                if k not in model_state:
                    # Key absent from current model architecture — skip silently.
                    continue
                if model_state[k].shape != v.shape:
                    # Separate head mismatches (num_classes change, always expected
                    # in cross-dataset fine-tuning) from temporal shape mismatches
                    # (num_frames change: kernel sizes, PE lengths, bias tables).
                    if any(h in k for h in ('video_head', 'aux_head', 'head.')):
                        skipped_head.append(
                            f'  {k}: ckpt {tuple(v.shape)} → model {tuple(model_state[k].shape)}')
                    else:
                        skipped_shape.append(
                            f'  {k}: ckpt {tuple(v.shape)} → model {tuple(model_state[k].shape)}')
                    continue
                filtered[k] = v

            if skipped_head:
                print(f'[INFO] Skipped {len(skipped_head)} head keys '
                      f'(num_classes mismatch — expected for cross-dataset fine-tuning):')
                for msg in skipped_head:
                    print(msg)

            if skipped_shape:
                print(f'[WARNING] Skipped {len(skipped_shape)} keys due to shape mismatch '
                      f'(likely num_frames change between checkpoint and current config):')
                for msg in skipped_shape:
                    print(msg)
                print('[WARNING] These parameters will be randomly re-initialized '
                      'and trained at full LR.')

            missing, unexpected = self.load_state_dict(filtered, strict=False)

            # [FIX 7] Build the ground-truth set of pretrained parameter names.
            # Rule: a parameter is "pretrained" (backbone LR) if and only if:
            #   (a) it was present in the checkpoint with a matching shape, AND
            #   (b) it does not belong to any temporal/new module.
            # Condition (b) is critical: temporal params that happen to be
            # shape-compatible (e.g. same-dim BN stats inside stages.*) must still
            # train at full LR — they were randomly initialized for the video task
            # and must not be handicapped by the backbone LR scale.
            param_names = {n for n, _ in self.named_parameters()}
            self._pretrained_param_names = (
                param_names & set(filtered.keys())
            ) - {
                n for n in param_names
                if any(m in n for m in TEMPORAL_MARKERS)
            }

            n_pretrained = len(self._pretrained_param_names)
            n_new        = len(param_names - self._pretrained_param_names)
            print(f'[FIX 7] Pretrained params (backbone LR): {n_pretrained} | '
                  f'New params (full LR): {n_new}')

            if missing:
                print(f'[INFO] Missing keys (absent from checkpoint, '
                      f'will use random init): {len(missing)}')
                print(f'  First 5: {missing[:5]}')
            if unexpected:
                print(f'[INFO] Unexpected keys (in checkpoint but not in model): '
                      f'{len(unexpected)}')
                print(f'  First 5: {unexpected[:5]}')

            print(f'[SUCCESS] {len(filtered)} parameters loaded from {src}')

        except FileNotFoundError as e:
            print(f'[ERROR] File not found: {src} -- {e}'); raise
        except Exception as e:
            print(f'[ERROR] {type(e).__name__}: {e}'); raise


def _cfg(crop_pct=0.9, **kwargs):
    import timm
    return {'num_classes': 1000, 'input_size': (3, 224, 224),
            'crop_pct': crop_pct, 'interpolation': 'bicubic',
            'mean': timm.data.IMAGENET_DEFAULT_MEAN,
            'std': timm.data.IMAGENET_DEFAULT_STD,
            'classifier': 'video_head', **kwargs}


def _pop_timm_args(kwargs):
    kwargs.pop('pretrained', False)
    for k in ('pretrained_cfg', 'checkpoint_path', 'features_only',
              'scriptable', 'exportable'):
        kwargs.pop(k, None)


@register_model
def videostorm_s(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth', (10, 4))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 4)
    kwargs.setdefault('t2s_cascade', True)
    kwargs.setdefault('ptgu_gamma', 8)
    kwargs.setdefault('use_ptgu', False)
    kwargs.setdefault('joint_attn_s2', 'all')
    kwargs.setdefault('joint_attn_s3', 'all')
    kwargs.setdefault('temporal_gate', True)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(96, 192, 384, 768),
        attempt_use_lk_impl=False, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


@register_model
def videostorm_b(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth', (10, 4))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 4)
    kwargs.setdefault('t2s_cascade', True)
    kwargs.setdefault('ptgu_gamma', 8)
    kwargs.setdefault('use_ptgu', False)
    kwargs.setdefault('joint_attn_s2', 'all')
    kwargs.setdefault('joint_attn_s3', 'all')
    kwargs.setdefault('temporal_gate', True)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(128, 256, 512, 1024),
        attempt_use_lk_impl=False, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


@register_model
def videostorm_l(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth', (10, 4))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 4)
    kwargs.setdefault('t2s_cascade', True)
    kwargs.setdefault('ptgu_gamma', 8)
    kwargs.setdefault('use_ptgu', False)
    kwargs.setdefault('joint_attn_s2', 'all')
    kwargs.setdefault('joint_attn_s3', 'all')
    kwargs.setdefault('temporal_gate', True)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(192, 384, 768, 1536),
        attempt_use_lk_impl=False, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model