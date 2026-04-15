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


def get_bn(dim, use_sync_bn=False):
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
    """Largest odd kernel <= num_frames, capped at 9."""
    k = (num_frames - 1) // 2 * 2 + 1
    return max(min(k, 9), 3)


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size,
                         padding=kernel_size // 2, groups=dim)

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
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


# ---------------------------------------------------------------------------
# Backbone modules -- names preserved for ImageNet checkpoint compatibility
# ---------------------------------------------------------------------------

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
    """SE block matching OverLoCK's design: GELU activation, dim//8 reduction.

    Used in STORMBlock (sub-stage) where weights are freshly initialized.
    SEBlock (ReLU, dim//4) is kept for backbone blocks to preserve pretrained
    weight compatibility with UniRepLKNet.
    """
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
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False,
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
# TemporalReceptiveBlock  (1D mirror of DilatedReparamBlock)
# ---------------------------------------------------------------------------

class TemporalReceptiveBlock(nn.Module):
    """Exact 1D mirror of DilatedReparamBlock.

    Large-kernel Conv1d origin + N dilated Conv1d branches, merged at
    reparameterization time. Supported kernel sizes: 5, 7, 9.
    """

    _BRANCH_CFG = {
        9: ([5, 5, 3, 3], [1, 2, 3, 4]),
        7: ([5, 3, 3],    [1, 2, 3]),
        5: ([3, 3],       [1, 2]),
    }

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False):
        super().__init__()
        if kernel_size not in self._BRANCH_CFG:
            raise ValueError(
                f'TemporalReceptiveBlock requires kernel_size in '
                f'{list(self._BRANCH_CFG)}, got {kernel_size}.')
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
# TemporalTransitionBranch  (temporal for SpatioTemporalLargeKernelBlock)
# ---------------------------------------------------------------------------

class TemporalTransitionBranch(nn.Module):
    """1D temporal mirror of SpatioTemporalLargeKernelBlock's (dwconv + norm) path.

    Unified k_t regime: the temporal kernel size is determined solely by
    num_frames, independently of the spatial kernel_size. All active blocks
    (kernel_size != 0) use the same temporal receptive field regardless of
    whether their spatial path uses ks=3, ks=5, or ks>=7.

      k_t >= 5  ->  TemporalReceptiveBlock(k_t) + BN1d   (reparameterizable)
      k_t == 3  ->  Conv1d(k_t=3) + BN1d                 (short clip fallback)

    Pivot: (B*T, C, H, W)  <->  (B*H*W, C, T).
    """

    def __init__(self, dim, kernel_size, num_frames,
                 deploy=False, use_sync_bn=False):
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
# TemporalOverarchingBranch  (temporal mirror for STORMBlock)
# ---------------------------------------------------------------------------

class TemporalOverarchingBranch(nn.Module):
    """1D temporal mirror of STORMBlock's spatial attention path.

    Option A: receives x_cat = dwconv_ctx(concat(x, h_x)) BEFORE norm_ctx,
    preserving direct access to the context signal h_x.

    Input: x_cat  (B*T, out_dim, H, W)  where out_dim = dim + ctx_ch.

    Structural mirrors (one-to-one with the spatial path):
      Split x_cat_1d -> x_1d (dim) + h_1d (ctx_ch)
        mirrors: query_src, key_src = split(norm_ctx(x_cat))
      lepe_1d     : TemporalReceptiveBlock(k_t) + BN1d  on x_1d
        mirrors: lepe = DilatedReparamBlock(dim, ks) + BN2d  on x (after fusion)
      weight_q_1d : Conv1d(dim, dim//2, 1) + BN1d  on x_1d
        mirrors: weight_q = Conv2d(dim, dim//2, 1) + BN2d  on query_src
      weight_k_1d : Conv1d(ctx_ch, dim//2, 1) + BN1d  on h_1d  (full T kept)
        mirrors: weight_k = AdaptiveAvgPool2d(7) + Conv2d(ctx_ch, dim//2, 1) + BN2d
        Note: no pooling along T.  K must vary across frames so na1d_qk
        produces distinct attention weights per frame.  Pooling to length 1
        makes K constant along T, which causes uniform softmax and zero
        gradient on weight_q_1d weights (degenerate average pooling).
      na1d_qk + na1d_av  local neighbourhood cross-attention along T
        mirrors: na2d_av  local neighbourhood attention along H x W
      dyconv_proj_1d : Conv1d(dim, dim, 1) + BN1d
        mirrors: dyconv_proj

    gate_1d is intentionally absent.  x_t is gated by the spatial gate in
    STORMBlock after spatio-temporal aggregation.  A second gate inside
    this branch would double-gate x_t while x_attn + lepe are gated only once,
    dampening temporal gradients and creating an asymmetric contribution.

    Output: x_attn_1d + lepe_1d  shape (B*T, dim, H, W).

    NATTEN 1D is a hard requirement. RuntimeError is raised at instantiation
    if na1d_qk / na1d_av are unavailable.
    """

    def __init__(self, dim, ctx_ch, num_heads, num_frames, spatial_kernel_size,
                 deploy=False, use_sync_bn=False):
        super().__init__()

        if not _NATTEN1D_AVAILABLE:
            raise RuntimeError(
                '[VideoSTORM] na1d_qk / na1d_av not found in natten.functional. '
                'TemporalOverarchingBranch requires NATTEN compiled with 1D support '
                '(NATTEN >= 0.17 with SM90 for H100). '
                'Recompile NATTEN or set num_frames=0 to disable temporal branches.')

        self.num_frames = num_frames
        self.dim = dim
        self.ctx_ch = ctx_ch
        self.num_heads = num_heads
        self.head_dim_qk = dim // (2 * num_heads)
        self.head_dim_v  = dim // num_heads
        self.scale = self.head_dim_qk ** -0.5
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
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            BN1d(dim // 2))

        self.weight_k_1d = nn.Sequential(
            nn.Conv1d(ctx_ch, dim // 2, kernel_size=1, bias=False),
            BN1d(dim // 2))

        self.dyconv_proj_1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            BN1d(dim))

        self.rpb_1d = nn.Parameter(torch.zeros(num_heads, 2 * k_t - 1))

    @staticmethod
    def _to_1d(x, B, T, H, W):
        """(B*T, C, H, W) -> (B*H*W, C, T)."""
        return (x.view(B, T, x.size(1), H, W)
                  .permute(0, 3, 4, 2, 1).contiguous()
                  .view(B * H * W, x.size(1), T))

    @staticmethod
    def _from_1d(x_1d, B, T, H, W, C):
        """(B*H*W, C, T) -> (B*T, C, H, W)."""
        return (x_1d.view(B, H, W, C, T)
                    .permute(0, 4, 3, 1, 2).contiguous()
                    .view(B * T, C, H, W))

    def _apply_rpb_1d(self, attn, T):
        """Apply relative positional bias with correct per-position indexing."""
        k = self.k_t
        dev = attn.device
        idx_t = torch.arange(0, k, device=dev)
        num_repeat = torch.ones(k, dtype=torch.long, device=dev)
        num_repeat[k // 2] = T - (k - 1)
        bias_idx = idx_t.repeat_interleave(num_repeat)
        bias_idx = torch.flip(bias_idx, [0])
        rpb = self.rpb_1d[:, bias_idx].unsqueeze(0).unsqueeze(-1)
        return attn + rpb

    def _temporal_attention(self, x_1d, h_1d):
        """Local neighbourhood cross-attention along T."""
        BHW, C, T = x_1d.shape

        Q = self.weight_q_1d(x_1d) * self.scale
        Q = Q.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()

        K = self.weight_k_1d(h_1d)
        K = K.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()

        V = x_1d.view(BHW, self.num_heads, self.head_dim_v, T).permute(0, 1, 3, 2).contiguous()

        attn = _na1d_qk(Q, K, kernel_size=self.k_t)
        attn = self._apply_rpb_1d(attn, T)
        attn = torch.softmax(attn, dim=-1)
        out  = _na1d_av(attn, V, kernel_size=self.k_t)

        return out.permute(0, 1, 3, 2).contiguous().view(BHW, C, T)

    def forward(self, x_cat):
        """x_cat: (B*T, out_dim, H, W) = dwconv_ctx output before norm_ctx."""
        BT, out_dim, H, W = x_cat.shape
        T = self.num_frames
        B = BT // T

        x_cat_1d = self._to_1d(x_cat, B, T, H, W)

        x_1d, h_1d = torch.split(x_cat_1d, [self.dim, self.ctx_ch], dim=1)

        lepe   = self.lepe_1d(x_1d)
        x_attn = self._temporal_attention(x_1d, h_1d)
        x_attn = self.dyconv_proj_1d(x_attn)

        x_t = x_attn + lepe

        return self._from_1d(x_t, B, T, H, W, self.dim)

    def reparameterize(self):
        if isinstance(self.lepe_1d[0], TemporalReceptiveBlock):
            self.lepe_1d[0].merge_temporal_branches()


# ---------------------------------------------------------------------------
# SpatioTemporalLargeKernelBlock  (unified k_t regime, no learnable fusion gate)
# ---------------------------------------------------------------------------

#STLKBlock
class SpatioTemporalLargeKernelBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=0.,
                 layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=False, ffn_factor=4, num_frames=0):
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

        self.temporal_branch = None
        if num_frames > 0 and kernel_size != 0 and not deploy:
            self.temporal_branch = TemporalTransitionBranch(
                dim=dim, kernel_size=kernel_size,
                num_frames=num_frames,
                deploy=False, use_sync_bn=use_sync_bn)

    def compute_residual(self, x):
        v_s = self.norm(self.dwconv(x))
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
# Sub-stage modules
# ---------------------------------------------------------------------------

class RetroActiveBridge(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim))
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4))

    def forward(self, x, ctx):
        return self.x_proj(x), self.h_proj(ctx)


# ---------------------------------------------------------------------------
# SpatioTemporalOverarchingRetroactiveModulationBlock
# (Option A, res_scale=True aligned with OverLoCK)
# ---------------------------------------------------------------------------

#STORMBlock
class SpatioTemporalOverarchingRetroactiveModulationBlock(nn.Module):
    def __init__(self, dim, kernel_size, ctx_ch, smk_size=5, num_heads=4,
                 drop_path=0., layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False, use_sync_bn=False,
                 ffn_factor=2, is_first=False, is_last=False, num_frames=0,
                 res_scale=True):
        super().__init__()
        assert _NATTEN_AVAILABLE and _EINOPS_AVAILABLE
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        self.scale = (dim // self.num_heads) ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.res_scale = res_scale
        self.with_cp = with_cp
        self.dim = dim
        self.ctx_ch = ctx_ch
        out_dim = dim + ctx_ch

        if not is_first:
            self.x_scale = LayerScale(ctx_ch, init_value=1)
            self.h_scale = LayerScale(ctx_ch, init_value=1)

        self.dwconv_ctx = ResDWConv(out_dim, kernel_size=3)
        self.norm_ctx = LayerNorm2d(out_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,
                      groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim))

        self.weight_q = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2))
        self.weight_k = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_ch, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2))
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim))

        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                use_sync_bn=use_sync_bn,
                                attempt_use_lk_impl=attempt_use_lk_impl),
            nn.BatchNorm2d(dim))
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU())
        self.se = SEModule(dim)

        self.temporal_branch = None
        if num_frames > 0 and not deploy:
            self.temporal_branch = TemporalOverarchingBranch(
                dim=dim, ctx_ch=ctx_ch,
                num_heads=self.num_heads,
                num_frames=num_frames,
                spatial_kernel_size=kernel_size,
                deploy=False, use_sync_bn=use_sync_bn)

        # FIX 2: proj always outputs out_dim — no is_last special case.
        # Context channels flow through to the classification head,
        # matching OverLoCK where the last DynamicConvBlock returns out_dim.
        self.proj = nn.Sequential(nn.BatchNorm2d(dim),
                                  nn.Conv2d(dim, out_dim, kernel_size=1))

        # FIX 2 (cont): all downstream layers uniformly use out_dim.
        # Previously is_last used dim, creating an inconsistency that
        # discarded context channels at the last block.
        self.ls1 = LayerScale(out_dim, init_value=layer_scale_init_value) \
                   if layer_scale_init_value and layer_scale_init_value > 0 \
                   else nn.Identity()

        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = LayerNorm2d(out_dim)

        ffn_dim = int(ffn_factor * dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, ffn_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(ffn_dim, kernel_size=3),
            GRN(ffn_dim),
            nn.Conv2d(ffn_dim, out_dim, kernel_size=1))

        self.ls2 = LayerScale(out_dim, init_value=layer_scale_init_value) \
                   if layer_scale_init_value and layer_scale_init_value > 0 \
                   else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.rpb1 = nn.Parameter(torch.zeros(
            self.num_heads, 2 * smk_size - 1, 2 * smk_size - 1))
        self.rpb2 = nn.Parameter(torch.zeros(
            self.num_heads, 2 * kernel_size - 1, 2 * kernel_size - 1))

    @torch.no_grad()
    def _generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        return idx_h, idx_w, ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)

    def _apply_rpb(self, attn, rpb, H, W, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = H - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = W - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1)
                    * (2 * kernel_size - 1)
                    + idx_w.repeat_interleave(num_repeat_w))
        bias_idx = torch.flip(
            (bias_hw.unsqueeze(-1) + idx_k).reshape(-1, int(kernel_size**2)), [0])
        return attn + torch.flatten(rpb, 1, 2)[:, bias_idx].reshape(
            1, int(self.num_heads), int(H), int(W), int(kernel_size**2))

    def _forward_inner(self, x, h_x, h_r):
        B, C, H, W = x.shape
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_cat = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv_ctx(x_cat)
        identity = x_f

        # Temporal branch (Option A): x_f before norm_ctx.
        x_t = self.temporal_branch(x_f) if self.temporal_branch is not None else None

        x_f = self.norm_ctx(x_f)
        query_src, key_src = torch.split(x_f, [C, self.ctx_ch], dim=1)

        x = self.fusion(x_f)

        # Spatial attention path
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = min(H, W) < self.kernel_size
        if is_pad:
            size = ((self.kernel_size, int(self.kernel_size / H * W))
                     if H < W else (int(self.kernel_size / W * H), self.kernel_size))
            x_pad = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            H_a, W_a = size
        else:
            x_pad, H_a, W_a = x, H, W

        q_src = F.interpolate(query_src, size=(H_a, W_a),
                              mode='bilinear', align_corners=False) if is_pad else query_src

        Q = self.weight_q(q_src) * self.scale
        K = self.weight_k(key_src)

        Q = rearrange(Q, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        K = rearrange(K, 'b (g c) h w -> b g c (h w)', g=self.num_heads)

        A = einops_einsum(Q, K, 'b g c n, b g c l -> b g n l')
        D = self.weight_proj(rearrange(A, 'b g n l -> b l g n').contiguous())
        D = rearrange(D, 'b l g (h w) -> b g h w l', h=H_a, w=W_a)

        attn1, attn2 = torch.split(D, [self.smk_size**2, self.kernel_size**2], dim=-1)

        attn1 = torch.softmax(self._apply_rpb(attn1, self.rpb1, H_a, W_a,
                              self.smk_size, *self._generate_idx(self.smk_size)), dim=-1)
        attn2 = torch.softmax(self._apply_rpb(attn2, self.rpb2, H_a, W_a,
                              self.kernel_size, *self._generate_idx(self.kernel_size)), dim=-1)

        value = rearrange(x_pad, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        x_attn = rearrange(
            torch.cat([na2d_av(attn1, value[0], kernel_size=self.smk_size),
                       na2d_av(attn2, value[1], kernel_size=self.kernel_size)], dim=1),
            'b g h w c -> b (g c) h w', h=H_a, w=W_a)

        if is_pad:
            x_attn = F.adaptive_avg_pool2d(x_attn, (H, W))

        x_attn = self.dyconv_proj(x_attn)

        # Spatio-temporal aggregation then SE then gate (OverLoCK order)
        x_agg = x_attn + lepe
        if x_t is not None:
            x_agg = x_agg + x_t

        x_mixed = gate * self.se(x_agg)
        x_mixed = self.proj(x_mixed)

        # FIX 1+3: residual always on identity (fused x+h_x after dwconv_ctx).
        # FIX 1: res_scale=True places LayerScale on the skip connection,
        # matching OverLoCK's residual pattern where the identity can be
        # attenuated to let new features dominate.
        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x_mixed)
        else:
            x = identity + self.drop_path(self.ls1(x_mixed))

        x = self.dwconv2(x)

        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        # is_last returns full out_dim tensor (dim + ctx_ch channels preserved)
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
# VideoSTORM
# ---------------------------------------------------------------------------

class VideoSTORM(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., kernel_sizes=None,
                 deploy=False, with_cp=False, attempt_use_lk_impl=True,
                 use_sync_bn=False, num_frames=8,
                 sub_depth=(4, 2), sub_num_heads=(4, 8), smk_size=5,
                 sub_drop_path_rate=0., sub_ffn_factor=2, projection=2048,
                 **kwargs):
        super().__init__()
        assert _NATTEN_AVAILABLE and _EINOPS_AVAILABLE
        if num_frames > 0:
            assert _NATTEN1D_AVAILABLE, (
                '[VideoSTORM] NATTEN 1D support (na1d_qk / na1d_av) is required '
                'when num_frames > 0. Recompile NATTEN with SM90 support.')
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
        self.patch_embedx = RetroActiveBridge(dims[2], dims[3])

        sub_ks_s2 = kernel_sizes[2][0]
        sub_dpr = [x.item() for x in
                     torch.linspace(0, sub_drop_path_rate, sum(sub_depth))]

        sub_ls_init = 1.0

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
                res_scale=True)
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
                res_scale=True)
            for i in range(n_s3)])

        # FIX 3: video_head accepts out_dim = dims[3] + ctx_ch.
        # Matches OverLoCK's fusion_dim = embed_dim[-1] + embed_dim[-1]//4.
        # Context channels now flow through the is_last block to the head.
        head_in_dim = dims[3] + ctx_ch
        self.video_head = nn.Sequential(
            nn.Conv2d(head_in_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(projection, num_classes))

        # Auxiliary head on raw stage-4 features (before sub-stages).
        self.aux_head = nn.Sequential(
            nn.BatchNorm2d(dims[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dims[3], num_classes))

        self.apply(self._init_weights)
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def reparameterize_backbone(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()
            if isinstance(m, TemporalReceptiveBlock):
                m.merge_temporal_branches()

    def _forward_features_2d(self, x):
        for i in range(2):
            x = self.stages[i](self.downsample_layers[i](x))
        x_s2 = self.stages[2](self.downsample_layers[2](x))
        x_s3 = self.stages[3](self.downsample_layers[3](x_s2))
        ctx_ori = self.context_encoder(x_s3)
        ctx_up = F.interpolate(ctx_ori, size=x_s2.shape[2:],
                                mode='bilinear', align_corners=False)
        x, ctx = x_s2, ctx_up
        for blk in self.sub_blocks_s2:
            x, ctx = blk(x, ctx, ctx_up)
        x, ctx = self.patch_embedx(x, ctx)
        for blk in self.sub_blocks_s3:
            x, ctx = blk(x, ctx, ctx_ori)
        # x has out_dim = dims[3] + ctx_ch channels (is_last returns full tensor)
        # x_s3 returned for auxiliary supervision
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
        feat_bt, ctx_bt = self._forward_features_2d(
            x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W))
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
                return torch.hub.load_state_dict_from_url(p, map_location=map_location,
                                                           check_hash=False)
            try:
                return torch.load(p, map_location=map_location, weights_only=True)
            except TypeError:
                return torch.load(p, map_location=map_location)

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

        src = ckpt if isinstance(ckpt, str) else '<dict>'
        print('=' * 80); print(f'[LOADING] {src}'); print('=' * 80)
        try:
            raw = ckpt if isinstance(ckpt, dict) else _load(ckpt)
            state = _strip(_unwrap(raw))
            if not isinstance(state, dict):
                raise ValueError('Unsupported checkpoint format.')
            if skip_head:
                skip_keys = [k for k in state
                             if k.startswith('head.') or k.startswith('norm.')]
                if skip_keys:
                    print(f'[INFO] Filtering {len(skip_keys)} head/norm keys.')
                state = {k: v for k, v in state.items()
                         if not (k.startswith('head.') or k.startswith('norm.'))}
            missing, unexpected = self.load_state_dict(state, strict=strict)
            if missing or unexpected:
                print(f'[WARNING] Missing: {len(missing)} | Unexpected: {len(unexpected)}')
                if missing: print(f' Missing (first 10): {missing[:10]}')
                if unexpected: print(f' Unexpected (first 10): {unexpected[:10]}')
            else:
                print(f'[SUCCESS] {len(state)} parameters loaded.')
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
    kwargs.setdefault('sub_depth', (4, 2))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 2)
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
    kwargs.setdefault('sub_depth', (4, 2))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 4)
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
    kwargs.setdefault('sub_depth', (4, 2))
    kwargs.setdefault('sub_num_heads', (4, 8))
    kwargs.setdefault('smk_size', 5)
    kwargs.setdefault('projection', 2048)
    kwargs.setdefault('sub_ffn_factor', 2)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(192, 384, 768, 1536),
        attempt_use_lk_impl=False, **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


if __name__ == '__main__':
    print('VideoSTORM v2 — res_scale + is_last fix aligned with OverLoCK')