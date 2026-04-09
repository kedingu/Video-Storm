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
# 1D reparameterization helpers  (mirrors of the 2D equivalents above)
# ---------------------------------------------------------------------------

def fuse_bn_1d(conv, bn):
    """Fuse Conv1d + BN1d into a single Conv1d with bias."""
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1),
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)


def convert_dilated_to_nondilated_1d(kernel, dilate_rate):
    """Expand a dilated depthwise Conv1d kernel to its non-dilated equivalent."""
    identity_kernel = torch.ones((1, 1, 1), device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose1d(kernel, identity_kernel, stride=dilate_rate)
    return torch.cat([F.conv_transpose1d(kernel[:, i:i+1], identity_kernel,
                                         stride=dilate_rate)
                      for i in range(kernel.size(1))], dim=1)


def merge_dilated_into_large_kernel_1d(large_kernel, dilated_kernel, dilated_r):
    """Accumulate a dilated 1D branch into the large-kernel equivalent."""
    large_k = large_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_kernel.size(2) - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated_1d(dilated_kernel, dilated_r)
    pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [pad, pad])


def _temporal_kernel_size(num_frames):
    """Largest odd kernel <= num_frames, capped at 9 for FLOPs budget."""
    k = (num_frames - 1) // 2 * 2 + 1   # largest odd number <= num_frames
    return max(min(k, 9), 3)             # in [3, 9], always odd


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------

class ResDWConv(nn.Conv2d):
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size,
                         padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class TemporalResDWConv(nn.Module):
    """Residual depthwise Conv1d along T. Pivot: (B*T,C,H,W) <-> (B*H*W,C,T).
    Kept for backward-compatibility; no longer used inside DynamicSTORMBlock."""
    def __init__(self, dim, num_frames, kernel_size=3):
        super().__init__()
        self.num_frames = num_frames
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=dim, bias=False)

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T
        x_1d = (x.view(B, T, C, H, W)
                  .permute(0, 3, 4, 2, 1).contiguous()
                  .view(B * H * W, C, T))
        x_1d = x_1d + self.conv(x_1d)
        return (x_1d.view(B, H, W, C, T)
                    .permute(0, 4, 3, 1, 2).contiguous()
                    .view(BT, C, H, W))


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
# 1D reparameterizable block  (temporal mirror of DilatedReparamBlock)
# ---------------------------------------------------------------------------

class TemporalReceptiveBlock(nn.Module):
    """Exact 1D mirror of DilatedReparamBlock.

    Same philosophy: one large-kernel Conv1d 'origin' + N dilated Conv1d
    branches that are merged at reparameterization time.
    Supported kernel sizes: 5, 7, 9  (chosen via _temporal_kernel_size).
    """

    # Branch configs mirroring the 2D counterparts
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
        # x: (N, C, T)
        if not hasattr(self, 'origin_bn'):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            out = out + self.__getattr__(f'dil_bn_k{k}_{r}')(
                            self.__getattr__(f'dil_conv_k{k}_{r}')(x))
        return out

    def merge_temporal_branches(self):
        """Fuse all branches into lk_origin (in-place). Sets deploy mode."""
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
# Temporal mirror for UniRepLKNetBlock
# ---------------------------------------------------------------------------

class TemporalTransitionBranch(nn.Module):
    """1D temporal mirror of UniRepLKNetBlock's (dwconv + norm) path.

    Three regimes matching the spatial side:
      kernel_size == 0  -> not instantiated (handled by the parent block)
      kernel_size in (3, 5) -> simple Conv1d + BN1d
      kernel_size >= 7  -> TemporalReceptiveBlock + BN1d  (with reparameterization)

    Pivot: (B*T, C, H, W)  <->  (B*H*W, C, T).
    Zero-initialised gate in the parent block ensures ImageNet-21K weights are
    preserved at the start of fine-tuning.
    """

    def __init__(self, dim, kernel_size, num_frames,
                 deploy=False, use_sync_bn=False):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        k_t = _temporal_kernel_size(num_frames)
        self.k_t = k_t

        if kernel_size >= 7:
            if k_t >= 5:
                self.dwconv_1d = TemporalReceptiveBlock(dim, k_t, deploy=deploy,
                                                        use_sync_bn=use_sync_bn)
            else:
                # Fallback for very short clips (k_t == 3)
                self.dwconv_1d = nn.Conv1d(dim, dim, kernel_size=k_t,
                                           padding=k_t // 2, groups=dim,
                                           bias=deploy)
        else:
            # kernel_size in (3, 5): simple Conv1d with matching temporal kernel
            k = min(k_t, kernel_size)
            self.dwconv_1d = nn.Conv1d(dim, dim, kernel_size=k,
                                        padding=k // 2, groups=dim,
                                        bias=deploy)

        self.norm_1d = nn.Identity() if deploy else BN1d(dim)

    # -- pivot helpers -------------------------------------------------------

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

    # -- reparameterization --------------------------------------------------

    def reparameterize(self):
        """Merge 1D dilated branches and fuse outer BN1d (deploy mode)."""
        # Step 1: merge dilated branches inside TemporalReceptiveBlock
        if isinstance(self.dwconv_1d, TemporalReceptiveBlock):
            self.dwconv_1d.merge_temporal_branches()

        # Step 2: fuse outer norm_1d (BN1d) into the conv
        if isinstance(self.norm_1d, (nn.BatchNorm1d, nn.SyncBatchNorm)):
            # Retrieve the Conv1d to fuse into
            if isinstance(self.dwconv_1d, TemporalReceptiveBlock):
                conv = self.dwconv_1d.lk_origin   # already has bias after merge
            else:
                conv = self.dwconv_1d
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
# Temporal mirror for DynamicSTORMBlock
# ---------------------------------------------------------------------------

class TemporalOverarchingBranch(nn.Module):
    """1D temporal mirror of DynamicSTORMBlock's spatial attention path.

    Option B: operates on x AFTER fusion (dim channels), no ctx needed.
    Mirrors the three structural elements of the spatial path that feed SE:
      (1) lepe_1d   : TemporalReceptiveBlock + BN1d  (mirror of lepe)
      (2) attn_1d   : na1d local self-attention      (mirror of na2d_av)
      (3) gate_1d   : Conv1d + BN1d + SiLU           (mirror of gate)
    Output: gate_1d * (x_attn_1d + lepe_1d), shape (B*T, C, H, W).

    Pivot: (B*T, C, H, W)  <->  (B*H*W, C, T).
    Zero-initialised gate in the parent block ensures spatial pretrained
    weights are preserved at the start of fine-tuning.
    """

    def __init__(self, dim, num_heads, num_frames, spatial_kernel_size,
                 deploy=False, use_sync_bn=False):
        super().__init__()
        self.num_frames = num_frames
        self.num_heads = num_heads                      # already doubled by parent
        self.head_dim_qk = dim // (2 * num_heads)       # mirrors weight_q/weight_k
        self.head_dim_v  = dim // num_heads             # full head dim for values
        self.scale = self.head_dim_qk ** -0.5
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        k_t = _temporal_kernel_size(num_frames)
        self.k_t = k_t

        # (1) lepe_1d: mirror of  DilatedReparamBlock(dim, kernel_size) + BN2d
        if k_t >= 5:
            lepe_conv = TemporalReceptiveBlock(dim, k_t, deploy=deploy,
                                              use_sync_bn=use_sync_bn)
        else:
            lepe_conv = nn.Conv1d(dim, dim, kernel_size=k_t,
                                  padding=k_t // 2, groups=dim,
                                  bias=deploy)
        self.lepe_1d = nn.Sequential(lepe_conv, BN1d(dim))

        # (2a) Q, K projections for local self-attention along T
        #      mirrors weight_q (dim -> dim//2) and weight_k (dim -> dim//2)
        self.weight_q_1d = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            BN1d(dim // 2))
        self.weight_k_1d = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=1, bias=False),
            BN1d(dim // 2))

        # (2b) dyconv_proj_1d: mirror of dyconv_proj (Conv2d + BN2d)
        self.dyconv_proj_1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            BN1d(dim))

        # Learnable relative positional bias (1D counterpart of rpb1/rpb2)
        self.rpb_1d = nn.Parameter(torch.zeros(num_heads, 2 * k_t - 1))

        # (3) gate_1d: mirror of  Conv2d(dim,dim,1) + BN2d + SiLU
        self.gate_1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, bias=False),
            BN1d(dim),
            nn.SiLU())

    # -- pivot helpers -------------------------------------------------------

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

    # -- RPB helper ----------------------------------------------------------

    def _apply_rpb_1d(self, attn):
        # attn: (BHW, heads, T, k_t)
        # rpb_1d: (heads, 2*k_t - 1)  — select center slice of length k_t
        half = self.k_t // 2
        rpb_vals = self.rpb_1d[:, half:half + self.k_t]     # (heads, k_t)
        return attn + rpb_vals.unsqueeze(0).unsqueeze(2)     # broadcast over BHW, T

    # -- attention along T ---------------------------------------------------

    def _temporal_attention(self, x_1d):
        # x_1d: (BHW, C, T)
        BHW, C, T = x_1d.shape

        # Q, K: (BHW, dim//2, T)
        Q = self.weight_q_1d(x_1d) * self.scale
        K = self.weight_k_1d(x_1d)

        # Reshape to (BHW, heads, T, head_dim_qk)
        Q = Q.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()
        K = K.view(BHW, self.num_heads, self.head_dim_qk, T).permute(0, 1, 3, 2).contiguous()

        # V: full C channels -> (BHW, heads, T, head_dim_v)
        V = x_1d.view(BHW, self.num_heads, self.head_dim_v, T).permute(0, 1, 3, 2).contiguous()

        if _NATTEN1D_AVAILABLE and _na1d_qk is not None:
            # Local neighbourhood attention along T
            attn = _na1d_qk(Q, K, kernel_size=self.k_t)          # (BHW, heads, T, k_t)
            attn = self._apply_rpb_1d(attn)
            attn = torch.softmax(attn, dim=-1)
            out = _na1d_av(attn, V, kernel_size=self.k_t)         # (BHW, heads, T, head_dim_v)
        else:
            # Fallback: global self-attention (acceptable for small T)
            attn = torch.einsum('bhnd,bhmd->bhnm', Q, K)          # (BHW, heads, T, T)
            attn = torch.softmax(attn, dim=-1)
            out = torch.einsum('bhnm,bhmd->bhnd', attn, V)        # (BHW, heads, T, head_dim_v)

        # (BHW, heads, T, head_dim_v) -> (BHW, C, T)
        return out.permute(0, 1, 3, 2).contiguous().view(BHW, C, T)

    # -- forward -------------------------------------------------------------

    def forward(self, x):
        """x: (B*T, C, H, W) after fusion -- Option B."""
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T

        x_1d = self._to_1d(x, B, T, H, W)    # (B*H*W, C, T)

        lepe = self.lepe_1d(x_1d)             # (B*H*W, C, T)
        gate = self.gate_1d(x_1d)             # (B*H*W, C, T)
        x_attn = self._temporal_attention(x_1d)
        x_attn = self.dyconv_proj_1d(x_attn)  # (B*H*W, C, T)

        x_t = gate * (x_attn + lepe)          # (B*H*W, C, T)

        return self._from_1d(x_t, B, T, H, W, C)   # (B*T, C, H, W)

    def reparameterize(self):
        """Merge 1D dilated branches inside lepe_1d."""
        if isinstance(self.lepe_1d[0], TemporalReceptiveBlock):
            self.lepe_1d[0].merge_temporal_branches()


# ---------------------------------------------------------------------------
# AdaptiveTemporalConv  (kept for reference, replaced by TemporalTransitionBranch)
# ---------------------------------------------------------------------------

class AdaptiveTemporalConv(nn.Module):
    """Legacy temporal branch -- superseded by TemporalTransitionBranch."""

    @staticmethod
    def _compute_dilations(num_frames):
        dilations = [1]
        d = 2
        while (2 * d + 1) <= num_frames:
            dilations.append(d)
            d *= 2
        d_terminal = (num_frames - 1) // 2
        if d_terminal > dilations[-1]:
            dilations.append(d_terminal)
        return dilations

    def __init__(self, dim, kernel_size, num_frames, use_sync_bn=False):
        super().__init__()
        self.num_frames = num_frames
        self.kernel_size = kernel_size
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        if kernel_size >= 7:
            self.dilations = self._compute_dilations(num_frames)
            for d in self.dilations:
                self.__setattr__(f't_dil_conv_d{d}',
                    nn.Conv1d(dim, dim, kernel_size=3, padding=d,
                              dilation=d, groups=dim, bias=False))
                self.__setattr__(f't_dil_bn_d{d}', BN1d(dim))
        elif kernel_size in (3, 5):
            self.t_dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                      padding=kernel_size // 2,
                                      groups=dim, bias=False)
            self.t_norm = BN1d(dim)

    def forward(self, x):
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T
        x_1d = (x.view(B, T, C, H, W)
                  .permute(0, 3, 4, 2, 1).contiguous()
                  .view(B * H * W, C, T))
        if self.kernel_size >= 7:
            out = sum(self.__getattr__(f't_dil_bn_d{d}')(
                          self.__getattr__(f't_dil_conv_d{d}')(x_1d))
                      for d in self.dilations)
        elif self.kernel_size in (3, 5):
            out = self.t_norm(self.t_dwconv(x_1d))
        else:
            out = x_1d
        return (out.view(B, H, W, C, T)
                   .permute(0, 4, 3, 1, 2).contiguous()
                   .view(BT, C, H, W))


# ---------------------------------------------------------------------------
# UniRepLKNetBlock  (updated: TemporalTransitionBranch replaces AdaptiveTemporalConv)
# ---------------------------------------------------------------------------

class UniRepLKNetBlock(nn.Module):
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

        # --- temporal branch (1D mirror of dwconv + norm) ---
        # No learnable gate: direct addition consistently outperforms gated fusion.
        self.temporal_branch = None
        if num_frames > 0 and kernel_size != 0 and not deploy:
            self.temporal_branch = TemporalTransitionBranch(
                dim=dim, kernel_size=kernel_size,
                num_frames=num_frames,
                deploy=False, use_sync_bn=use_sync_bn)

    def compute_residual(self, x):
        # Spatial path
        v_s = self.norm(self.dwconv(x))

        # Temporal path: 1D mirror of (dwconv + norm), direct addition before SE
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
        # Reparameterize spatial DilatedReparamBlock
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

        # Reparameterize temporal branch
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
# DynamicSTORMBlock  (updated: single TemporalOverarchingBranch fused before SE)
# ---------------------------------------------------------------------------

class DynamicSTORMBlock(nn.Module):
    def __init__(self, dim, kernel_size, ctx_ch, smk_size=5, num_heads=4,
                 drop_path=0., layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False, use_sync_bn=False,
                 ffn_factor=2, is_first=False, is_last=False, num_frames=0):
        super().__init__()
        assert _NATTEN_AVAILABLE and _EINOPS_AVAILABLE
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        self.scale = (dim // self.num_heads) ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.with_cp = with_cp
        self.dim = dim
        self.ctx_ch = ctx_ch
        out_dim = dim + ctx_ch

        if not is_first:
            self.x_scale = LayerScale(ctx_ch, init_value=1)
            self.h_scale = LayerScale(ctx_ch, init_value=1)

        # Spatio-temporal context injection (spatial only; temporal handled by branch)
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
        self.se = SEBlock(dim, dim // 4)

        # --- temporal attention branch (1D mirror of spatial attention path) ---
        # Operates on x after fusion (Option B, dim channels).
        # No learnable gate: TemporalOverarchingBranch already contains its own
        # gate_1d (Conv1d+BN1d+SiLU) which controls its internal output magnitude;
        # an additional scalar gate at fusion level is redundant and empirically
        # suboptimal vs direct addition.
        self.temporal_branch = None
        if num_frames > 0 and not deploy:
            self.temporal_branch = TemporalOverarchingBranch(
                dim=dim, num_heads=self.num_heads,
                num_frames=num_frames, spatial_kernel_size=kernel_size,
                deploy=False, use_sync_bn=use_sync_bn)

        if not is_last:
            self.proj = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.Conv2d(dim, out_dim, kernel_size=1))
        else:
            self.proj = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.Conv2d(dim, dim, kernel_size=1))

        res_dim = out_dim if not is_last else dim
        self.ls1 = LayerScale(res_dim, init_value=layer_scale_init_value) \
                   if layer_scale_init_value and layer_scale_init_value > 0 \
                   else nn.Identity()

        ffn_in = res_dim
        self.dwconv2 = ResDWConv(ffn_in, kernel_size=3)
        self.norm2 = LayerNorm2d(ffn_in)

        ffn_dim = int(ffn_factor * ffn_in)
        self.mlp = nn.Sequential(
            nn.Conv2d(ffn_in, ffn_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(ffn_dim, kernel_size=3),
            GRN(ffn_dim),
            nn.Conv2d(ffn_dim, ffn_in, kernel_size=1))

        self.ls2 = LayerScale(ffn_in, init_value=layer_scale_init_value) \
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

        orig_x = x

        x_cat = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv_ctx(x_cat)
        # norm_ctx applied on pure spatial output (no scattered temporal here)
        identity = x_f
        x_f = self.norm_ctx(x_f)

        query_src, key_src = torch.split(x_f, [C, self.ctx_ch], dim=1)

        x = self.fusion(x_f)      # (B, dim, H, W) -- Option B pivot point

        # --- spatial attention path ---
        gate    = self.gate(x)
        lepe    = self.lepe(x)

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

        # --- spatio-temporal aggregation then SE then gate (OverLoCK order) ---
        # Step 1: aggregate spatial (attn + lepe) and temporal contributions
        x_agg = x_attn + lepe
        if self.temporal_branch is not None:
            x_agg = x_agg + self.temporal_branch(x)

        # Step 2: SE recalibrates channels on the full spatio-temporal aggregate
        # Step 3: gate controls output magnitude after SE (matches OverLoCK design)
        x_mixed = gate * self.se(x_agg)

        x_mixed = self.proj(x_mixed)

        if self.is_last:
            x = orig_x + self.drop_path(self.ls1(x_mixed))
        else:
            x = identity + self.drop_path(self.ls1(x_mixed))

        x = self.dwconv2(x)
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
                UniRepLKNetBlock(
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
        self.sub_blocks_s2 = nn.ModuleList([
            DynamicSTORMBlock(
                dim=dims[2], kernel_size=sub_ks_s2,
                ctx_ch=ctx_ch, smk_size=smk_size,
                num_heads=sub_num_heads[0], drop_path=sub_dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                with_cp=with_cp, use_sync_bn=use_sync_bn,
                ffn_factor=sub_ffn_factor,
                is_first=(i == 0), is_last=False, num_frames=num_frames)
            for i in range(sub_depth[0])])

        sub_ks_s3 = kernel_sizes[3][0]
        n_s3 = sub_depth[1]
        self.sub_blocks_s3 = nn.ModuleList([
            DynamicSTORMBlock(
                dim=dims[3], kernel_size=sub_ks_s3,
                ctx_ch=ctx_ch, smk_size=smk_size,
                num_heads=sub_num_heads[1], drop_path=sub_dpr[sub_depth[0] + i],
                layer_scale_init_value=layer_scale_init_value,
                deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                with_cp=with_cp, use_sync_bn=use_sync_bn,
                ffn_factor=sub_ffn_factor,
                is_first=False, is_last=(i == n_s3 - 1), num_frames=num_frames)
            for i in range(n_s3)])

        self.video_head = nn.Sequential(
            nn.Conv2d(dims[3], projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(projection, num_classes))

        self.apply(self._init_weights)
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
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
        return x

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
        feat_bt = self._forward_features_2d(
            x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W))
        return feat_bt, B, T

    def forward(self, x):
        feat_bt, B, T = self.forward_features(x)
        return self.video_head(feat_bt).view(B, T, -1).mean(1)

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
    print('VideoSTORM')