import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model

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


# ---------------------------------------------------------------------------
# TRB branch config — full mirror of DilatedReparamBlock up to k=15
# ---------------------------------------------------------------------------

_TRB_BRANCH_CFG = {
    5:  ([3, 3],            [1, 2]),
    7:  ([5, 3, 3],         [1, 2, 3]),
    9:  ([5, 5, 3, 3],      [1, 2, 3, 4]),
    11: ([5, 5, 3, 3, 3],   [1, 2, 3, 4, 5]),
    13: ([5, 7, 3, 3, 3],   [1, 2, 3, 4, 5]),
    15: ([5, 7, 3, 3, 3],   [1, 2, 3, 5, 7]),
}


def _temporal_kernel_size_sovereign(num_frames):
    """Largest TRB-supported odd kernel <= num_frames."""
    if num_frames <= 0:
        return 3
    k_max = (num_frames - 1) // 2 * 2 + 1
    valid = [k for k in sorted(_TRB_BRANCH_CFG) if k <= k_max]
    return valid[-1] if valid else 3


# ---------------------------------------------------------------------------
# 2D large-kernel helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backbone modules — names preserved for ImageNet checkpoint compatibility
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
# TemporalReceptiveBlock: 1D mirror of DilatedReparamBlock, up to k=15
# ---------------------------------------------------------------------------

class TemporalReceptiveBlock(nn.Module):

    _BRANCH_CFG = _TRB_BRANCH_CFG

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False):
        super().__init__()
        if kernel_size not in self._BRANCH_CFG:
            raise ValueError(
                f'TemporalReceptiveBlock: kernel_size must be in '
                f'{sorted(self._BRANCH_CFG)}, got {kernel_size}.')
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
# TemporalTransitionBranch: sovereign k_t, reshape + 1D conv + BN
# ---------------------------------------------------------------------------

class TemporalTransitionBranch(nn.Module):
    """1D temporal branch for STLKBlock.

    k_t determined solely by num_frames (sovereign regime):
      num_frames=8  -> k_t=7
      num_frames>=16 -> k_t=15

    k_t >= 5 -> TemporalReceptiveBlock(k_t) + BN1d
    k_t == 3 -> Conv1d(k_t=3) + BN1d
    """

    def __init__(self, dim, num_frames, deploy=False, use_sync_bn=False):
        super().__init__()
        self.num_frames = num_frames
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        k_t = _temporal_kernel_size_sovereign(num_frames)
        self.k_t = k_t

        if k_t >= 5:
            self.dwconv_1d = TemporalReceptiveBlock(dim, k_t, deploy=deploy,
                                                    use_sync_bn=use_sync_bn)
        else:
            self.dwconv_1d = nn.Conv1d(dim, dim, kernel_size=k_t,
                                       padding=k_t // 2, groups=dim, bias=deploy)
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
# SpatioTemporalLargeKernelBlock = UniRepLKNetBlock + parallel temporal add
# ---------------------------------------------------------------------------

class SpatioTemporalLargeKernelBlock(nn.Module):
    """UniRepLKNetBlock with a single parallel temporal branch.

    Spatial path is an exact copy of UniRepLKNetBlock — unchanged.
    Temporal branch output is added directly (no learnable scale).
    """

    def __init__(self, dim, kernel_size, drop_path=0.,
                 layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=False, ffn_factor=4, num_frames=0):
        super().__init__()
        self.with_cp = with_cp

        # === Spatial path — identical to UniRepLKNetBlock ===
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

        # === Temporal branch — single parallel addition ===
        self.temporal_branch = None
        if num_frames > 0 and kernel_size != 0 and not deploy:
            self.temporal_branch = TemporalTransitionBranch(
                dim=dim, num_frames=num_frames,
                deploy=False, use_sync_bn=use_sync_bn)

    def compute_residual(self, x):
        """Exact copy of UniRepLKNetBlock.compute_residual."""
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        def _f(x):
            out = x + self.compute_residual(x)
            if self.temporal_branch is not None:
                out = out + self.temporal_branch(x)
            return out
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
# VideoSTORM — main model
# ---------------------------------------------------------------------------

class VideoSTORM(nn.Module):
    """VideoSTORM: UniRepLKNet backbone + parallel temporal branches.

    Exactly UniRepLKNet with a single TemporalTransitionBranch per block,
    added in parallel to the spatial residual. No sub-stages, no learnable
    temporal scale. Head mirrors UniRepLKNet: temporal mean + spatial GAP +
    LayerNorm + Linear.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., kernel_sizes=None,
                 deploy=False, with_cp=False, attempt_use_lk_impl=True,
                 use_sync_bn=False, num_frames=8, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in _default_ks:
                kernel_sizes = _default_ks[depths]
            else:
                raise ValueError('Provide kernel_sizes for non-standard depths.')
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i]
        dp_rates = [x.item() for x in
                    torch.linspace(0, drop_path_rate, sum(depths))]

        # Downsample layers — identical to UniRepLKNet
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

        # 4 stages — UniRepLKNetBlock replaced by STLKBlock (spatial identical,
        # temporal branch added in parallel)
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

        # Head — mirrors UniRepLKNet: LayerNorm + Linear
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
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
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x, B, T

    def forward(self, x):
        feat_bt, B, T = self.forward_features(x)
        _, C, H, W = feat_bt.shape
        # Temporal mean + spatial GAP + LayerNorm (mirrors UniRepLKNet)
        feat = feat_bt.view(B, T, C, H, W).mean(dim=1)   # (B, C, H, W)
        feat = self.norm(feat.mean([-2, -1]))               # (B, C)
        return self.head(feat)                              # (B, num_classes)

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
            if 'model' in raw and isinstance(raw['model'], dict): return raw['model']
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
            'classifier': 'head', **kwargs}


def _pop_timm_args(kwargs):
    kwargs.pop('pretrained', False)
    for k in ('pretrained_cfg', 'checkpoint_path', 'features_only',
              'scriptable', 'exportable'):
        kwargs.pop(k, None)


@register_model
def videostorm_s(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    _pop_timm_args(kwargs)
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
    print('VideoSTORM v4 — UniRepLKNet + parallel temporal')