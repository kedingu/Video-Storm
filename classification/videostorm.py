"""
Video-STORM: Spatio-Temporal Overarching Retro-active Modulation for Video Recognition.

Pipeline (frame-wise on B*T frames):
    downsample_layers[0-1] + stages[0-1]        -- low/mid-level features
    downsample_layers[2]   + stages[2] -> x_s2  -- mid-level features (dims[2] ch.)
    downsample_layers[3]   + stages[3] -> x_s3  -- high-level features (dims[3] ch.)
    context_encoder(x_s3)             -> ctx_ori -- overarching context prior
    interpolate(ctx_ori, x_s2)        -> ctx_up  -- upsampled context prior
    sub_blocks_s2 x sub_depth[0]  DynamicSTORMBlock (dim=dims[2])
    patch_embedx   RetroActiveBridge  (dims[2] -> dims[3], stride=2)
    sub_blocks_s3 x sub_depth[1]  DynamicSTORMBlock (dim=dims[3])
    video_head (Conv1x1->BN->SiLU->AdaptiveAvgPool->Linear)

Spatio-temporal modeling (all stages, k != 0):
    AdaptiveTemporalConv runs alongside the spatial conv+norm pipeline.
    Dilation schedule adapts to num_frames via geometric progression.
    Fusion via parameter-free RMSNorm before the shared SE recalibration.

ImageNet checkpoint compatibility:
    stages[0-3] and downsample_layers load directly (module names preserved).
    head.* and norm.* keys are filtered; sub-stage and temporal keys are absent
    from ImageNet checkpoints and initialised from scratch.

Overflow prevention (IN-22K gamma/beta preservation):
    normalize_block_bn() scales BN gamma/beta proportionally at load time instead
    of resetting them to 1/0. DilatedReparamBlock BN gammas are divided by the
    number of branches (N) so their sum has controlled magnitude. block.norm and
    FFN BN gammas are max-normalized to bound |gamma| <= 1. This preserves the
    relative channel importance structure learned on IN-22K while preventing fp16
    overflow. Zero runtime memory cost, zero forward path modification.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IGEMM_CHECKED  = False
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
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

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
# Utility modules
# ---------------------------------------------------------------------------

class RMSNorm2d(nn.Module):
    """Parameter-free RMS normalisation over the channel dimension of (N,C,H,W) tensors.
    Used to equalise magnitudes between ImageNet-pretrained spatial features and
    randomly-initialised temporal features before their additive fusion [AMP-safe]."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms.to(x.dtype)


class GRN(nn.Module):
    """Global Response Normalisation [NCHW] from ConvNeXt V2 [Woo et al., CVPR 2023].
    Applied in the context fusion module of DynamicSTORMBlock."""
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


# ---------------------------------------------------------------------------
# Core spatial blocks -- module names preserved for ImageNet checkpoint compat.
# ---------------------------------------------------------------------------

class GRNwithNHWC(nn.Module):
    """GRN [NHWC] with parameter-free LayerNorm pre-normalisation for fp16 stability."""
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        x_n = F.layer_norm(x, [x.shape[-1]])
        Gx  = torch.norm(x_n, p=2, dim=(1, 2), keepdim=True)
        Nx  = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return (self.gamma * Nx + 1) * x + self.beta if self.use_bias \
               else (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x): return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x): return x.permute(0, 3, 1, 2)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels-last and channels-first layouts.
    Uses .contiguous() after permute to prevent DDP gradient stride warnings."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
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
    """Squeeze-and-Excitation channel recalibration [Hu et al., CVPR 2018], NCHW."""
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, bias=True)
        self.up   = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        return inputs * torch.sigmoid(x).view(-1, self.input_channels, 1, 1)


class DilatedReparamBlock(nn.Module):
    """Multi-scale dilated depthwise convolution with structural reparameterisation.
    At inference, all branches are merged into a single large-kernel conv.
    Module names are frozen for ImageNet checkpoint compatibility."""
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
            self.kernel_sizes = [5, 5, 3, 3];    self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3];        self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3];           self.dilates = [1, 2]
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
        merged.bias.data   = origin_b
        self.lk_origin = merged
        self.__delattr__('origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__delattr__(f'dil_conv_k{k}_{r}')
            self.__delattr__(f'dil_bn_k{k}_{r}')


class UniRepLKNetBlock(nn.Module):
    """Spatial building block with large-kernel depthwise convolution, SE recalibration,
    and a GRN-FFN. Module names (dwconv, norm, se, pwconv1, act, pwconv2, gamma,
    drop_path) are frozen for ImageNet checkpoint compatibility. When num_frames > 0,
    an AdaptiveTemporalConv branch is fused with the spatial features via parameter-free
    RMSNorm before the shared SE, enabling spatio-temporal recalibration at every scale.

    Overflow prevention is handled at load time by normalize_block_bn() which scales
    BN gamma/beta proportionally (division by N for DilatedReparamBlock branches,
    max-normalization for block.norm and FFN BN). Forward path is unmodified --
    zero runtime overhead."""
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
        self.se   = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act     = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=not deploy))
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

        # Temporal branch -- absent from ImageNet checkpoints, initialised from scratch.
        # Not created for k=0 (Identity has no convolution to mirror) or deploy mode.
        self.temporal_branch = None
        if num_frames > 0 and kernel_size != 0 and not deploy:
            self.temporal_branch = AdaptiveTemporalConv(
                dim=dim, kernel_size=kernel_size,
                num_frames=num_frames, use_sync_bn=use_sync_bn)
            self.rms_norm = RMSNorm2d()

    def compute_residual(self, x):
        v_s = self.norm(self.dwconv(x))                    # spatial branch
        if self.temporal_branch is not None:
            v_t = self.temporal_branch(x)                  # temporal branch
            y = self.se(self.rms_norm(v_s) + self.rms_norm(v_t))
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
                conv.bias.data   = (self.norm.bias
                                    - self.norm.running_mean * self.norm.weight / std)
                self.dwconv = conv
            self.norm = nn.Identity()
        final_scale = self.gamma.data if self.gamma is not None else 1
        self.gamma  = None
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn  = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = (
                linear.weight * (bn.weight / std * final_scale).view(-1, 1))
            lb = (0 if linear.bias is None else linear.bias.data) + grn_bias_projected
            new_linear.bias.data = (
                (bn.bias + (lb - bn.running_mean) * bn.weight / std) * final_scale)
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


# ---------------------------------------------------------------------------
# Video-STORM contributions
# ---------------------------------------------------------------------------

class AdaptiveTemporalConv(nn.Module):
    """Adaptive 1D temporal convolution branch for Video-STORM.
    Mirrors the spatial conv+norm pipeline in the temporal dimension using depthwise
    Conv1d. The dilation schedule is computed automatically from num_frames via a
    geometric progression (doubling) plus a terminal branch, achieving near-complete
    temporal coverage at any T (88% at T=8, 94% at T=16, 97% at T=32, 98% at T=64).
    Pivot: (B*T, C, H, W) <-> (B*H*W, C, T)."""

    @staticmethod
    def _compute_dilations(num_frames: int):
        # Geometric doubling until RF = 2d+1 exceeds num_frames; terminal branch
        # at (num_frames-1)//2 to tightly cover the sequence.
        dilations = [1]
        d = 2
        while (2 * d + 1) <= num_frames:
            dilations.append(d)
            d *= 2
        d_terminal = (num_frames - 1) // 2
        if d_terminal > dilations[-1]:
            dilations.append(d_terminal)
        return dilations

    def __init__(self, dim: int, kernel_size: int, num_frames: int,
                 use_sync_bn: bool = False):
        super().__init__()
        self.num_frames  = num_frames
        self.kernel_size = kernel_size
        BN1d = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm1d

        if kernel_size >= 7:
            self.dilations = self._compute_dilations(num_frames)
            for d in self.dilations:
                self.__setattr__(f't_dil_conv_d{d}',
                    nn.Conv1d(dim, dim, kernel_size=3, padding=d,
                              dilation=d, groups=dim, bias=False))
                self.__setattr__(f't_dil_bn_d{d}', BN1d(dim))
            self.t_norm = BN1d(dim)
        elif kernel_size in (3, 5):
            self.t_dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                      padding=kernel_size // 2,
                                      groups=dim, bias=False)
            self.t_norm   = BN1d(dim)
        else:
            self.t_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T

        # (B*T, C, H, W) -> (B*H*W, C, T)
        x_1d = (x.view(B, T, C, H, W)
                  .permute(0, 3, 4, 2, 1)
                  .contiguous()
                  .view(B * H * W, C, T))

        if self.kernel_size >= 7:
            out = sum(self.__getattr__(f't_dil_bn_d{d}')(
                          self.__getattr__(f't_dil_conv_d{d}')(x_1d))
                      for d in self.dilations)
        elif self.kernel_size in (3, 5):
            out = self.t_dwconv(x_1d)
        else:
            out = x_1d

        out = self.t_norm(out)

        # (B*H*W, C, T) -> (B*T, C, H, W)
        return (out.view(B, H, W, C, T)
                   .permute(0, 4, 3, 1, 2)
                   .contiguous()
                   .view(BT, C, H, W))


class RetroActiveBridge(nn.Module):
    """Joint stride-2 downsampling of spatial features x and context prior ctx.

    Implements the 'retro-active' mechanism of Video-STORM: the high-level context
    prior is propagated downward and spatially aligned with mid-level features,
    enabling DynamicSTORMBlock to modulate earlier representations with overarching
    semantic information from stage 3."""
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


class DynamicSTORMBlock(nn.Module):
    """Core building block of Video-STORM sub-stages implementing retro-active
    spatio-temporal modulation. Context is injected via concatenation-fusion
    (ResidualDWConv3x3 on cat[x, h_x]) for rich spatial-context interaction.
    ContMix computes token-wise dynamic kernels from Q (fused features) and K
    (context prior pooled to 7x7 region centers). A spatio-temporal LEPE fuses
    spatial and adaptive temporal branches via RMSNorm addition. Per-channel
    alpha/beta blending modulates the context flow. proj->split extracts the
    updated context at zero extra parameter cost. FFN operates on the full
    out_dim = dim+ctx_ch, giving direct access to the updated context.

    Interface: forward(x, h_x, h_r) -> (x_out, h_x_new)
      x:   (N, dim,    H, W)  spatial features Z_i
      h_x: (N, ctx_ch, H, W)  context prior P_i  (updated each block)
      h_r: (N, ctx_ch, H, W)  context reference P_o (fixed anchor)"""

    def __init__(self, dim, kernel_size, ctx_ch, smk_size=5, num_heads=4,
                 drop_path=0., layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False, use_sync_bn=False,
                 ffn_factor=2, is_first=False, is_last=False, num_frames=0):
        super().__init__()
        assert _NATTEN_AVAILABLE,  'DynamicSTORMBlock requires natten.'
        assert _EINOPS_AVAILABLE,  'DynamicSTORMBlock requires einops.'

        self.kernel_size = kernel_size
        self.smk_size    = smk_size
        self.num_heads   = num_heads * 2        # G: half large-k, half small-k
        self.scale       = (dim // self.num_heads) ** -0.5
        self.is_first    = is_first
        self.is_last     = is_last
        self.with_cp     = with_cp
        self.dim         = dim
        self.ctx_ch      = ctx_ch
        out_dim          = dim + ctx_ch

        # Context injection -- ResidualDWConv3x3 on cat[x, h_x] then fusion to dim
        self.dwconv_ctx = nn.Conv2d(out_dim, out_dim, kernel_size=3,
                                    padding=1, groups=out_dim, bias=True)
        self.norm_ctx   = LayerNorm(out_dim, data_format='channels_first')
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,
                      groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim))

        # Per-channel context blending: P_{i+1} = alpha * P'_i + beta * P_o
        if not is_first:
            self.alpha = nn.Parameter(torch.ones(1, ctx_ch, 1, 1))
            self.beta  = nn.Parameter(torch.ones(1, ctx_ch, 1, 1))

        # Spatial LEPE: DilatedReparamBlock + BN (Q computed from fused x)
        self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                          use_sync_bn=use_sync_bn,
                                          attempt_use_lk_impl=attempt_use_lk_impl)
        self.norm = get_bn(dim, use_sync_bn)

        # Temporal LEPE: AdaptiveTemporalConv fused with spatial LEPE via RMSNorm
        self.t_lepe = None
        if num_frames > 0 and not deploy:
            self.t_lepe   = AdaptiveTemporalConv(dim=dim, kernel_size=kernel_size,
                                                  num_frames=num_frames,
                                                  use_sync_bn=use_sync_bn)
            self.lepe_rms = RMSNorm2d()

        # ContMix attention -- Q from fused x, K from raw h_x (S=7 context centers)
        self.weight_q    = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2))
        self.weight_k    = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_ch, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2))
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim))

        # Feature-level guidance gate and SE recalibration
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU())
        self.se = SEBlock(dim, dim // 4)

        # proj -> out_dim (non-last) or dim (last) to avoid DDP unused-param error
        if not is_last:
            self.proj = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.Conv2d(dim, out_dim, kernel_size=1))
        else:
            self.proj = nn.Sequential(nn.BatchNorm2d(dim),
                                      nn.Conv2d(dim, dim, kernel_size=1))

        # FFN on out_dim (non-last) or dim (last) -- operates on context-enriched features
        ffn_in  = out_dim if not is_last else dim
        ffn_dim = int(ffn_factor * ffn_in)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(ffn_in, ffn_dim))
        self.act     = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=True))
        self.pwconv2 = nn.Sequential(
            nn.Linear(ffn_dim, ffn_in, bias=False),
            NHWCtoNCHW(),
            get_bn(ffn_in, use_sync_bn))

        self.gamma = (nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
                      if layer_scale_init_value is not None
                         and layer_scale_init_value > 0 else None)
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
        bias_hw  = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1)
                    * (2 * kernel_size - 1)
                    + idx_w.repeat_interleave(num_repeat_w))
        bias_idx = torch.flip(
            (bias_hw.unsqueeze(-1) + idx_k).reshape(-1, int(kernel_size**2)), [0])
        return attn + torch.flatten(rpb, 1, 2)[:, bias_idx].reshape(
            1, int(self.num_heads), int(H), int(W), int(kernel_size**2))

    def _forward_inner(self, x, h_x, h_r):
        B, C, H, W = x.shape

        # Per-channel retro-active context blending
        if not self.is_first:
            h_x = self.alpha * h_x + self.beta * h_r

        # Concatenation-fusion context injection
        x_cat = torch.cat([x, h_x], dim=1)
        x_f   = self.norm_ctx(x_cat + self.dwconv_ctx(x_cat))
        x     = self.fusion(x_f)                           # (N, dim, H, W)

        # ContMix -- dual-scale neighbourhood attention (Q from x, K from h_x)
        is_pad = min(H, W) < self.kernel_size
        if is_pad:
            size  = ((self.kernel_size, int(self.kernel_size / H * W))
                     if H < W else (int(self.kernel_size / W * H), self.kernel_size))
            x_pad = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            H_a, W_a = size
        else:
            x_pad, H_a, W_a = x, H, W

        Q = self.weight_q(x_pad) * self.scale
        K = self.weight_k(h_x)
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

        value  = rearrange(x_pad, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        x_attn = rearrange(
            torch.cat([na2d_av(attn1, value[0], kernel_size=self.smk_size),
                       na2d_av(attn2, value[1], kernel_size=self.kernel_size)], dim=1),
            'b g h w c -> b (g c) h w', h=H_a, w=W_a)
        if is_pad:
            x_attn = F.adaptive_avg_pool2d(x_attn, (H, W))
        x_attn = self.dyconv_proj(x_attn)

        # Spatio-temporal LEPE -- spatial + adaptive temporal, fused via RMSNorm
        lepe_s = self.norm(self.dwconv(x))
        if self.t_lepe is not None:
            lepe_t = self.t_lepe(x)
            lepe   = self.lepe_rms(lepe_s) + self.lepe_rms(lepe_t)
        else:
            lepe = lepe_s

        # Gated aggregation + SE recalibration on spatio-temporal features
        x_mixed = self.se(self.gate(x) * (x_attn + lepe))
        if self.gamma is not None:
            x_mixed = self.gamma.view(1, -1, 1, 1) * x_mixed
        x = x + self.drop_path(x_mixed)

        # proj + FFN -- context-enriched feature path
        x = self.proj(x)
        x = x + self.drop_path(self.pwconv2(self.act(self.pwconv1(x))))

        # Split: context update is free (no additional projection needed)
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
# Depth / kernel-size presets
# ---------------------------------------------------------------------------

default_UniRepLKNet_A_F_P_kernel_sizes    = ((3,3),(13,13),(13,13,13,13,13,13),(13,13))
default_UniRepLKNet_N_kernel_sizes        = ((3,3),(13,13),(13,13,13,13,13,13,13,13),(13,13))
default_UniRepLKNet_T_kernel_sizes        = (
    (3,3,3),(13,13,13),
    (13,3,13,3,13,3,13,3,13,3,13,3,13,3,13,3,13,3),(13,13,13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = (
    (3,3,3),(13,13,13),
    (13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3,13,3,3),(13,13,13))

UniRepLKNet_A_F_P_depths    = (2, 2, 6,  2)
UniRepLKNet_N_depths         = (2, 2, 8,  2)
UniRepLKNet_T_depths         = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths  = (3, 3, 27, 3)

_default_ks = {
    UniRepLKNet_A_F_P_depths:    default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths:         default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths:         default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths:  default_UniRepLKNet_S_B_L_XL_kernel_sizes,
}


# ---------------------------------------------------------------------------
# VideoSTORM -- main model
# ---------------------------------------------------------------------------

class VideoSTORM(nn.Module):
    """Video-STORM: Spatio-Temporal Overarching Retro-active Modulation.
    Four-stage convolutional backbone pretrained on ImageNet with AdaptiveTemporalConv
    added to every stage block (k != 0). A context prior extracted from stage-3 features
    is retro-actively propagated to DynamicSTORMBlock sub-stages via RetroActiveBridge,
    enabling overarching semantic guidance at intermediate resolutions.
    All parameters participate in the forward pass (no find_unused_parameters).
    Requires: natten, einops."""

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
        assert _NATTEN_AVAILABLE, 'VideoSTORM requires natten.'
        assert _EINOPS_AVAILABLE, 'VideoSTORM requires einops.'

        self.num_classes = num_classes
        self.num_frames  = num_frames
        depths    = tuple(depths)
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

        # Stem + stage downsampling (module names preserved for checkpoint compat.)
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

        # Stages 0-3 -- AdaptiveTemporalConv activated per block when k!=0
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

        # Overarching context encoder: projects stage-3 features to ctx_ch
        ctx_ch = dims[3] // 4
        self.context_encoder = nn.Conv2d(dims[3], ctx_ch, kernel_size=1)

        # RetroActiveBridge: aligns context with stage-2 resolution
        self.patch_embedx = RetroActiveBridge(dims[2], dims[3])

        # Sub-stage s2 -- dynamic modulation at stage-2 resolution
        sub_ks_s2 = kernel_sizes[2][0]
        sub_dpr   = [x.item() for x in
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

        # Sub-stage s3 -- dynamic modulation at stage-3 resolution
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

        # Classification head
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

    def _forward_features_2d(self, x):
        # Stages 0-1 -- low/mid-level spatio-temporal features
        for i in range(2):
            x = self.stages[i](self.downsample_layers[i](x))

        # Stage 2 -- mid-level features retained for sub-stage s2
        x_s2 = self.stages[2](self.downsample_layers[2](x))

        # Stage 3 -- high-level overarching context
        x_s3 = self.stages[3](self.downsample_layers[3](x_s2))

        # Context prior at two resolutions
        ctx_ori = self.context_encoder(x_s3)               # (N, ctx_ch, h/32, w/32)
        ctx_up  = F.interpolate(ctx_ori, size=x_s2.shape[2:],
                                mode='bilinear', align_corners=False)

        # Sub-stage s2 -- retro-active modulation at stage-2 resolution
        x, ctx = x_s2, ctx_up
        for blk in self.sub_blocks_s2:
            x, ctx = blk(x, ctx, ctx_up)

        # RetroActiveBridge -- joint downsampling to stage-3 resolution
        x, ctx = self.patch_embedx(x, ctx)

        # Sub-stage s3 -- retro-active modulation at stage-3 resolution
        for blk in self.sub_blocks_s3:
            x, ctx = blk(x, ctx, ctx_ori)

        return x                                            # (N, dims[3], h/32, w/32)

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

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------

    def normalize_block_bn(self):
        """Normalize BN gamma/beta in backbone stages 0-3 to prevent fp16 overflow
        from IN-22K multi-branch accumulation, while preserving relative channel
        importance learned during pretraining.

        For each target covered by the original reset_block_bn():
          - DilatedReparamBlock BNs (origin_bn + all dil_bn): gamma and beta are
            divided by the number of branches (N). Since forward() sums all N
            branches, dividing each by N makes the sum magnitude equivalent to a
            single branch -- prevents forward overflow AND backward gradient
            explosion through the N-branch gamma chain.
          - block.norm (BN after dwconv): gamma and beta are divided by
            max(|gamma|, 1) to bound |gamma| <= 1 while preserving relative order.
          - block.pwconv2[-1] (FFN final BN): same max-normalization.

        Running_mean/var are untouched (warm-start statistics preserved).
        Zero runtime memory cost (weights modified in-place at load time only)."""
        n_dilated = n_block_bn = 0
        for stage in self.stages:
            for block in stage:
                if not isinstance(block, UniRepLKNetBlock):
                    continue

                # --- DilatedReparamBlock BNs: divide by number of branches ---
                if isinstance(block.dwconv, DilatedReparamBlock):
                    drb = block.dwconv
                    n_branches = 1 + len(drb.kernel_sizes)  # origin + dilated
                    if hasattr(drb, 'origin_bn'):
                        drb.origin_bn.weight.data.div_(n_branches)
                        drb.origin_bn.bias.data.div_(n_branches)
                        n_dilated += 1
                    for k, r in zip(drb.kernel_sizes, drb.dilates):
                        bn = getattr(drb, f'dil_bn_k{k}_{r}', None)
                        if bn is not None:
                            bn.weight.data.div_(n_branches)
                            bn.bias.data.div_(n_branches)
                            n_dilated += 1

                # --- block.norm (BN post-dwconv): max-normalize gamma ---
                if hasattr(block.norm, 'weight'):
                    scale = block.norm.weight.data.abs().max().clamp(min=1.0)
                    block.norm.weight.data.div_(scale)
                    block.norm.bias.data.div_(scale)
                    n_block_bn += 1

                # --- pwconv2 final BN: max-normalize gamma ---
                ffn_bn = block.pwconv2[-1]
                if hasattr(ffn_bn, 'weight'):
                    scale = ffn_bn.weight.data.abs().max().clamp(min=1.0)
                    ffn_bn.weight.data.div_(scale)
                    ffn_bn.bias.data.div_(scale)
                    n_block_bn += 1

        print(f'[NORMALIZE] {n_dilated + n_block_bn} BNs normalized '
              f'({n_dilated} DilatedReparamBlock + {n_block_bn} block BNs). '
              f'Relative gamma structure preserved. running_mean/var untouched.')

    def load_pretrained_2d(self, ckpt, strict=False, skip_head=True,
                            map_location='cpu'):
        """Load a 2D ImageNet checkpoint into VideoSTORM.
        For ImageNet checkpoints, normalize_block_bn() is called after loading to
        scale BN gamma/beta proportionally -- preventing fp16 overflow while preserving
        the relative channel importance structure learned on IN-22K. Forward path is
        unchanged (no runtime overhead). For video resume checkpoints, gammas are
        preserved as-is (they were already normalized or trained from normalized state).
        DDP prefix stripped automatically."""
        def _load(p):
            if isinstance(p, str) and p.startswith('http'):
                return torch.hub.load_state_dict_from_url(p, map_location=map_location,
                                                           check_hash=False)
            try:
                return torch.load(p, map_location=map_location, weights_only=True)
            except TypeError:
                return torch.load(p, map_location=map_location)

        def _unwrap(raw):
            if not isinstance(raw, dict): return raw, 'unknown'
            ckpt_type = ('video_resume'
                         if ('epoch' in raw or 'optimizer' in raw) else 'imagenet')
            if 'model' in raw and isinstance(raw['model'], dict):
                return raw['model'], ckpt_type
            if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
                return raw['state_dict'], ckpt_type
            return raw, ckpt_type

        def _strip(sd):
            return {k[len('module.'):]: v for k, v in sd.items()} \
                   if any(k.startswith('module.') for k in sd) else sd

        src = ckpt if isinstance(ckpt, str) else '<dict>'
        print('=' * 80); print(f'[LOADING] {src}'); print('=' * 80)
        try:
            raw, ckpt_type = _unwrap(ckpt if isinstance(ckpt, dict) else _load(ckpt))
            if not isinstance(raw, dict):
                raise ValueError('Unsupported checkpoint format.')
            state = _strip(raw)

            if ckpt_type == 'video_resume':
                epoch_info = f"epoch {raw['epoch']}" if 'epoch' in raw else 'epoch unknown'
                print(f'[RESUME] {epoch_info} -- BN gammas preserved as-is.')
            else:
                print('[INIT] ImageNet checkpoint -- BN gamma normalization will apply.')

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
                if missing:    print(f'  Missing (first 10): {missing[:10]}')
                if unexpected: print(f'  Unexpected (first 10): {unexpected[:10]}')
            else:
                print(f'[SUCCESS] {len(state)} parameters loaded.')

            if ckpt_type == 'imagenet':
                self.normalize_block_bn()

        except FileNotFoundError as e:
            print(f'[ERROR] File not found: {src} -- {e}'); raise
        except Exception as e:
            print(f'[ERROR] {type(e).__name__}: {e}'); raise


# ---------------------------------------------------------------------------
# timm helpers
# ---------------------------------------------------------------------------

def _cfg(crop_pct=0.9, **kwargs):
    import timm
    return {'num_classes': 1000, 'input_size': (3, 224, 224),
            'crop_pct': crop_pct, 'interpolation': 'bicubic',
            'mean': timm.data.IMAGENET_DEFAULT_MEAN,
            'std':  timm.data.IMAGENET_DEFAULT_STD,
            'classifier': 'video_head', **kwargs}


def _pop_timm_args(kwargs):
    kwargs.pop('pretrained', False)
    for k in ('pretrained_cfg', 'checkpoint_path', 'features_only',
              'scriptable', 'exportable'):
        kwargs.pop(k, None)


# ---------------------------------------------------------------------------
# timm builder
# ---------------------------------------------------------------------------

@register_model
def videostorm_s(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    """Video-STORM-S: small variant, large-kernel backbone with DynamicSTORMBlock sub-stages.
    Requires: natten, einops."""
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth',      (4, 2))
    kwargs.setdefault('sub_num_heads',  (4, 8))
    kwargs.setdefault('smk_size',       5)
    kwargs.setdefault('projection',     2048)
    kwargs.setdefault('sub_ffn_factor', 2)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(96, 192, 384, 768),
        attempt_use_lk_impl=False,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


@register_model
def videostorm_b(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    """Video-STORM-B: large-kernel backbone with DynamicSTORMBlock sub-stages.
    Requires: natten, einops."""
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth',      (4, 2))
    kwargs.setdefault('sub_num_heads',  (4, 8))
    kwargs.setdefault('smk_size',       5)
    kwargs.setdefault('projection',     2048)
    kwargs.setdefault('sub_ffn_factor', 2)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(128, 256, 512, 1024),
        attempt_use_lk_impl=False,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


@register_model
def videostorm_l(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    """Video-STORM-L: large variant, large-kernel backbone with DynamicSTORMBlock sub-stages.
    Requires: natten, einops."""
    _pop_timm_args(kwargs)
    kwargs.setdefault('sub_depth',      (4, 2))
    kwargs.setdefault('sub_num_heads',  (4, 8))
    kwargs.setdefault('smk_size',       5)
    kwargs.setdefault('projection',     2048)
    kwargs.setdefault('sub_ffn_factor', 2)
    model = VideoSTORM(
        depths=UniRepLKNet_S_B_L_XL_depths,
        dims=(192, 384, 768, 1536),
        attempt_use_lk_impl=False,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained_2d:
        model.load_pretrained_2d(pretrained_2d, strict=pretrained_2d_strict,
                                  skip_head=(model.num_classes != 1000))
    return model


if __name__ == '__main__':
    print('test')