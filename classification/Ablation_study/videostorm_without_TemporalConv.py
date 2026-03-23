"""
Video-STORM Ablation C: pure spatial baseline (no temporal, no sub-stages).

Ablation target: all Video-STORM contributions are removed:
  - AdaptiveTemporalConv (no temporal branch in any block)
  - DynamicSTORMBlock, RetroActiveBridge, context_encoder (no sub-stages)

This is the minimal spatial baseline: a 4-stage large-kernel convolutional
backbone applied frame-wise, with temporal aggregation only at the video_head
via mean-pooling over T. It measures the contribution of both spatio-temporal
modeling and retro-active context modulation jointly.

Pipeline (frame-wise on B*T frames):
    downsample_layers[0-1] + stages[0-1]  -- low/mid-level spatial features
    downsample_layers[2]   + stages[2]    -- mid-level spatial features
    downsample_layers[3]   + stages[3]    -- high-level spatial features (dims[3])
    video_head (Conv1x1->BN->SiLU->AdaptiveAvgPool->Linear)
    mean over T frames -> class logits

ImageNet checkpoint compatibility: identical to full VideoSTORM.

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
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


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
# Utility modules
# ---------------------------------------------------------------------------

class RMSNorm2d(nn.Module):
    """Parameter-free RMS normalisation over the channel dimension of (N,C,H,W) tensors.
    Used to stabilise large-magnitude activations from IN-22K pretrained gammas
    before SE recalibration [AMP-safe]."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms.to(x.dtype)


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
    """LayerNorm supporting channels-last and channels-first layouts."""
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
    and a GRN-FFN. Module names are frozen for ImageNet checkpoint compatibility.
    No temporal branch in this ablation (num_frames is not used).

    Overflow prevention is handled at load time by normalize_block_bn() which scales
    BN gamma/beta proportionally (division by N for DilatedReparamBlock branches,
    max-normalization for block.norm and FFN BN). Forward path is unmodified --
    zero runtime overhead."""
    def __init__(self, dim, kernel_size, drop_path=0.,
                 layer_scale_init_value=1e-6, deploy=False,
                 attempt_use_lk_impl=True, with_cp=False,
                 use_sync_bn=False, ffn_factor=4):
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

        # Parameter-free RMSNorm applied unconditionally to stabilise fp16 with
        # ImageNet-22K pretrained gammas. Mirrors the normalisation that the full
        # model applies before the shared SE when a temporal branch is present.
        self.rms_norm = RMSNorm2d()

    def compute_residual(self, x):
        v_s = self.norm(self.dwconv(x))
        # RMSNorm stabilises large-magnitude activations from ImageNet-22K gammas
        y = self.se(self.rms_norm(v_s))
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
# VideoSTORM -- Ablation C (pure spatial baseline)
# ---------------------------------------------------------------------------

class VideoSTORM(nn.Module):
    """Video-STORM Ablation C: pure spatial baseline.
    Four-stage large-kernel convolutional backbone applied frame-wise.
    No AdaptiveTemporalConv, no sub-stages, no context flow.
    Temporal aggregation via mean-pooling over T at the video_head only."""

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024),
                 drop_path_rate=0., layer_scale_init_value=1e-6,
                 head_init_scale=1., kernel_sizes=None,
                 deploy=False, with_cp=False, attempt_use_lk_impl=True,
                 use_sync_bn=False, num_frames=8,
                 projection=2048,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_frames  = num_frames
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

        # Stages 0-3 -- spatial only, no temporal branch
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                UniRepLKNetBlock(
                    dim=dims[i], kernel_size=kernel_sizes[i][j],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    deploy=deploy, attempt_use_lk_impl=attempt_use_lk_impl,
                    with_cp=with_cp, use_sync_bn=use_sync_bn)
                for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        # Classification head -- directly from stage-3
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
        # Pure spatial pipeline -- no temporal branch, no sub-stages
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
        return x                                        # (N, dims[3], h/32, w/32)

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
        """Load a 2D ImageNet checkpoint into VideoSTORM Ablation C.
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
# timm builders
# ---------------------------------------------------------------------------

@register_model
def videostorm_s(pretrained_2d=None, pretrained_2d_strict=False, **kwargs):
    """Video-STORM-S Ablation C: pure spatial baseline, no temporal, no sub-stages."""
    _pop_timm_args(kwargs)
    for k in ('sub_depth', 'sub_num_heads', 'smk_size',
              'sub_ffn_factor', 'sub_drop_path_rate'):
        kwargs.pop(k, None)
    kwargs.setdefault('projection', 2048)
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
    """Video-STORM-B Ablation C: pure spatial baseline, no temporal, no sub-stages."""
    _pop_timm_args(kwargs)
    for k in ('sub_depth', 'sub_num_heads', 'smk_size',
              'sub_ffn_factor', 'sub_drop_path_rate'):
        kwargs.pop(k, None)
    kwargs.setdefault('projection', 2048)
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
    """Video-STORM-L Ablation C: pure spatial baseline, no temporal, no sub-stages."""
    _pop_timm_args(kwargs)
    for k in ('sub_depth', 'sub_num_heads', 'smk_size',
              'sub_ffn_factor', 'sub_drop_path_rate'):
        kwargs.pop(k, None)
    kwargs.setdefault('projection', 2048)
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
    print('test')