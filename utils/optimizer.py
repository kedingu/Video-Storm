import copy
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist


def is_main_process():
    return dist.get_rank() == 0


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


# [ADDED #8] Backbone prefix heuristic (fallback only, see FIX 7 below).
# 'downsample_layers' and 'stages' hold the pretrained UniRepLKNet weights.
_VIDEOSTORM_BACKBONE_PREFIXES = ('downsample_layers.', 'stages.')

# [FIX 7] Randomly-initialized modules that live INSIDE the backbone
# prefixes and must therefore be carved out of the reduced-LR groups:
#   stages.*.temporal_branch.*  -> TemporalTransitionBranch (TTB)
#   stages.*.tsm.*              -> TemporalShift (parameter-free; listed
#                                  defensively in case it ever gains params)
_VIDEOSTORM_NEW_IN_BACKBONE_MARKERS = ('temporal_branch', 'tsm')


def _is_videostorm_model(model) -> bool:
    """Detect VideoSTORM by the presence of its characteristic attributes."""
    return hasattr(model, 'stages') and hasattr(model, 'sub_blocks_s2')


def _videostorm_backbone_param_names(model):
    """[FIX 7] Return the set of parameter names entitled to the reduced LR.

    Criterion: a parameter deserves LR * LR_BACKBONE_RATIO if and only if it
    was loaded from the IN-21K checkpoint. Prefix-based grouping is WRONG on
    its own: the randomly-initialized TTBs live under stages.*.temporal_branch
    and were silently trained at 0.1x LR, crippling temporal learning.

    Primary path: model.pretrained_param_names() (exact key set recorded by
    load_pretrained_2d; falls back internally to a name heuristic when no 2D
    checkpoint was loaded). Secondary path, for models predating FIX 7:
    prefix heuristic minus the temporal markers.
    """
    if hasattr(model, 'pretrained_param_names'):
        return set(model.pretrained_param_names())
    return {
        name for name, _ in model.named_parameters()
        if any(name.startswith(p) for p in _VIDEOSTORM_BACKBONE_PREFIXES)
        and not check_keywords_in_name(name, _VIDEOSTORM_NEW_IN_BACKBONE_MARKERS)
    }


def set_weight_decay_with_lr_scale(
        model,
        skip_list=(),
        skip_keywords=(),
        weight_decay=0.001,
        lr=8e-4,
        lr_backbone=None,
        have=(),
        not_have=()):
    """Build AdamW parameter groups with optional per-layer LR scaling.

    When lr_backbone is not None and the model is VideoSTORM, parameters
    genuinely loaded from the 2D checkpoint are placed in separate groups
    with lr=lr_backbone, while ALL randomly-initialized parameters
    (sub_blocks, heads, temporal_embed, and -- per FIX 7 -- the TTBs living
    inside stages.*) use lr=lr.

    Weight-decay exemption logic is unchanged: 1-D parameters and biases
    are exempted regardless of which LR group they belong to.

    Returns a list of dicts suitable for torch.optim.AdamW.
    """
    is_vs = _is_videostorm_model(model) and (lr_backbone is not None)

    # [FIX 7] Checkpoint-derived (or heuristic) ground truth, computed once.
    backbone_names = _videostorm_backbone_param_names(model) if is_vs else set()

    # [FIX 7] Hard audit: no randomly-initialized temporal parameter may sit
    # in the reduced-LR groups. Holds for both the checkpoint-derived path
    # (these keys are absent from the IN-21K checkpoint) and the heuristic
    # path (markers explicitly excluded). Fails loudly rather than silently
    # under-training the temporal branches.
    leaked = [n for n in backbone_names
              if check_keywords_in_name(n, _VIDEOSTORM_NEW_IN_BACKBONE_MARKERS)]
    assert not leaked, \
        f'[FIX 7] Temporal params wrongly assigned backbone LR: {leaked[:5]}'

    # Four buckets: (backbone|new) x (decay|no_decay)
    # When not VideoSTORM or lr_backbone is None, everything goes to head buckets.
    backbone_decay    = []
    backbone_no_decay = []
    head_decay        = []
    head_no_decay     = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue

        # Weight-decay exemption: 1-D tensors (BN weight, bias, gamma, beta),
        # named biases, and skip-list / skip-keyword entries.
        no_wd = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        )

        # [FIX 7] Membership test against the checkpoint-derived name set
        # replaces the old prefix test (_is_videostorm_backbone).
        if is_vs and name in backbone_names:
            (backbone_no_decay if no_wd else backbone_decay).append(param)
        else:
            (head_no_decay if no_wd else head_decay).append(param)

    if is_vs and dist.is_available() and dist.is_initialized() \
            and is_main_process():
        print(f'[FIX 7] Param groups -- backbone (reduced LR): '
              f'{len(backbone_decay) + len(backbone_no_decay)} tensors | '
              f'new (full LR): {len(head_decay) + len(head_no_decay)} tensors')

    groups = []

    # New / head parameters at full learning rate
    if head_decay:
        groups.append({'params': head_decay,
                       'weight_decay': weight_decay,
                       'lr': lr})
    if head_no_decay:
        groups.append({'params': head_no_decay,
                       'weight_decay': 0.,
                       'lr': lr})

    # Backbone parameters at reduced learning rate (appended last so that
    # param_groups[0]['lr'] in logging always refers to the head/new-block LR)
    if backbone_decay:
        groups.append({'params': backbone_decay,
                       'weight_decay': weight_decay,
                       'lr': lr_backbone})
    if backbone_no_decay:
        groups.append({'params': backbone_no_decay,
                       'weight_decay': 0.,
                       'lr': lr_backbone})

    return groups


# Keep the old name as an alias so that any external call sites remain valid.
def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001,
                     lr=2e-6, have=(), not_have=()):
    """Legacy wrapper: uniform LR for all parameters (no differential LR)."""
    return set_weight_decay_with_lr_scale(
        model, skip_list=skip_list, skip_keywords=skip_keywords,
        weight_decay=weight_decay, lr=lr,
        lr_backbone=None,       # no differential LR
        have=have, not_have=not_have)


def fix_text(model):
    for name, param in model.named_parameters():
        if "visual." in name or "mit" in name or "prompts" in name:
            continue
        else:
            param.requires_grad = False


def build_optimizer(config, model):
    model = model.module if hasattr(model, 'module') else model

    # fix_text only for CLIP-like models (ignored for VideoSTORM via FIX_TEXT=False)
    if config.MODEL.FIX_TEXT:
        fix_text(model)

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    # [ADDED #8] Differential LR: backbone vs new layers.
    # LR_BACKBONE_RATIO < 1.0  -> backbone gets LR * ratio, new layers get LR.
    # LR_BACKBONE_RATIO = 1.0  -> uniform LR for all parameters (legacy).
    # [FIX 7] "backbone" now means "loaded from the IN-21K checkpoint", not
    # "name starts with stages./downsample_layers." -- so the TTBs (and any
    # future module added inside stages) automatically train at full LR.
    # NOTE: build_optimizer should be called AFTER load_pretrained_2d so the
    # checkpoint-derived key set is available; otherwise the name-marker
    # fallback inside pretrained_param_names() applies (same result for the
    # current architecture).
    lr_backbone_ratio = getattr(config.TRAIN, 'LR_BACKBONE_RATIO', 1.0)
    lr_backbone = (config.TRAIN.LR * lr_backbone_ratio
                   if lr_backbone_ratio < 1.0 else None)

    parameters = set_weight_decay_with_lr_scale(
        model,
        skip_list=skip,
        skip_keywords=skip_keywords,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        lr=config.TRAIN.LR,
        lr_backbone=lr_backbone)

    optimizer = optim.AdamW(parameters, betas=(0.9, 0.999), eps=1e-8)
    return optimizer


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)

    # timm's CosineLRScheduler reads 'initial_lr' (set automatically by
    # AdamW from each group's 'lr') and scales each group independently.
    # With differential LR, the backbone group decays from lr*ratio -> lr_min,
    # while the head group decays from lr -> lr_min.
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.TRAIN.LR / 100,
        warmup_lr_init=0,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return lr_scheduler