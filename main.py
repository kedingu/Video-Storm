import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random

# Apex (optional)
try:
    from apex import amp
    HAS_APEX = True
except ImportError:
    HAS_APEX = False

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from classification import build_model
from timm.models.layers import trunc_normal_


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str)
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument("--local-rank", type=int, default=-1,
                        help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _has_any_grad(params) -> bool:
    for p in params:
        if p.grad is not None:
            return True
    return False


def _count_trainables(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _amp_flags_and_dtype(config):
    """Return (amp_enabled, amp_dtype) honoring TRAIN.OPT_LEVEL and AMP_DTYPE."""
    amp_enabled = (getattr(config.TRAIN, "OPT_LEVEL", "O1") != "O0")
    dtype_str = getattr(config, "AMP_DTYPE", "fp16")
    amp_dtype = torch.bfloat16 if str(dtype_str).lower() == "bf16" else torch.float16
    return amp_enabled, amp_dtype


def _strip_prefix_if_present(state_dict, prefix="module."):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}


def _safe_manual_load_pretrained(model: torch.nn.Module, pretrained_path: str,
                                  logger, config):
    """Local .pth loading with shape-mismatch handling."""
    if not pretrained_path:
        return
    if os.path.isdir(pretrained_path):
        logger.warning(
            f"[PRETRAINED] Ignored directory path (expecting a .pth file): {pretrained_path}")
        return
    if not os.path.isfile(pretrained_path):
        logger.warning(f"[PRETRAINED] File not found: {pretrained_path}")
        return

    logger.info(f"[PRETRAINED] Loading weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', None)
        if state_dict is None:
            state_dict = checkpoint.get('state_dict', checkpoint)
    else:
        state_dict = checkpoint

    state_dict = _strip_prefix_if_present(
        state_dict,
        prefix="module:" if "module:" in next(iter(state_dict.keys()), "")
               else "module.")

    model_state = model.state_dict()

    # Re-init head if shape mismatches
    head_names = [
        ("head.weight",       "head.bias"),
        ("fc.weight",         "fc.bias"),
        ("classifier.weight", "classifier.bias"),
    ]
    for w_name, b_name in head_names:
        if (w_name in model_state) and (b_name in model_state):
            if (w_name in state_dict) and (b_name in state_dict):
                if (model_state[w_name].shape != state_dict[w_name].shape
                        or model_state[b_name].shape != state_dict[b_name].shape):
                    state_dict[w_name] = torch.empty_like(model_state[w_name])
                    trunc_normal_(state_dict[w_name], std=.02)
                    state_dict[b_name] = torch.empty_like(model_state[b_name])
                    trunc_normal_(state_dict[b_name], std=.02)
                    logger.info(
                        f"[PRETRAINED] Reinit head due to shape mismatch: "
                        f"replaced {w_name}/{b_name}")

    # Drop any remaining mismatched keys
    drop_keys = []
    for k, v in list(state_dict.items()):
        if k in model_state and model_state[k].shape != v.shape:
            drop_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            state_dict.pop(k)
    for k, sh_src, sh_dst in drop_keys:
        logger.info(
            f"[PRETRAINED] Drop '{k}' due to shape mismatch: {sh_src} != {sh_dst}")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"[PRETRAINED] Loaded with msg: {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

    logger.info(f"Creating model: {config.MODEL.TYPE}")

    # VideoSTORM loads 2D ImageNet weights via TRAIN.PRETRAINED_PATH -> pretrained_2d
    # inside build_model; MODEL.PRETRAINED / _safe_manual_load_pretrained must be skipped.
    is_videostorm = "videostorm" in str(config.MODEL.TYPE).lower()

    user_pretrained_path = (
        config.MODEL.PRETRAINED
        if isinstance(config.MODEL.PRETRAINED, str)
           and len(config.MODEL.PRETRAINED) > 0
        else None
    )

    config.defrost()
    config.MODEL.PRETRAINED = False
    config.freeze()

    model = build_model(config)

    config.defrost()
    config.MODEL.PRETRAINED = user_pretrained_path
    config.freeze()

    if is_videostorm:
        if user_pretrained_path:
            logger.warning(
                "[VideoSTORM] Ignoring MODEL.PRETRAINED. "
                "Use TRAIN.PRETRAINED_PATH instead "
                "(passed to build_model as 'pretrained_2d').")
    else:
        if user_pretrained_path:
            _safe_manual_load_pretrained(model, user_pretrained_path, logger, config)

    logger.info(f"Trainable params count (pre-optim): {_count_trainables(model):,}")

    # Disable FIX_TEXT for non-CLIP models
    if getattr(config.MODEL, "FIX_TEXT", False):
        has_clip_like_names = any(
            ("visual." in n or "text" in n or "token" in n)
            for n, _ in model.named_parameters())
        if not has_clip_like_names:
            logger.info(
                "[CONFIG PATCH] Disabling MODEL.FIX_TEXT (model has no text branch).")
            config.defrost()
            config.MODEL.FIX_TEXT = False
            config.freeze()

    # [ADDED #1] Convert all BatchNorm layers to SyncBatchNorm BEFORE cuda()
    # and DDP wrapping.
    #
    # Root cause fixed: VideoSTORM.__init__ previously contained:
    #   self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
    # which is a Python no-op -- reassigning 'self' inside __init__ has no
    # effect on the object visible to the caller. As a result, all BN layers
    # in the model (especially the many fresh BN2d layers in STORMBlocks)
    # remained as independent per-GPU BatchNorm throughout training, producing
    # unreliable statistics with small per-GPU batch sizes and preventing the
    # STORMBlock from converging properly.
    #
    # nn.SyncBatchNorm.convert_sync_batchnorm converts ALL _BatchNorm
    # subclasses (BatchNorm1d, BatchNorm2d, BatchNorm3d) while preserving
    # existing running_mean / running_var / weight / bias values -- so
    # pretrained BN statistics loaded above are fully preserved.
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info(
            f"[SyncBN] Converted all BatchNorm layers to SyncBatchNorm "
            f"(world_size={dist.get_world_size()}).")
    else:
        logger.info("[SyncBN] Single-GPU mode: skipping SyncBN conversion.")

    # Loss & mixup
    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(
            num_classes=config.DATA.NUM_CLASSES,
            smoothing=config.AUG.LABEL_SMOOTH,
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()

    model = model.cuda()

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    num_trainable_after_opt = _count_trainables(model)
    eval_only = (bool(getattr(config.TEST, "ONLY_TEST", False))
                 or bool(getattr(config, "EVAL_MODE", False)))

    if num_trainable_after_opt == 0 and not eval_only:
        logger.warning(
            "[DDP SAFETY] No trainable params after build_optimizer(). "
            "Attempting to unfreeze heads...")
        restored = 0
        for n, p in model.named_parameters():
            lname = n.lower()
            if any(k in lname for k in ["head", "fc", "classifier"]):
                if not p.requires_grad:
                    p.requires_grad = True
                    restored += p.numel()
        if restored == 0:
            logger.warning(
                "[DDP SAFETY] No head detected — unfreezing the entire model.")
            for p in model.parameters():
                p.requires_grad = True
        optimizer = build_optimizer(config, model)
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
        num_trainable_after_opt = _count_trainables(model)
        logger.info(f"[DDP SAFETY] Restored trainables: {num_trainable_after_opt:,}")
        if num_trainable_after_opt == 0:
            raise RuntimeError(
                "Still no trainable parameter after unfreezing attempt. "
                "Check your config.")

    # AMP (fp16 / bf16)
    amp_enabled, amp_dtype = _amp_flags_and_dtype(config)
    if HAS_APEX and (getattr(config.TRAIN, "OPT_LEVEL", "O1") != "O0") and amp_dtype != torch.bfloat16 and not eval_only:
        model, optimizer = amp.initialize(
            models=model, optimizers=optimizer,
            opt_level=config.TRAIN.OPT_LEVEL)
        scaler = None
        logger.info("Using NVIDIA Apex AMP.")
    else:
        use_scaler = amp_enabled and (amp_dtype == torch.float16)
        scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        logger.info(
            f"Using PyTorch AMP: enabled={amp_enabled}, "
            f"dtype={'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")

    # DDP
    if not eval_only and _count_trainables(model) > 0:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.LOCAL_RANK],
            broadcast_buffers=False, find_unused_parameters=False)
    else:
        logger.info("Skipping DDP wrapping (eval-only or no trainable parameter).")

    start_epoch, max_accuracy = 0, 0.0

    # Auto-resume
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # Resume
    if config.MODEL.RESUME:
        target_model = (model.module
                        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                        else model)
        start_epoch, max_accuracy = load_checkpoint(
            config, target_model, optimizer, lr_scheduler, logger)

    # [ADDED] Evaluate the resumed checkpoint before continuing training.
    # Rationale: if the job was killed between the periodic save (epoch N) and
    # the validation that would have followed it, the checkpoint holds weights
    # from epoch N but max_accuracy only reflects the last *evaluated* epoch.
    # Validating here closes that gap: if epoch N is actually the best, its
    # accuracy is measured now and best.pth is saved immediately, before
    # epoch N+1 training begins.
    if config.MODEL.RESUME and not config.TEST.ONLY_TEST:
        logger.info(
            f"[RESUME] Validating checkpoint (epoch={start_epoch - 1}) "
            f"before resuming training (stored max_accuracy={max_accuracy:.2f}%)...")
        resumed_acc = validate(val_loader, model, config, amp_enabled, amp_dtype, scaler)
        logger.info(
            f"[RESUME] Checkpoint validation: Acc@1={resumed_acc:.2f}% "
            f"(stored max_accuracy={max_accuracy:.2f}%)")
        is_best = resumed_acc > max_accuracy
        if is_best:
            max_accuracy = resumed_acc
            if dist.get_rank() == 0:
                epoch_saving(
                    config, start_epoch - 1,
                    model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel) else model,
                    max_accuracy, optimizer, lr_scheduler, logger,
                    config.OUTPUT, is_best=True)
                logger.info(
                    f"[RESUME] New best checkpoint saved at epoch {start_epoch - 1}: "
                    f"{max_accuracy:.2f}%")
        else:
            max_accuracy = max(max_accuracy, resumed_acc)
        logger.info(f"[RESUME] max_accuracy updated to {max_accuracy:.2f}%")

    if config.TEST.ONLY_TEST:
        acc1 = validate(val_loader, model, config, amp_enabled, amp_dtype, scaler)
        logger.info(
            f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    logger.info(f"ACCUMULATION_STEPS = {config.TRAIN.ACCUMULATION_STEPS}")
    logger.info(f"AUX_LOSS_WEIGHT = {config.TRAIN.AUX_LOSS_WEIGHT}")

    # [ADDED #8] Log differential LR config so it is visible in experiment logs.
    lr_backbone_ratio = getattr(config.TRAIN, 'LR_BACKBONE_RATIO', 1.0)
    if lr_backbone_ratio < 1.0:
        logger.info(
            f"Differential LR: backbone lr={config.TRAIN.LR * lr_backbone_ratio:.2e} "
            f"({lr_backbone_ratio}x), heads lr={config.TRAIN.LR:.2e}")
    else:
        logger.info(f"Uniform LR: {config.TRAIN.LR:.2e} for all parameters.")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(start_epoch, config.TRAIN.EPOCHS):

        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler,
                        train_loader, config, mixup_fn, amp_enabled, amp_dtype, scaler)

        # Step 1: checkpoint immediately after training
        if dist.get_rank() == 0 and (
                epoch % config.SAVE_FREQ == 0
                or epoch == config.TRAIN.EPOCHS - 1):
            epoch_saving(
                config, epoch,
                model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel) else model,
                max_accuracy, optimizer, lr_scheduler, logger,
                config.OUTPUT, is_best=False)

        # Step 2: selective validation schedule
        every_5   = (epoch % 5 == 0)
        last_3    = (epoch >= config.TRAIN.EPOCHS - 20)
        do_validate = every_5 or last_3

        if not do_validate:
            logger.info(f"Epoch {epoch}: skipping validation.")
            continue

        # Step 3: validation
        acc1 = validate(val_loader, model, config, amp_enabled, amp_dtype, scaler)
        logger.info(
            f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")

        # Step 4: save best checkpoint if accuracy improved
        is_best = acc1 > max_accuracy
        if is_best:
            max_accuracy = acc1
            if dist.get_rank() == 0:
                epoch_saving(
                    config, epoch,
                    model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel) else model,
                    max_accuracy, optimizer, lr_scheduler, logger,
                    config.OUTPUT, is_best=True)
        else:
            max_accuracy = max(max_accuracy, acc1)

        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    # Final multi-view eval (4 clips x 3 crops)
    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, model, config, amp_enabled, amp_dtype, scaler)
    logger.info(
        f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader,
                    config, mixup_fn, amp_enabled, amp_dtype, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    aux_ratio = config.TRAIN.AUX_LOSS_WEIGHT

    start = time.time()
    end = time.time()

    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)

        images = images.view(
            (-1, config.DATA.NUM_FRAMES, 3) + images.size()[-2:])

        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if HAS_APEX and (getattr(config.TRAIN, "OPT_LEVEL", "O1") != "O0") and amp_dtype != torch.bfloat16:
            # ---- Apex AMP branch ------------------------------------------
            output = model(images)
            if isinstance(output, dict):
                loss_main = criterion(output['main'], label_id)
                loss_aux  = criterion(output['aux'],  label_id)
                total_loss = ((loss_main + aux_ratio * loss_aux)
                              / max(1, config.TRAIN.ACCUMULATION_STEPS))
            else:
                loss_main = None
                loss_aux  = None
                total_loss = (criterion(output, label_id)
                              / max(1, config.TRAIN.ACCUMULATION_STEPS))
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if amp_enabled:
                # ---- PyTorch native AMP branch ----------------------------
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype,
                                        enabled=True):
                    output = model(images)

                if isinstance(output, dict):
                    loss_main = criterion(output['main'].float(), label_id)
                    loss_aux  = criterion(output['aux'].float(),  label_id)
                    total_loss = ((loss_main + aux_ratio * loss_aux)
                                  / max(1, config.TRAIN.ACCUMULATION_STEPS))
                else:
                    loss_main = None
                    loss_aux  = None
                    total_loss = (criterion(output.float(), label_id)
                                  / max(1, config.TRAIN.ACCUMULATION_STEPS))

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
            else:
                # ---- fp32 branch ------------------------------------------
                output = model(images)
                if isinstance(output, dict):
                    loss_main = criterion(output['main'], label_id)
                    loss_aux  = criterion(output['aux'],  label_id)
                    total_loss = ((loss_main + aux_ratio * loss_aux)
                                  / max(1, config.TRAIN.ACCUMULATION_STEPS))
                else:
                    loss_main = None
                    loss_aux  = None
                    total_loss = (criterion(output, label_id)
                                  / max(1, config.TRAIN.ACCUMULATION_STEPS))
                total_loss.backward()

        def _do_step():
            if (amp_enabled and scaler is not None
                    and scaler.is_enabled() and not HAS_APEX):
                scaler.unscale_(optimizer)
                if config.TRAIN.CLIP_GRAD > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD)
                if _has_any_grad(model.parameters()):
                    scaler.step(optimizer)
                scaler.update()
            else:
                if config.TRAIN.CLIP_GRAD > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.TRAIN.CLIP_GRAD)
                if _has_any_grad(model.parameters()):
                    optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step_update(epoch * num_steps + idx)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                _do_step()
        else:
            _do_step()

        torch.cuda.synchronize()

        tot_loss_meter.update(total_loss.item(), len(label_id))
        if loss_aux is not None:
            aux_loss_meter.update(loss_aux.item(), len(label_id))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            # [ADDED #8] Also log backbone LR when differential LR is active.
            lr_backbone_ratio = getattr(config.TRAIN, 'LR_BACKBONE_RATIO', 1.0)
            lr_info = f'lr {lr:.9f}'
            if lr_backbone_ratio < 1.0 and len(optimizer.param_groups) > 2:
                # Backbone groups are appended last by set_weight_decay_with_lr_scale
                lr_bb = optimizer.param_groups[-1]['lr']
                lr_info = f'lr_head {lr:.9f} lr_bb {lr_bb:.9f}'

            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if aux_loss_meter.count > 0:
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} {lr_info}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                    f'aux_loss {aux_loss_meter.val:.4f} ({aux_loss_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')
            else:
                logger.info(
                    f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} {lr_info}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                    f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(val_loader, model, config, amp_enabled, amp_dtype, scaler):
    model.eval()

    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()

    logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
    for idx, batch_data in enumerate(val_loader):
        _image = batch_data["imgs"]
        label_id = batch_data["label"]
        label_id = label_id.reshape(-1).cuda(non_blocking=True)

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()

        for i in range(n):
            image = _image[:, i, :, :, :, :]  # (b, t, c, h, w)
            image_input = image.cuda(non_blocking=True)

            if amp_enabled:
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype,
                                        enabled=True):
                    output = model(image_input)
            else:
                output = model(image_input)

            if isinstance(output, dict):
                output = output['main']
            elif isinstance(output, (list, tuple)):
                output = output[0]

            similarity = output.view(b, -1).softmax(dim=-1)
            tot_similarity += similarity

        values_1, indices_1 = tot_similarity.topk(1, dim=-1)
        values_5, indices_5 = tot_similarity.topk(5, dim=-1)

        acc1, acc5 = 0, 0
        for i in range(b):
            if indices_1[i] == label_id[i]:
                acc1 += 1
            if label_id[i] in indices_5[i]:
                acc5 += 1

        acc1_meter.update(float(acc1) / b * 100, b)
        acc5_meter.update(float(acc5) / b * 100, b)

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx}/{len(val_loader)}]\tAcc@1: {acc1_meter.avg:.3f}')

    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.TYPE}")
    logger.info(f"working dir: {config.OUTPUT}")

    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)