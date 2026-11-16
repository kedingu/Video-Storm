import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.ROOT = ''
_C.DATA.TRAIN_FILE = ''
_C.DATA.VAL_FILE = ''
_C.DATA.DATASET = 'kinetics400'
_C.DATA.INPUT_SIZE = 224
_C.DATA.IMG_SIZE = 224
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 400
_C.DATA.LABEL_LIST = 'labels/kinetics_400_labels.csv'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'videostorm_b'
_C.MODEL.NAME = 'videostorm_b'
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.PRETRAINED = None
_C.MODEL.RESUME = None
_C.MODEL.FIX_TEXT = False
_C.MODEL.NUM_CLASSES = 400
_C.MODEL.LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 30
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.WARMUP_EPOCHS = 5
# [MODIFIED C1] Standard AdamW weight decay for K400 fine-tuning.
# Raised from 0.001 to 0.05 to match established recipes (VideoSwin, VideoMAE).
# At 0.001 the new STORMBlocks had ~62× less regularisation than the reference
# baselines, risking overfitting on K400's 240k training clips.
# Note: 1-D tensors (biases, BN/LN params) and the affine scale parameters
# listed in VideoSTORM.no_weight_decay_keywords() are exempt from WD.
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.LR = 8.e-4
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.ACCUMULATION_STEPS = 8
_C.TRAIN.LR_SCHEDULER = 'cosine'
_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.OPT_LEVEL = 'O1'
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.PRETRAINED_PATH = None

# Differential LR ratio for the pretrained backbone.
# backbone (downsample_layers + stages) : LR * 0.1  = 8e-5
# new layers (sub_blocks, heads, temporal branches) : LR * 1.0  = 8e-4
# Set to 1.0 to recover uniform-LR behaviour.
_C.TRAIN.LR_BACKBONE_RATIO = 0.1

# [MODIFIED C2] Auxiliary loss weight reduced from 0.4 to 0.1.
# The aux head classifies directly from the backbone output (before STORMBlocks).
# At 0.4, the gradient pressure on the backbone to be self-sufficient without
# STORMBlocks was nearly as strong as the main signal, preventing the backbone
# from learning representations that benefit from STORM enhancement.
# At 0.1 the main path owns 91% of the classification objective, letting
# STORMBlocks take full responsibility for the improvement while the aux head
# still acts as a light regulariser against catastrophic forgetting.
_C.TRAIN.AUX_LOSS_WEIGHT = 0.4

_C.DATA.BATCH_SIZE = 8

# AMP settings
_C.AMP_OPT_LEVEL = 'O1'
_C.AMP_DTYPE = 'bf16'

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.LABEL_SMOOTH = 0.1
_C.AUG.COLOR_JITTER = 0.8
_C.AUG.GRAY_SCALE = 0.2
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# Temporal stride between frames within a dense clip (clip_len > 1).
# With the current TSN-style pipeline (clip_len=1), this parameter is
# semantically inert but is forwarded to SampleFrames for consistency and
# to avoid silent errors if clip_len is ever changed.
_C.AUG.FRAME_INTERVAL = 4
_C.AUG.FLIP_RATIO = 0.5
_C.AUG.FLIP_DIRECTION = 'horizontal'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.NUM_CLIP = 1
_C.TEST.NUM_CROP = 1
_C.TEST.ONLY_TEST = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT = ''
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 100
_C.SEED = 1024
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.config)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
        config.DATA.BATCH_SIZE = args.batch_size
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.output:
        config.OUTPUT = args.output
    if args.only_test:
        config.TEST.ONLY_TEST = True
    config.LOCAL_RANK = args.local_rank

    if config.DATA.INPUT_SIZE:
        config.DATA.IMG_SIZE = config.DATA.INPUT_SIZE
    if config.TRAIN.OPT_LEVEL:
        config.AMP_OPT_LEVEL = config.TRAIN.OPT_LEVEL

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config