import argparse
import torch
from thop import profile, clever_format
from utils.config import get_config
from classification import build_model

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_option():
    parser = argparse.ArgumentParser('FLOPs/MACs probe', add_help=False)
    parser.add_argument('--config', '-cfg', type=str, required=False, metavar="FILE",
                        help='path to config file',
                        default='./configs/k400/k400_storm_base.yaml')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.",
                        default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument("--local_rank", type=int, default=0,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='path to pretrained weights (kept for config compatibility)')
    parser.add_argument('--only_test', action='store_true',
                        help='eval-only flag (kept for config compatibility)')
    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config

_, config = parse_option()

config.defrost()
config.DATA.NUM_FRAMES = getattr(config.DATA, "NUM_FRAMES", 8)
config.freeze()

model = build_model(config)
model = model.to(device)

img_size = getattr(config.DATA, "IMG_SIZE", 224)

if "videostorm" in str(config.MODEL.TYPE).lower():
    data = torch.randn(1, 3, config.DATA.NUM_FRAMES, img_size, img_size, device=device)
else:
    data = torch.randn(1, config.DATA.NUM_FRAMES, 3, img_size, img_size, device=device)

with torch.no_grad():
    macs, params = profile(model, inputs=(data, ))
macs_str, params_str = clever_format([macs, params], "%.3f")
print("FLOPs:", macs_str, "Params:", params_str)