from timm.models import create_model
from . import videostorm

def build_model(config):
    model_type = config.MODEL.TYPE
    print(f"Creating model: {model_type}")

    if "videostorm" in model_type:
        model = create_model(
            model_type,
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            num_frames=config.DATA.NUM_FRAMES,
            pretrained_2d=(config.TRAIN.PRETRAINED_PATH
                           if getattr(config.TRAIN, "PRETRAINED_PATH", "") else None),
        )
    else:
        model = create_model(
            model_type,
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
        )
    return model