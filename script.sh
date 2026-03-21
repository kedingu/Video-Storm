# Checkpoints : 
unireplknet_b_in22k_to_in1k_384_acc87.40.pth
unireplknet_b_in22k_pretrain.pth

#### k400

# 4 gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 8 \
    --batch-size 8 \
    --output videostorm_base_k400 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in22k_to_in1k_384_acc87.40.pth


# 2 gpus
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in22k_to_in1k_384_acc87.40.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --config configs/k400/k400_storm_small.yaml \
    --accumulation-steps 8 \
    --batch-size 8 \
    --output videostorm_small_k400 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_s_in22k_to_in1k_384_acc86.44.pth


#### ssv2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --config configs/ssv2/ssv2_storm_base.yaml \
    --accumulation-steps 8 \
    --batch-size 8 \
    --output videostorm_base_ssv2 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 1 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/ \
        DATA.TRAIN_FILE datasets/train_ssv2_s.txt \
        DATA.VAL_FILE datasets/val_ssv2_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in22k_to_in1k_384_acc87.40.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --config configs/ssv2/ssv2_storm_small.yaml \
    --accumulation-steps 8 \
    --batch-size 8 \
    --output videostorm_small_ssv2 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 1 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/ \
        DATA.TRAIN_FILE datasets/train_ssv2_s.txt \
        DATA.VAL_FILE datasets/val_ssv2_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_s_in22k_to_in1k_384_acc86.44.pth