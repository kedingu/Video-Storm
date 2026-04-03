# Checkpoints : 
unireplknet_b_in22k_to_in1k_384_acc87.40.pth
unireplknet_b_in22k_pretrain.pth


#### k400
# bf without LN
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e_4Lr_30epochs_brainFloatingPoint16\
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 30 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth

# bf with LN
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e_4Lr_30epochs_bf16 \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 30 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth


# break at epoch 4 Alpha temporal, no RMS_norm
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e_4Lr_30epochs_normalizeGamma_noRMS_alphaTemporal \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 30 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth

# without RMS_nom
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_4e-4Lr_30epochs_normalizeGamma \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 30 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth


# epoch 35, 78.1% Normalize gamma and beta, base, 50 epochs
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e-5Lr_50epochs_normalizeGamma \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 50 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 8 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e-5Lr_50epochs_normalizeGamma \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 50 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth


#reset gamma and beta, base, 50 epochs, the training will resume at epoch 5
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e-5Lr_50epochs \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        TRAIN.EPOCHS 50 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth

#reset gamma and beta, base, 30 epochs, the training will resume at epoch 3
CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node=2 \
    main.py \
    --config configs/k400/k400_storm_base.yaml \
    --accumulation-steps 16 \
    --batch-size 8 \
    --output videostorm_base_k400_img21k_8e-5Lr \
    --opts \
        TRAIN.AUTO_RESUME True \
        TEST.NUM_CLIP 4 TEST.NUM_CROP 3 \
        DATA.ROOT /storage/Knguetche/kinetics_resized/k400 \
        DATA.TRAIN_FILE datasets/train_k400_s.txt \
        DATA.VAL_FILE datasets/val_k400_s.txt \
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth

# small
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
        TRAIN.PRETRAINED_PATH unireplknet_s_in21k_pretrain.pth


# img1k 384*384, Break after 1 epochs, 4 gpus
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
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_to_in1k_384_acc87.40.pth


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
        TRAIN.PRETRAINED_PATH unireplknet_b_in21k_pretrain.pth


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
        TRAIN.PRETRAINED_PATH unireplknet_s_in21k_pretrain.pth