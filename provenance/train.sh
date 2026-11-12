#!/bin/bash
# CLIP-ViT-B-16 触觉 MAE 继续预训练 (4 卡 torchrun)
# 在 vit-b-tac2_s 目录下运行: bash train.sh
set -e

cd "$(dirname "$0")"

torchrun --nproc_per_node=4 train.py \
    --png_dir /mnt/data/xidong_data/lerobot_tactile_ws/AnyTouch/zxd/data/data_tac2_s \
    --pretrained /mnt/data/xidong_data/tac_infra/playground/pretrained_models/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin \
    --batch_size 256 \
    --epochs 100 \
    --img_size 224 \
    --patch_size 16 \
    --num_workers 16 \
    --save_freq 5 \
    --vis_freq 5 \
    --output_dir ./output
