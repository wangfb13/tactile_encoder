"""
CLIP-ViT-B-16 触觉 MAE 继续预训练 (微调) 配置。

数据集: data_tac2_s —— 单帧 3 通道 (RGB) uint16 PNG 图片。
模型: 用 CLIP-ViT-B-16 视觉编码器权重初始化 MAE encoder, 继续 MAE 预训练。
"""

import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("CLIP-ViT-B-16 Tactile MAE Finetuning")

    # ---- data ----
    parser.add_argument(
        "--png_dir",
        type=str,
        required=True,
        help="PNG 数据目录 (内含 000000000.png ... 的单帧图片)",
    )
    parser.add_argument("--img_size", type=int, default=224)

    # ---- pretrained ----
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="CLIP open_clip_pytorch_model.bin 路径; 为空则随机初始化",
    )

    # ---- model ----
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument(
        "--in_channels",
        type=int,
        default=3,
        help="输入通道数 (data_tac2_s 为单帧 3 通道; 若为 6 则会扩展 CLIP patch_embed 权重)",
    )
    parser.add_argument("--encoder_dim", type=int, default=768)
    parser.add_argument("--encoder_layers", type=int, default=12)
    parser.add_argument("--encoder_heads", type=int, default=12)
    parser.add_argument("--decoder_dim", type=int, default=384)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_heads", type=int, default=6)

    # ---- MAE ----
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--norm_pix_loss", action="store_true", default=False)

    # ---- training ----
    parser.add_argument("--batch_size", type=int, default=256, help="per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="绝对学习率; 为空时由 blr 按有效 batch 计算",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1.5e-4,
        help="base lr: absolute_lr = blr * eff_batch / 256",
    )
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    # ---- system ----
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_mem", action="store_true", default=True)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--vis_freq", type=int, default=5)

    # ---- distributed ----
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--dist_backend", type=str, default="nccl")

    return parser
