"""
CLIP-ViT-B-16 触觉 MAE 继续预训练 (微调) 主入口。

用法 (4 卡):
  bash train.sh
或:
  torchrun --nproc_per_node=4 train.py --png_dir ... --pretrained ... [其它参数]
"""

import os
import time
import json
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import misc
from misc import NativeScalerWithGradNormCount as NativeScaler
from config import get_args_parser
from dataset import TactilePNGDataset
from model import TactileMAE
from engine import train_one_epoch, save_reconstruction_vis


def main(args):
    misc.init_distributed_mode(args)

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # --- Dataset ---
    print(f"[Dataset] Loading from: {args.png_dir}")
    dataset_train = TactilePNGDataset(
        png_dir=args.png_dir,
        img_size=args.img_size,
        in_channels=args.in_channels,
        augment=True,
    )

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print(f"Sampler_train = {sampler_train}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # --- Tensorboard ---
    log_dir = args.log_dir or args.output_dir
    if global_rank == 0 and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    # --- Model ---
    model = TactileMAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
    )

    # --- 加载 CLIP 预训练视觉编码器权重 ---
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print(f"[Pretrained] Loading CLIP weights: {args.pretrained}")
            model.load_clip_pretrained(args.pretrained, verbose=misc.is_main_process())
        else:
            print(f"[Pretrained] 文件不存在, 跳过: {args.pretrained}")

    model.to(device)
    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable params: {n_params / 1e6:.2f}M")

    # --- 学习率: 未显式指定则按有效 batch 由 blr 计算 ---
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr (blr): {args.blr:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    # --- Optimizer (bias / LayerNorm / token / embed 不做 weight decay) ---
    decay_params, no_decay_params = [], []
    for name, param in model_without_ddp.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "token" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        # 每 save_freq 个 epoch 保存 (最后一个 epoch 必存)
        if args.output_dir and (
            epoch % args.save_freq == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        # 每 vis_freq 个 epoch 重建可视化 (仅主进程)
        if args.output_dir and misc.is_main_process() and (
            epoch % args.vis_freq == 0 or epoch + 1 == args.epochs
        ):
            save_reconstruction_vis(
                model, data_loader_train, device, epoch, args.output_dir
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
