"""
训练引擎: 单 epoch 训练循环 + 余弦学习率调度 + MAE 重建可视化。
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import misc


# ----------------------------------------------------------------------
# 学习率调度: warmup + 半周期余弦
# ----------------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    num_steps = len(data_loader)

    for step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if step % accum_iter == 0:
            adjust_learning_rate(optimizer, step / num_steps + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(step + 1) % accum_iter == 0,
        )
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((step / num_steps + epoch) * 1000)
            log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def save_reconstruction_vis(model, data_loader, device, epoch, output_dir, n_samples=9):
    """保存 MAE 重建可视化 (9 行 x 9 列 大网格图)。

    每行一个样本, 9 列逐通道对比:
      1. Orig_depth      原图通道0(depth), viridis colormap
      2. Orig_deform_x   原图通道1(deform_x), RdBu_r diverging colormap
      3. Orig_deform_y   原图通道2(deform_y), RdBu_r diverging colormap
      4. Recon_depth     重建通道0(depth), viridis
      5. Recon_deform_x  重建通道1(deform_x), RdBu_r
      6. Recon_deform_y  重建通道2(deform_y), RdBu_r
      7. Orig_RGB        原图 3 通道合成 RGB
      8. Recon_RGB       重建 3 通道合成 RGB
      9. Error_map       |orig - recon| 的 3 通道均值绝对误差, hot colormap

    Recon_XXX 为混合重建结果: visible patches = 原图, masked patches = 模型重建。
    每个通道的 colormap 范围在整张图内统一 (orig 与 recon 共享同一 vmin/vmax),
    error map 也共用统一 vmax, 便于横向比较。
    """
    model.eval()
    vis_dir = Path(output_dir) / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    samples = next(iter(data_loader)).to(device, non_blocking=True)
    n_samples = min(n_samples, samples.shape[0])
    imgs = samples[:n_samples]  # (N, C, 224, 224)

    loss, pred, mask = model(imgs)

    m = model.module if hasattr(model, "module") else model

    recon = m.unpatchify(pred)
    if m.norm_pix_loss:
        target = m.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        denormed = pred * (var + 1e-6).sqrt() + mean
        recon = m.unpatchify(denormed)

    p = m.patch_size
    grid = m.img_size // p
    mask_2d = mask[:n_samples].reshape(-1, grid, grid)
    mask_img = (
        mask_2d.repeat_interleave(p, dim=1)
        .repeat_interleave(p, dim=2)
        .cpu()
        .numpy()
    )  # (N, 224, 224)

    def to_rgb(arr_chw):
        """(C, H, W) -> (H, W, 3); 取前 3 通道, 直接clip到[0,1]显示(数据已归一化)。"""
        c = arr_chw.shape[0]
        x = arr_chw[:3] if c >= 3 else arr_chw[:1].repeat(3, 0)
        # 数据经 dataset 归一化后已在 [0,1]，直接显示保留真实色彩
        x = x.transpose(1, 2, 0)
        return x.clip(0, 1)

    # 逐样本计算: 原图各通道、混合重建各通道、合成 RGB、error map
    origs = []       # 每个元素: (C, H, W) 原图
    recons = []      # 每个元素: (C, H, W) 混合重建
    orig_rgbs = []   # 每个元素: (H, W, 3)
    recon_rgbs = []  # 每个元素: (H, W, 3)
    error_maps = []  # 每个元素: (H, W)
    for i in range(n_samples):
        orig = imgs[i].cpu().numpy()  # (C, H, W)
        rec = recon[i].clamp(0, 1).cpu().numpy()  # (C, H, W)
        m_np3 = mask_img[i][None]  # (1, H, W) 广播

        # 混合重建: visible 区域用原图, masked 区域用模型重建
        recon_mix = rec * m_np3 + orig * (1 - m_np3)

        orig_rgb = to_rgb(orig)
        recon_rgb = to_rgb(recon_mix)
        err = np.abs(orig_rgb - recon_rgb).mean(axis=2)  # (H, W)

        origs.append(orig)
        recons.append(recon_mix)
        orig_rgbs.append(orig_rgb)
        recon_rgbs.append(recon_rgb)
        error_maps.append(err)

    # 各通道 colormap 范围在整张图内统一 (orig + recon 共享)
    def channel_range(ch):
        vals = [o[ch] for o in origs] + [r[ch] for r in recons]
        vmin = min(float(v.min()) for v in vals)
        vmax = max(float(v.max()) for v in vals)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    n_ch = origs[0].shape[0]
    depth_min, depth_max = channel_range(0)
    dx_min, dx_max = channel_range(1) if n_ch > 1 else (0.0, 1.0)
    dy_min, dy_max = channel_range(2) if n_ch > 2 else (0.0, 1.0)

    # error map 统一 vmax
    err_vmax = max(float(e.max()) for e in error_maps)
    if err_vmax <= 0:
        err_vmax = 1e-6

    col_titles = [
        "Orig_depth", "Orig_deform_x", "Orig_deform_y",
        "Recon_depth", "Recon_deform_x", "Recon_deform_y",
        "Orig_RGB", "Recon_RGB", "Error_map",
    ]
    depth_cmap = "viridis"
    deform_cmap = "RdBu_r"

    fig, axes = plt.subplots(n_samples, 9, figsize=(27, 27))
    if n_samples == 1:
        axes = axes[None, :]  # 保证 axes 始终为 2D

    for i in range(n_samples):
        orig = origs[i]
        rec = recons[i]

        # 列 1-3: 原图逐通道
        axes[i, 0].imshow(orig[0], cmap=depth_cmap, vmin=depth_min, vmax=depth_max)
        if n_ch > 1:
            axes[i, 1].imshow(orig[1], cmap=deform_cmap, vmin=dx_min, vmax=dx_max)
        if n_ch > 2:
            axes[i, 2].imshow(orig[2], cmap=deform_cmap, vmin=dy_min, vmax=dy_max)

        # 列 4-6: 重建逐通道
        axes[i, 3].imshow(rec[0], cmap=depth_cmap, vmin=depth_min, vmax=depth_max)
        if n_ch > 1:
            axes[i, 4].imshow(rec[1], cmap=deform_cmap, vmin=dx_min, vmax=dx_max)
        if n_ch > 2:
            axes[i, 5].imshow(rec[2], cmap=deform_cmap, vmin=dy_min, vmax=dy_max)

        # 列 7-8: 合成 RGB
        axes[i, 6].imshow(orig_rgbs[i])
        axes[i, 7].imshow(recon_rgbs[i])

        # 列 9: error map, 统一 vmax
        axes[i, 8].imshow(error_maps[i], cmap="hot", vmin=0, vmax=err_vmax)

        for j in range(9):
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=14)

    fig.suptitle(
        f"Epoch {epoch} - MAE reconstruction (mask={m.mask_ratio:.0%})",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(vis_dir / f"epoch_{epoch:04d}.png", dpi=100)
    plt.close(fig)

    print(f"[Vis] Saved reconstruction to {vis_dir}")
