"""
验证 engine.py 中 to_rgb() 逐通道归一化修复效果的测试脚本。

加载最新 checkpoint 与一批数据, 调用修复后的 save_reconstruction_vis(),
生成可视化图片到 output/vis_fixed/ 目录。

用法 (在 vit-b-tac2_s 目录下):
    python test_vis.py
"""

import os
import sys
import glob
import shutil
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 保证可以 import 同目录下的扁平模块
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model import TactileMAE
from dataset import TactilePNGDataset
from engine import save_reconstruction_vis


PNG_DIR = "/mnt/data/xidong_data/lerobot_tactile_ws/AnyTouch/zxd/data/data_tac2_s"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
N_SAMPLES = 4


def find_latest_checkpoint(output_dir):
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"未找到 checkpoint: {output_dir}")
    # 按 epoch 序号取最新
    def epoch_of(p):
        name = os.path.basename(p)
        return int(name.replace("checkpoint-", "").replace(".pth", ""))
    return max(ckpts, key=epoch_of)


def main():
    # 用 CPU 验证, 避免占用训练中的 GPU 显存 / 干扰训练
    device = torch.device("cpu")

    # --- 模型 (与 train.sh / config 默认一致) ---
    model = TactileMAE(
        img_size=224,
        patch_size=16,
        in_channels=3,
        encoder_dim=768,
        encoder_layers=12,
        encoder_heads=12,
        decoder_dim=384,
        decoder_layers=4,
        decoder_heads=6,
        mask_ratio=0.75,
        norm_pix_loss=True,
    )

    ckpt_path = find_latest_checkpoint(OUTPUT_DIR)
    print(f"[test_vis] 加载最新 checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[test_vis] missing keys: {len(missing)}")
    if unexpected:
        print(f"[test_vis] unexpected keys: {len(unexpected)}")
    epoch = ckpt.get("epoch", 0)
    model.to(device)
    model.eval()

    # --- 数据 (关闭增强, 便于与原图直接对比) ---
    dataset = TactilePNGDataset(
        png_dir=PNG_DIR,
        img_size=224,
        in_channels=3,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=N_SAMPLES,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    # save_reconstruction_vis 固定写入 {output_dir}/vis,
    # 用临时目录承接后再移动到 output/vis_fixed/
    tmp_dir = tempfile.mkdtemp(prefix="vis_fixed_")
    save_reconstruction_vis(
        model, loader, device, epoch, tmp_dir, n_samples=N_SAMPLES
    )

    fixed_dir = Path(OUTPUT_DIR) / "vis_fixed"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    for png in glob.glob(os.path.join(tmp_dir, "vis", "*.png")):
        shutil.copy(png, fixed_dir / os.path.basename(png))
    shutil.rmtree(tmp_dir, ignore_errors=True)

    out_files = sorted(glob.glob(str(fixed_dir / "*.png")))
    print(f"[test_vis] 已生成 {len(out_files)} 张可视化到: {fixed_dir}")
    for f in out_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
