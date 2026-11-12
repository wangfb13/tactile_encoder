"""
触觉数据集: data_tac2_s —— 单帧 3 通道 (RGB) uint16 PNG。

每个样本: 单张 PNG (000000000.png ...), 命名为纯序号 (无 f0/f1 配对)。
输出 tensor: (in_channels, 224, 224), float32, 取值约 [0, 1]。

预处理:
  - cv2 读取 16-bit PNG (BGR -> RGB)
  - uint16 -> float32, 逐通道归一化:
      通道0 (depth):       clamp(value / 3000, 0, 1)
      通道1,2 (deform):    clamp((value - 25000) / 10000, 0, 1)
  - nopadding: 直接双线性 resize 到 img_size×img_size (不加 padding)
  - 数据增强: 随机水平/垂直翻转

通道适配:
  - in_channels == 3: 直接使用 3 通道。
  - in_channels == 6: 将 3 通道复制一份拼成 6 通道 (兼容 6 通道模型配置)。
"""

import glob
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TactilePNGDataset(Dataset):
    def __init__(
        self,
        png_dir: str,
        img_size: int = 224,
        in_channels: int = 3,
        augment: bool = True,
    ):
        super().__init__()
        self.png_dir = Path(png_dir)
        self.img_size = img_size
        self.in_channels = in_channels
        self.augment = augment

        if not self.png_dir.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {self.png_dir}")

        # 收集所有 PNG, 按文件名排序 (纯序号命名)
        self.files: List[str] = sorted(
            glob.glob(str(self.png_dir / "*.png"))
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"目录中未找到 PNG: {self.png_dir}")

        print(f"[TactilePNGDataset] dir={self.png_dir}, "
              f"{len(self.files)} images, in_channels={in_channels}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self._read_png(self.files[idx])          # (H, W, 3) uint16, RGB
        # 逐通道归一化:
        #   通道0 (R/depth):       depth = clamp(value / 3000, 0, 1)
        #   通道1,2 (G,B/deform):  deform = clamp((value - 25000) / 10000, 0, 1)
        img = img.astype(np.float32)
        img[:, :, 0] = np.clip(img[:, :, 0] / 3000.0, 0.0, 1.0)
        img[:, :, 1] = np.clip((img[:, :, 1] - 25000.0) / 10000.0, 0.0, 1.0)
        img[:, :, 2] = np.clip((img[:, :, 2] - 25000.0) / 10000.0, 0.0, 1.0)
        img = self._resize(img)                          # (img_size, img_size, 3)
        img = np.transpose(img, (2, 0, 1))               # (3, H, W)

        # 通道适配
        if self.in_channels == 6:
            img = np.concatenate([img, img], axis=0)     # (6, H, W)
        elif self.in_channels != 3:
            raise ValueError(f"不支持的 in_channels={self.in_channels} (数据为 3 通道)")

        if self.augment:
            img = self._augment(img)

        return torch.from_numpy(img.copy())

    @staticmethod
    def _read_png(path: str) -> np.ndarray:
        """读取 16-bit PNG, 返回 (H, W, 3) RGB。"""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取: {path}")
        if img.ndim == 2:  # 灰度兜底 -> 3 通道
            img = np.stack([img, img, img], axis=-1)
        # BGR -> RGB
        return img[:, :, ::-1].copy()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """nopadding: 直接双线性 resize 到 img_size×img_size。"""
        if img.shape[0] == self.img_size and img.shape[1] == self.img_size:
            return img
        return cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        """随机水平/垂直翻转 (RGB 通道, 无 shear 取反逻辑)。"""
        if random.random() < 0.5:
            img = img[:, :, ::-1]  # flip W
        if random.random() < 0.5:
            img = img[:, ::-1, :]  # flip H
        return img
