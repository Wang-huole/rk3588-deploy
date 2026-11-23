"""MVTec AD 数据集加载。

目录结构（标准 MVTec AD）:
    <root>/<category>/
        train/good/xxxx.png
        test/good/xxxx.png
        test/<defect_type>/xxxx.png
        ground_truth/<defect_type>/xxxx_mask.png
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _is_valid_image(path: str) -> bool:
    """快速检查图片能否正常打开（跳过损坏文件）。"""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


class MVTecADDataset(Dataset):
    """MVTec AD 数据集。

    Args:
        root: 数据集根目录，如 C:/.../mvtec_anomaly_detection
        category: 类别名，如 toothbrush / pill
        split: train / test
        transform: 图像预处理
        mask_transform: mask 预处理（仅 test 有缺陷样本时返回）
        image_size: 输出图尺寸
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        assert split in ("train", "test"), f"split 必须是 train/test，得到 {split}"
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_size = image_size

        self.category_dir = self.root / category
        self.img_dir = self.category_dir / split
        self.gt_dir = self.category_dir / "ground_truth"

        # 收集 (图片路径, mask路径或None, label)
        # label: 0=正常(good), 1=有缺陷
        self.samples = []
        skipped = 0
        if split == "train":
            # 训练集只有 good
            good_dir = self.img_dir / "good"
            for p in sorted(good_dir.glob("*.png")):
                if not _is_valid_image(str(p)):
                    print(f"  [警告] 跳过损坏文件: {p}")
                    skipped += 1
                    continue
                self.samples.append((str(p), None, 0))
        else:
            # 测试集: good + 各种缺陷
            for sub in sorted(self.img_dir.iterdir()):
                if not sub.is_dir():
                    continue
                label = 0 if sub.name == "good" else 1
                for img_path in sorted(sub.glob("*.png")):
                    if not _is_valid_image(str(img_path)):
                        print(f"  [警告] 跳过损坏文件: {img_path}")
                        skipped += 1
                        continue
                    if label == 0:
                        mask_path = None
                    else:
                        # ground_truth/<defect>/<name>_mask.png
                        stem = img_path.stem
                        mask_path = str(self.gt_dir / sub.name / f"{stem}_mask.png")
                    self.samples.append((str(img_path), mask_path, label))

        if skipped:
            print(f"  [提示] {category}/{split} 共跳过 {skipped} 个损坏文件")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if self.transform is not None:
            img = self.transform(img)

        mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        if mask_path is not None and os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            if self.mask_transform is not None:
                mask = self.mask_transform(m)
            mask = (mask > 0.5).float()  # 二值化

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "path": img_path,
            "original_size": (h, w),
        }
