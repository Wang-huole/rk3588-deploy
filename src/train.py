"""训练：为指定类别构建 PatchCore memory bank。

用法:
    python -m src.train --category toothbrush
    python -m src.train --config configs/patchcore_config.yaml --category pill
"""
from __future__ import annotations

import argparse
import os

import torch

from .feature_extractor import FeatureExtractor
from .patchcore import PatchCore
from .utils import build_dataloader, load_config, resolve_model_cfg


def train_category(cfg: dict, category: str) -> str:
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("  [警告] CUDA 不可用，回退到 CPU")
        device = "cpu"

    print(f"\n=== 训练类别: {category} ===")
    out_dir = os.path.join(cfg["output"]["dir"], category)
    os.makedirs(out_dir, exist_ok=True)

    # 数据
    train_loader = build_dataloader(cfg, category, split="train")
    print(f"  训练样本数: {len(train_loader.dataset)}")

    # 特征提取器
    extractor = FeatureExtractor(
        backbone_name=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
    )

    # PatchCore（按类别解析超参，支持 model_overrides 覆盖）
    mc = resolve_model_cfg(cfg, category)
    model = PatchCore(
        coreset_ratio=mc["coreset_ratio"],
        n_neighbors=mc["n_neighbors"],
        anomaly_dir=mc["anomaly_dir"],
        device=device,
    )
    model.build_memory_bank(train_loader, extractor)

    save_path = os.path.join(out_dir, "memory_bank.pt")
    model.save(save_path)
    print(f"  Memory Bank 已保存: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", default=None, help="单个类别名；不指定则跑配置里全部")
    args = parser.parse_args()

    cfg = load_config(args.config)
    categories = [args.category] if args.category else cfg["dataset"]["categories"]
    for cat in categories:
        train_category(cfg, cat)


if __name__ == "__main__":
    main()
