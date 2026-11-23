"""公共工具：配置加载、数据变换、数据加载器。"""
from __future__ import annotations

import os

import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import MVTecADDataset


def load_config(path: str) -> dict:
    """加载 yaml 配置。

    数据集根目录支持用环境变量 PATCHCORE_DATA_ROOT 覆盖，换机器/换数据盘时
    不必改 yaml：

        $env:PATCHCORE_DATA_ROOT = "D:\\datasets\\mvtec_anomaly_detection"
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_root = os.environ.get("PATCHCORE_DATA_ROOT")
    if env_root:
        cfg["dataset"]["root"] = env_root
    return cfg


def resolve_model_cfg(cfg: dict, category: str) -> dict:
    """返回某个类别最终生效的 model 配置。

    支持 config 里的 model_overrides.<category> 按类别覆盖 model 段默认值，
    因为不同类别的最优超参可能不同（例如 anomaly_dir 聚合方式、n_neighbors）。
    """
    merged = dict(cfg["model"])
    overrides = (cfg.get("model_overrides") or {}).get(category) or {}
    merged.update(overrides)
    return merged


def build_transforms(cfg: dict):
    """返回 (image_transform, mask_transform)"""
    size = cfg["dataset"]["image_size"]
    mean = cfg["dataset"]["mean"]
    std = cfg["dataset"]["std"]

    img_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )
    return img_tf, mask_tf


def build_dataloader(cfg: dict, category: str, split: str) -> DataLoader:
    img_tf, mask_tf = build_transforms(cfg)
    ds = MVTecADDataset(
        root=cfg["dataset"]["root"],
        category=category,
        split=split,
        transform=img_tf,
        mask_transform=mask_tf,
        image_size=cfg["dataset"]["image_size"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"] if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(cfg["train"]["device"] == "cuda"),
    )
    return loader
