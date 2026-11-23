"""导出部署包：特征提取器转 ONNX + Memory Bank 转 npy + 配置。

用法（先在训练环境执行）:
    python -m src.export_onnx --category toothbrush
    python -m src.export_onnx --category pill
    python -m src.export_onnx                       # 导出 config 里全部类别

产物（输出到 deploy/<category>/）:
    backbone.onnx    特征提取器（输入 [0,255] 的 float32 图像，输出 layer2/layer3 特征）
    memory_bank.npy  coreset 后的 memory bank（已 L2 归一化）
    config.json      部署推理所需参数（image_size、feature_shape、k、聚合方式等）

部署端运行只需 numpy + onnxruntime + opencv/Pillow，不需要 PyTorch。
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .utils import load_config, resolve_model_cfg


class ExportWrapper(nn.Module):
    """把归一化 + 多层特征提取打包成一个可导出的模型。

    输入: [0,255] 的 float32 图像 (1,3,H,W)
    输出: (layer2_feat, layer3_feat)，尺寸分别为 (1,C2,H2,W2)、(1,C3,H3,W3)
    归一化 (mean/std) 直接固化在计算图里，部署端无需再处理。
    """

    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [0,255] float32 -> [0,1] -> Normalize
        x = x / 255.0
        x = (x - self.mean) / self.std

        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        f2 = b.layer2(x)
        f3 = b.layer3(f2)
        return f2, f3


def export_category(cfg: dict, category: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载训练产物
    train_dir = os.path.join(cfg["output"]["dir"], category)
    bank_path = os.path.join(train_dir, "memory_bank.pt")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"未找到 {bank_path}，请先训练该类别")

    ckpt = torch.load(bank_path, map_location="cpu", weights_only=False)
    memory_bank = ckpt["memory_bank"].astype(np.float32)
    feature_shape = ckpt["feature_shape"]
    # 推理超参以 config（含 model_overrides）为准，保证部署端与评估端口径一致
    mc = resolve_model_cfg(cfg, category)
    anomaly_dir = mc["anomaly_dir"]
    n_neighbors = int(mc["n_neighbors"])

    # 2. 构建可导出模型（加载与训练一致的预训练权重）
    try:
        backbone = FeatureExtractor.LAYER_MAP[cfg["model"]["backbone"]](
            weights="IMAGENET1K_V1"
        )
    except TypeError:
        backbone = FeatureExtractor.LAYER_MAP[cfg["model"]["backbone"]](pretrained=True)
    backbone.eval()

    model = ExportWrapper(
        backbone,
        mean=cfg["dataset"]["mean"],
        std=cfg["dataset"]["std"],
    ).to(device).eval()

    # 3. 导出 ONNX
    deploy_dir = os.path.join("deploy", category)
    os.makedirs(deploy_dir, exist_ok=True)
    onnx_path = os.path.join(deploy_dir, "backbone.onnx")

    size = cfg["dataset"]["image_size"]
    dummy = torch.randn(1, 3, size, size).to(device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["feat_layer2", "feat_layer3"],
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"  ONNX 已导出: {onnx_path}")

    # 4. 保存 memory bank 与配置
    np.save(os.path.join(deploy_dir, "memory_bank.npy"), memory_bank)
    deploy_cfg = {
        "category": category,
        "backbone": cfg["model"]["backbone"],
        "image_size": size,
        "feature_shape": list(feature_shape),
        "mean": cfg["dataset"]["mean"],
        "std": cfg["dataset"]["std"],
        "n_neighbors": n_neighbors,
        "anomaly_dir": anomaly_dir,
        "memory_bank_size": int(memory_bank.shape[0]),
        "feature_dim": int(memory_bank.shape[1]),
    }
    with open(os.path.join(deploy_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(deploy_cfg, f, indent=2, ensure_ascii=False)
    print(f"  Memory Bank: {memory_bank.shape[0]:,} patches × {memory_bank.shape[1]} dims")
    print(f"  部署包就绪: {deploy_dir}")
    return deploy_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    categories = [args.category] if args.category else cfg["dataset"]["categories"]
    for cat in categories:
        print(f"\n=== 导出类别: {cat} ===")
        export_category(cfg, cat)


if __name__ == "__main__":
    main()
