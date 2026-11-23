"""单图推理：对一张图片判断是否有缺陷并输出热力图。

用法:
    python -m src.infer --category pill --image path/to/test.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from .feature_extractor import FeatureExtractor
from .patchcore import PatchCore
from .utils import build_transforms, load_config
from .visualize import overlay_heatmap


def infer(cfg: dict, category: str, image_path: str, out_path: str | None = None):
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = os.path.join(cfg["output"]["dir"], category)
    bank_path = os.path.join(out_dir, "memory_bank.pt")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"未找到 {bank_path}，请先运行 train")

    model = PatchCore(device=device)
    model.load(bank_path)

    extractor = FeatureExtractor(
        backbone_name=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
    ).to(device).eval()

    size = cfg["dataset"]["image_size"]
    img_tf, _ = build_transforms(cfg)

    img_pil = Image.open(image_path).convert("RGB")
    x = img_tf(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = extractor(x)
        score, amap = model.predict(feats)

    # 用原图做叠加
    img_np = np.array(img_pil.resize((size, size)))
    vis = overlay_heatmap(img_np, amap, mask=None)

    if out_path is None:
        out_path = os.path.join(out_dir, "infer_output.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(vis).save(out_path)

    print(f"图像异常分数: {score:.4f}")
    print(f"可视化结果已保存: {out_path}")
    return score, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer(cfg, args.category, args.image, args.out)


if __name__ == "__main__":
    main()
