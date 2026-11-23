"""评估：在测试集上计算 Image/Pixel AUROC、AUPRO，并保存热力图。

用法:
    python -m src.evaluate --category toothbrush
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from .feature_extractor import FeatureExtractor
from .metrics import compute_metrics
from .patchcore import PatchCore
from .utils import build_dataloader, load_config, resolve_model_cfg
from .visualize import save_comparison


def denormalize(img_tensor, mean, std):
    """把 Normalize 后的 tensor 还原为 uint8 numpy (H, W, 3) RGB。"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def evaluate_category(
    cfg: dict,
    category: str,
    n_neighbors: int | None = None,
    anomaly_dir: str | None = None,
) -> dict:
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"\n=== 评估类别: {category} ===")
    out_dir = os.path.join(cfg["output"]["dir"], category)
    heat_dir = os.path.join(out_dir, "heatmaps")
    os.makedirs(heat_dir, exist_ok=True)

    # 加载 memory bank
    bank_path = os.path.join(out_dir, "memory_bank.pt")
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"未找到 {bank_path}，请先运行 train")

    model = PatchCore(device=device)
    model.load(bank_path)
    # 用 config（含 model_overrides）覆盖推理超参，方便调参不重训
    mc = resolve_model_cfg(cfg, category)
    if n_neighbors is not None:
        mc["n_neighbors"] = n_neighbors
    if anomaly_dir is not None:
        mc["anomaly_dir"] = anomaly_dir
    model.n_neighbors = int(mc["n_neighbors"])
    model.anomaly_dir = mc["anomaly_dir"]
    model._build_index()  # n_neighbors 可能变化，重建最近邻索引
    print(f"  已加载 Memory Bank: {bank_path} ({model.memory_bank.shape[0]:,} patches)")
    print(f"  图像级聚合方式: {model.anomaly_dir}，n_neighbors: {model.n_neighbors}")

    extractor = FeatureExtractor(
        backbone_name=cfg["model"]["backbone"],
        layers=cfg["model"]["layers"],
    ).to(device).eval()

    test_loader = build_dataloader(cfg, category, split="test")
    print(f"  测试样本数: {len(test_loader.dataset)}")

    image_scores, image_labels = [], []
    all_maps, all_masks = [], []
    heat_saved = 0

    for batch in tqdm(test_loader, desc=f"评估 {category}"):
        img = batch["image"].to(device)
        mask = batch["mask"].numpy()  # (1, 1, H, W)
        label = int(batch["label"].item())

        feats = extractor(img)
        score, amap = model.predict(feats)

        image_scores.append(score)
        image_labels.append(label)

        # amap (h, w) -> 上采样到 mask 尺寸
        import torch.nn.functional as F
        amap_t = torch.tensor(amap)[None, None]
        amap_up = F.interpolate(amap_t, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        amap_up = amap_up.squeeze().numpy()
        all_maps.append(amap_up)
        all_masks.append(mask.squeeze())

        # 保存热力图（只保存有缺陷的，且数量受限）
        if cfg["output"]["save_heatmap"] and label == 1 and heat_saved < cfg["output"]["heatmap_num"]:
            img_np = denormalize(batch["image"][0], cfg["dataset"]["mean"], cfg["dataset"]["std"])
            gt = mask.squeeze()
            save_comparison(
                os.path.join(heat_dir, f"{heat_saved:03d}_score{score:.3f}.png"),
                img_np,
                amap,
                gt,
            )
            heat_saved += 1

    all_maps = np.stack(all_maps)
    all_masks = np.stack(all_masks)

    metrics = compute_metrics(image_scores, image_labels, all_maps, all_masks)
    print(f"  Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"  AUPRO      : {metrics['aupro']:.4f}")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--n_neighbors", type=int, default=None, help="临时覆盖 n_neighbors（调参用，不写回 config）")
    parser.add_argument("--anomaly_dir", choices=["max", "avg"], default=None, help="临时覆盖图像级聚合方式")
    args = parser.parse_args()

    cfg = load_config(args.config)
    categories = [args.category] if args.category else cfg["dataset"]["categories"]
    for cat in categories:
        evaluate_category(cfg, cat, n_neighbors=args.n_neighbors, anomaly_dir=args.anomaly_dir)


if __name__ == "__main__":
    main()
