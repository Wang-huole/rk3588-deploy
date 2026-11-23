"""评估指标：Image AUROC、Pixel AUROC、AUPRO。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(scores: List[float], labels: List[int]) -> float:
    """图像级 AUROC：整张图是否有缺陷。"""
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(anomaly_maps: np.ndarray, masks: np.ndarray) -> float:
    """像素级 AUROC：每个像素是否是缺陷。

    Args:
        anomaly_maps: (N, H, W) 预测异常图
        masks: (N, H, W) 二值 GT mask
    """
    y_score = anomaly_maps.reshape(-1)
    y_true = masks.reshape(-1).astype(int)
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def aupro(anomaly_maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    """AUPRO：FPR<=fpr_limit 下的 Per-Region Overlap 曲线下面积。

    对每个 GT 连通区域 r，PRO(threshold) = mean_r( |pred ∩ r| / |r| )，
    再在 FPR ∈ [0, fpr_limit] 上积分并归一化（除以 fpr_limit）。
    """
    import cv2

    y_score = anomaly_maps.reshape(-1).astype(np.float64)
    y_true = masks.reshape(-1).astype(np.uint8)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")

    n_pixels = len(y_true)
    n_bg = n_pixels - int(y_true.sum())

    # 为每个像素分配区域 id（背景为 -1），并记录各区域面积
    region_id = np.full(n_pixels, -1, dtype=np.int32)
    areas = []
    rid = 0
    H, W = masks.shape[1], masks.shape[2]
    for i in range(masks.shape[0]):
        m = masks[i].astype(np.uint8)
        if m.max() == 0:
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for r in range(1, num):  # 跳过背景 label 0
            area = int(stats[r, cv2.CC_STAT_AREA])
            flat_idx = i * H * W + np.where(labels.reshape(-1) == r)[0]
            region_id[flat_idx] = rid
            areas.append(area)
            rid += 1

    if rid == 0:
        return float("nan")
    areas = np.asarray(areas, dtype=np.float64)

    # 阈值集合（去重 + 降采样，避免过多点拖慢计算）
    thresholds = np.unique(y_score)
    if len(thresholds) > 2000:
        idx = np.linspace(0, len(thresholds) - 1, 2000).astype(int)
        thresholds = thresholds[idx]

    fprs, pros = [], []
    region_mask = region_id >= 0
    for t in thresholds:
        pred = y_score >= t
        fp = int(np.count_nonzero(pred & (y_true == 0)))
        fpr = fp / n_bg
        # 每个区域的 TP 像素数 -> overlap = TP / area
        tp_per_region = np.bincount(region_id[region_mask & pred], minlength=rid)
        pro = float(np.mean(tp_per_region / areas))
        fprs.append(fpr)
        pros.append(pro)

    if len(fprs) < 1:
        return float("nan")

    order = np.argsort(fprs)
    fprs = np.asarray(fprs)[order]
    pros = np.asarray(pros)[order]

    # 只积分 FPR ∈ [0, fpr_limit]；若曲线未达到上限，用最后一个 PRO 补齐到 fpr_limit
    keep = fprs <= fpr_limit
    fprs_c = fprs[keep]
    pros_c = pros[keep]
    if len(fprs_c) == 0:
        return float("nan")
    if fprs_c[-1] < fpr_limit:
        fprs_c = np.append(fprs_c, fpr_limit)
        pros_c = np.append(pros_c, pros_c[-1])

    area = np.trapz(pros_c, fprs_c)
    return float(area / fpr_limit)


def compute_metrics(
    image_scores: List[float],
    image_labels: List[int],
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, float]:
    return {
        "image_auroc": image_auroc(image_scores, image_labels),
        "pixel_auroc": pixel_auroc(anomaly_maps, masks),
        "aupro": aupro(anomaly_maps, masks),
    }
