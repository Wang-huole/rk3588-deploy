"""缺陷热力图可视化。"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def overlay_heatmap(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    color_order: str = "rgb",
) -> np.ndarray:
    """把异常热力图叠加到原图上（训练端与部署端共用同一实现）。

    Args:
        image: (H, W, 3) uint8 原图，通道顺序由 color_order 指定
        anomaly_map: (h, w) 异常分数（任意范围，会归一化到 0~255）
        mask: 可选 GT mask (h, w) 0/1
        alpha: 热力图透明度
        color_order: "rgb"（训练/评估，PIL）或 "bgr"（部署，OpenCV）

    Returns:
        (H, W, 3) uint8 叠加结果，通道顺序与输入一致
    """
    if color_order not in ("rgb", "bgr"):
        raise ValueError(f"color_order 必须是 'rgb' 或 'bgr'，收到 {color_order!r}")

    h, w = image.shape[:2]
    # 归一化
    amap = anomaly_map.astype(np.float32)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    amap_u8 = (amap * 255).astype(np.uint8)
    # 上采样到原图大小
    amap_resized = cv2.resize(amap_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    # 伪彩色（JET，applyColorMap 输出 BGR）
    heat = cv2.applyColorMap(amap_resized, cv2.COLORMAP_JET)
    if color_order == "rgb":
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heat, alpha, 0)

    if mask is not None:
        m = (mask.astype(np.float32) * 255).astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        # GT mask 用绿色轮廓画
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def save_comparison(
    out_path: str,
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> None:
    """保存 3 列对比图：原图 | GT mask | 热力图叠加。"""
    h, w = image.shape[:2]

    # GT mask 列
    if mask is not None:
        m = (mask.astype(np.float32) * 255).astype(np.uint8)
        m_rgb = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_rgb = cv2.cvtColor(m_rgb, cv2.COLOR_GRAY2RGB)
    else:
        m_rgb = np.zeros_like(image)

    overlay = overlay_heatmap(image, anomaly_map, mask=None)

    row = np.concatenate([image, m_rgb, overlay], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(row).save(out_path)
