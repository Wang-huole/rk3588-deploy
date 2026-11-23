"""精度对照：ONNX(FP32) 与 RKNN(INT8/FP16) 在同一批图上的分数偏差与排序一致性。

在 RK3588 板端运行（需要 .rknn 与原 deploy 包都在板上）：
    python rknn_deploy/compare_precision.py --category pill --input /data/test_images

输出:
    - 每张图两种后端的分数、差值、相对偏差
    - 正常/缺陷分数均值、最大偏差
    - 排序一致性（Spearman 秩相关），以及按 FP32 阈值判定是否一致

注意:
    - 本脚本只做 backbone 输出分数对比，不重训；阈值标定见 README 第 4 节。
    - 若 INT8 分数整体偏移明显，应换现场正常图重新校准，而不是直接比对绝对分。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sys

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from rknn_infer import RknnPatchCore, imread_unicode  # noqa: E402

try:
    from deploy_infer import DeployPatchCore  # FP32 对照后端；板端未传时可缺省
    _HAS_ORT = True
except ImportError:
    DeployPatchCore = None
    _HAS_ORT = False


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--input", required=True, help="图片文件夹")
    ap.add_argument("--deploy_root", default=str(PROJECT_ROOT / "deploy"))
    ap.add_argument("--rknn_root", default=str(Path(__file__).parent / "rknn"))
    ap.add_argument("--model", default="backbone_int8.rknn")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    ort_model = None
    if _HAS_ORT:
        ort_model = DeployPatchCore(args.category, deploy_root=args.deploy_root)
    rknn_model = RknnPatchCore(args.category, deploy_root=args.deploy_root,
                               rknn_root=args.rknn_root, model_name=args.model)
    threshold = args.threshold if args.threshold is not None else float(rknn_model.cfg.get("threshold", 0.5))

    files = sorted(Path(args.input).glob("*.png")) + sorted(Path(args.input).glob("*.jpg"))
    if not files:
        raise SystemExit(f"目录下无图片: {args.input}")

    fp32_scores, rknn_scores, rows = [], [], []
    try:
        for p in files:
            img = imread_unicode(str(p))
            if img is None:
                continue
            s_rknn, _, _ = rknn_model.predict(img)
            if ort_model is not None:
                s_fp32, _ = ort_model.predict(img)
                fp32_scores.append(s_fp32)
                rows.append((p.name, s_fp32, s_rknn, s_rknn - s_fp32))
            else:
                rows.append((p.name, None, s_rknn, None))
            rknn_scores.append(s_rknn)
    finally:
        rknn_model.release()

    rknn_scores = np.array(rknn_scores)
    print(f"\n类别: {args.category}  样本: {len(rows)}  阈值: {threshold:.4f}  模型: {args.model}")

    if ort_model is not None:
        fp32_scores = np.array(fp32_scores)
        diff = rknn_scores - fp32_scores
        rel = np.abs(diff) / (np.abs(fp32_scores) + 1e-6)
        print(f"{'图片':28s} {'FP32':>8s} {'RKNN':>8s} {'Δ':>8s} {'相对偏差':>9s}")
        for name, a, b, d in rows:
            print(f"{name:28s} {a:8.4f} {b:8.4f} {d:8.4f} {abs(d)/(abs(a)+1e-6)*100:8.1f}%")
        print("\n汇总:")
        print(f"  FP32 分数: 均值 {fp32_scores.mean():.4f}, 范围 [{fp32_scores.min():.4f}, {fp32_scores.max():.4f}]")
        print(f"  RKNN 分数: 均值 {rknn_scores.mean():.4f}, 范围 [{rknn_scores.min():.4f}, {rknn_scores.max():.4f}]")
        print(f"  绝对偏差: 均值 {np.abs(diff).mean():.4f}, 最大 {np.abs(diff).max():.4f}")
        print(f"  相对偏差: 均值 {rel.mean()*100:.1f}%, 最大 {rel.max()*100:.1f}%")
        print(f"  排序一致性 Spearman: {rank_corr(fp32_scores, rknn_scores):.4f}")
        fp32_judge = fp32_scores > threshold
        rknn_judge = rknn_scores > threshold
        agree = int((fp32_judge == rknn_judge).sum())
        print(f"  阈值判定一致: {agree}/{len(rows)}")
    else:
        print("  （板端未提供 deploy_infer/ORT，仅输出 RKNN 分数与 FPS，未做 FP32 对照）")
        print(f"  RKNN 分数: 均值 {rknn_scores.mean():.4f}, 范围 [{rknn_scores.min():.4f}, {rknn_scores.max():.4f}]")


if __name__ == "__main__":
    main()
