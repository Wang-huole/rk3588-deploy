"""板端端到端编排：RKNN/NPU backbone + CPU 分类器（两阶段）。

与 web_app._predict_from_image 的切分一致：
    阶段1  RknnPatchCore（NPU 提特征 + CPU KNN）-> 正常/缺陷 + 异常图
    阶段2  仅当判为缺陷时，调用 DeployClassifier（板端 ONNX Runtime CPU）细分类型

分类器默认走 CPU ONNX（4 通道首层在 NPU 上兼容性见 README 第 6 节）；
若后续把分类器也转成 RKNN，可在本文件里替换 clf 后端。

用法:
    from board_pipeline import BoardPipeline
    pipe = BoardPipeline("pill", deploy_root="../deploy", rknn_root="rknn")
    result = pipe.predict(img_bgr, threshold=None)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from rknn_infer import RknnPatchCore  # noqa: E402

try:
    from classify_infer import DeployClassifier  # 板端需 onnxruntime CPU
    _HAS_ORT = True
except ImportError:
    DeployClassifier = None
    _HAS_ORT = False


class BoardPipeline:
    def __init__(self, category: str, deploy_root: str = "deploy",
                 rknn_root: str | None = None, model_name: str = "backbone_int8.rknn",
                 use_classifier: bool = True):
        self.category = category
        self.detector = RknnPatchCore(category, deploy_root=deploy_root,
                                       rknn_root=rknn_root, model_name=model_name)
        self.cfg = self.detector.cfg

        self.clf = None
        if use_classifier and _HAS_ORT:
            try:
                self.clf = DeployClassifier(category, deploy_root=deploy_root)
                print(f"  [分类器] ONNX CPU 后端已加载 ({self.clf.n_models} 折)")
            except FileNotFoundError:
                print("  [分类器] 无 classify_config.json，回退启发式")
            except Exception as e:
                print(f"  [分类器] 加载失败，回退启发式: {e}")

    def predict(self, img_bgr: np.ndarray, threshold: float | None = None) -> dict:
        threshold = float(threshold if threshold is not None else self.cfg.get("threshold", 0.5))

        t0 = time.time()
        score, amap, det_stats = self.detector.predict(img_bgr)
        patchcore_ms = det_stats["total_ms"]
        patchcore_fps = det_stats["fps"]
        verdict = "缺陷" if score > threshold else "正常"

        defect_type: dict = {"defect_type": "good" if verdict == "正常" else "",
                             "confidence": 0.0, "source": "threshold",
                             "zh": "正常" if verdict == "正常" else "", "probs": {}, "top3": []}

        clf_ms = 0.0
        if verdict == "缺陷" and self.clf is not None:
            tc = time.time()
            try:
                r = self.clf.predict_defect_type(img_bgr, amap)
                defect_type = {"defect_type": r["type"], "confidence": r["conf"],
                               "source": "model", "zh": r["zh"],
                               "probs": r["probs"], "top3": r["top3"]}
            except Exception as e:
                print(f"  [分类器推理失败] {e}")
            clf_ms = (time.time() - tc) * 1000

        total_ms = patchcore_ms + clf_ms
        total_fps = 1000.0 / total_ms if total_ms > 1e-6 else 0.0
        return {
            "category": self.category,
            "score": round(float(score), 4),
            "verdict": verdict,
            "threshold": round(threshold, 4),
            "patchcore_ms": round(patchcore_ms, 1),
            "classifier_ms": round(clf_ms, 1),
            "total_ms": round(total_ms, 1),
            "patchcore_fps": round(patchcore_fps, 1),
            "fps": round(total_fps, 1),
            "npu_ms": round(det_stats.get("npu_ms", 0.0), 1),
            "knn_ms": round(det_stats.get("knn_ms", 0.0), 1),
            "defect_type": defect_type,
            "amap": amap,
        }

    def release(self):
        self.detector.release()


def _main():
    import argparse
    from rknn_infer import imread_unicode

    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--deploy_root", default=str(PROJECT_ROOT / "deploy"))
    ap.add_argument("--rknn_root", default=str(Path(__file__).parent / "rknn"))
    ap.add_argument("--no-classifier", action="store_true")
    args = ap.parse_args()

    pipe = BoardPipeline(args.category, deploy_root=args.deploy_root,
                         rknn_root=args.rknn_root, use_classifier=not args.no_classifier)
    try:
        img = imread_unicode(args.image)
        res = pipe.predict(img, threshold=args.threshold)
        res.pop("amap", None)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    finally:
        pipe.release()


if __name__ == "__main__":
    _main()
