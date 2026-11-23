"""板端 ONNX Runtime CPU 推理器（NPU 不可用 / 尚无 .rknn 时的兜底）。

与 rknn_infer.RknnPatchCore 接口一致:
    model = OrtPatchCore("toothbrush", deploy_root="../deploy")
    score, amap, stats = model.predict(img_bgr)

后端分工:
    - backbone 特征提取: ONNX Runtime CPU（deploy/<cat>/backbone.onnx）
    - memory bank 最近邻: CPU numpy
    - 无需 PyTorch、无需 rknnlite；板端依赖 numpy + onnxruntime + opencv + pillow

用法（板端）:
    python rknn_deploy/ort_infer.py --category toothbrush --image /data/a.png
    python rknn_deploy/ort_infer.py --category pill --input /data/images --out outputs/ort_demo
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
from visualize import overlay_heatmap  # noqa: E402


def np_bilinear_resize(feat: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """numpy 双线性插值上采样特征图 (C,H,W) -> (C,out_h,out_w)。"""
    C, H, W = feat.shape
    if (H, W) == (out_h, out_w):
        return feat
    sy, sx = H / out_h, W / out_w
    ys = np.clip((np.arange(out_h) + 0.5) * sy - 0.5, 0, H - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * sx - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    out = (
        feat[:, y0][:, :, x0] * (1 - wy) * (1 - wx)
        + feat[:, y0][:, :, x1] * (1 - wy) * wx
        + feat[:, y1][:, :, x0] * wy * (1 - wx)
        + feat[:, y1][:, :, x1] * wy * wx
    )
    return out.astype(np.float32)


def imread_unicode(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class OrtPatchCore:
    def __init__(self, category: str, deploy_root: str = "deploy"):
        deploy_dir = Path(deploy_root) / category
        cfg_path = deploy_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"缺少配置: {cfg_path}")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        self.bank = np.load(deploy_dir / "memory_bank.npy")
        self.bank_norm = (self.bank ** 2).sum(axis=1)

        onnx_path = deploy_dir / "backbone.onnx"
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]

        self.size = self.cfg["image_size"]
        self.fh, self.fw = self.cfg["feature_shape"]
        self.k = self.cfg["n_neighbors"]
        print(f"  [ONNX/CPU] {onnx_path.name} 输入 {self.size}x{self.size}")

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(pil, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1))[None]  # (1,3,H,W)
        return np.ascontiguousarray(x)

    def extract(self, x: np.ndarray) -> np.ndarray:
        outs = self.sess.run(self.out_names, {self.input_name: x})
        f2, f3 = outs[0][0], outs[1][0]  # CHW
        th, tw = f2.shape[1], f2.shape[2]
        if f3.shape[1] != th or f3.shape[2] != tw:
            f3 = np_bilinear_resize(f3, th, tw)
        cat = np.concatenate([f2, f3], axis=0)
        C, H, W = cat.shape
        patches = cat.transpose(1, 2, 0).reshape(H * W, C)
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-10
        return (patches / norms).astype(np.float32)

    def knn_scores(self, patches: np.ndarray, block: int = 2048) -> np.ndarray:
        P = patches.shape[0]
        q = np.ascontiguousarray(patches.astype(np.float32))
        M = self.bank.shape[0]
        p_norm = (q ** 2).sum(axis=1)
        scores = np.empty(P, dtype=np.float32)
        for s in range(0, P, block):
            e = min(s + block, P)
            pb = q[s:e]
            d2 = p_norm[s:e, None] + self.bank_norm[None, :] - 2.0 * (pb @ self.bank.T)
            np.maximum(d2, 0, out=d2)
            d = np.sqrt(d2)
            kk = min(self.k, M)
            idx = np.argpartition(d, kth=kk - 1, axis=1)[:, :kk]
            scores[s:e] = np.take_along_axis(d, idx, axis=1).mean(axis=1)
        return scores

    def predict(self, img: np.ndarray) -> tuple[float, np.ndarray, dict]:
        t0 = time.time()
        x = self.preprocess(img)
        t_pre = time.time()
        patches = self.extract(x)
        t_ext = time.time()
        patch_scores = self.knn_scores(patches)
        t_knn = time.time()
        anomaly_map = patch_scores.reshape(self.fh, self.fw)
        if self.cfg["anomaly_dir"] == "max":
            image_score = float(patch_scores.max())
        else:
            kk = max(1, int(0.01 * len(patch_scores)))
            image_score = float(np.sort(patch_scores)[-kk:].mean())

        total_ms = (t_knn - t0) * 1000
        stats = {
            "total_ms": total_ms,
            "fps": 1000.0 / total_ms if total_ms > 1e-6 else 0.0,
            "preprocess_ms": (t_pre - t0) * 1000,
            "backend_ms": (t_ext - t_pre) * 1000,
            "knn_ms": (t_knn - t_ext) * 1000,
        }
        return image_score, anomaly_map, stats

    @staticmethod
    def overlay(img: np.ndarray, amap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        return overlay_heatmap(img, amap, alpha=alpha, color_order="bgr")

    def release(self):
        pass


def _threshold(model: OrtPatchCore, override: float | None) -> float:
    if override is not None:
        return override
    return float(model.cfg.get("threshold", 0.5))


def infer_single(model: OrtPatchCore, img_path: str, out_path: str | None,
                 threshold: float, save_vis: bool) -> dict:
    img = imread_unicode(img_path)
    if img is None:
        with Image.open(img_path) as pil:
            img = np.asarray(pil.convert("RGB"))[:, :, ::-1].copy()
    if img is None or img.size == 0:
        raise ValueError(f"无法读取图片: {img_path}")

    score, amap, stats = model.predict(img)
    ms, fps = stats["total_ms"], stats["fps"]
    label = "缺陷" if score > threshold else "正常"
    label_en = "NG" if score > threshold else "OK"
    print(f"  [{Path(img_path).name}] score={score:.4f} -> {label}  "
          f"(推理 {ms:.0f}ms, {fps:.1f} FPS |  backbone {stats['backend_ms']:.0f}ms, "
          f"KNN {stats['knn_ms']:.0f}ms)")

    if save_vis and out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vis = model.overlay(img, amap)
        cv2.putText(vis, f"score={score:.3f} {label_en}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if label == "正常" else (0, 0, 255), 2)
        cv2.putText(vis, f"{ms:.0f}ms {fps:.1f} FPS", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imwrite(out_path, vis)
    return {"path": img_path, "score": score, "label": label, "ms": ms, "fps": fps, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--deploy_root", default=str(PROJECT_ROOT / "deploy"))
    args = ap.parse_args()

    model = OrtPatchCore(args.category, deploy_root=args.deploy_root)
    threshold = _threshold(model, args.threshold)
    print(f"已加载: bank={model.cfg['memory_bank_size']:,} × dim {model.cfg['feature_dim']}, "
          f"输入 {model.size}×{model.size}, k={model.k}, 阈值 {threshold:.4f}")

    try:
        if args.image:
            out = args.out or str(PROJECT_ROOT / "outputs" / "ort_demo" / f"{Path(args.image).stem}_vis.png")
            infer_single(model, args.image, out, threshold, True)
            print(f"可视化已保存: {out}")
        elif args.input:
            out_dir = args.out or str(PROJECT_ROOT / "outputs" / "ort_demo" / args.category)
            results = []
            for p in sorted(Path(args.input).glob("*.png")) + sorted(Path(args.input).glob("*.jpg")):
                out = os.path.join(out_dir, p.stem + "_vis.png")
                results.append(infer_single(model, str(p), out, threshold, True))
            good = sum(1 for r in results if r["label"] == "正常")
            avg_ms = float(np.mean([r["ms"] for r in results])) if results else 0.0
            avg_fps = 1000.0 / avg_ms if avg_ms > 1e-6 else 0.0
            print(f"\n批量完成: {len(results)} 张（正常 {good} / 缺陷 {len(results) - good}）")
            print(f"平均耗时 {avg_ms:.0f}ms/帧, 平均 {avg_fps:.1f} FPS "
                  f"(峰值 {max(r['fps'] for r in results):.1f} / 最低 {min(r['fps'] for r in results):.1f})")
        else:
            ap.error("必须提供 --image 或 --input")
    finally:
        model.release()


if __name__ == "__main__":
    main()
