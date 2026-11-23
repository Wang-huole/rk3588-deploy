"""PatchCore 部署推理（不依赖 PyTorch）。

仅需: numpy + onnxruntime + opencv-python + Pillow
数据: deploy/<category>/ 下的 backbone.onnx、memory_bank.npy、config.json

用法:
    # 单图推理
    python deploy_infer.py --category toothbrush --image path/to/img.png

    # 文件夹批量推理（输出到 --out 目录）
    python deploy_infer.py --category pill --input path/to/folder --out outputs/deploy_demo

    # 指定阈值判定（score > threshold 判为缺陷）；不填则读 config.json 里标定好的阈值
    python deploy_infer.py --category toothbrush --image x.png --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# 可视化与训练端共用同一实现（src/visualize.py 仅依赖 numpy/cv2/PIL，不引入 PyTorch）
from src.visualize import overlay_heatmap


def _try_import_faiss():
    """尝试导入 faiss（可选依赖），失败返回 None。"""
    try:
        import faiss
        return faiss
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def np_bilinear_resize(feat: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """numpy 双线性插值上采样特征图（与训练端 F.interpolate align_corners=False 对齐）。

    Args:
        feat: (C, H, W) float32
    Returns:
        (C, out_h, out_w)
    """
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
    wy = (ys - y0)[:, None]  # (out_h, 1)
    wx = (xs - x0)[None, :]  # (1, out_w)
    out = (
        feat[:, y0][:, :, x0] * (1 - wy) * (1 - wx)
        + feat[:, y0][:, :, x1] * (1 - wy) * wx
        + feat[:, y1][:, :, x0] * wy * (1 - wx)
        + feat[:, y1][:, :, x1] * wy * wx
    )
    return out.astype(np.float32)


def imread_unicode(path: str) -> np.ndarray | None:
    """读取图片（支持中文/特殊字符路径）。返回 BGR 数组或 None。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ---------------------------------------------------------------------------
# 部署包加载
# ---------------------------------------------------------------------------
class DeployPatchCore:
    """ONNX + Memory Bank 部署推理器。"""

    def __init__(self, category: str, deploy_root: str = "deploy"):
        deploy_dir = Path(deploy_root) / category
        self.deploy_dir = str(deploy_dir)  # 供标定等逻辑写回 config.json 使用
        self.onnx_path = deploy_dir / "backbone.onnx"
        self.bank_path = deploy_dir / "memory_bank.npy"
        cfg_path = deploy_dir / "config.json"
        for p in (self.onnx_path, self.bank_path, cfg_path):
            if not p.exists():
                raise FileNotFoundError(f"缺少部署文件: {p}（请先运行 python -m src.export_onnx）")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.bank = np.load(self.bank_path)  # (M, D) 已 L2 归一化
        self.bank_norm = (self.bank ** 2).sum(axis=1)  # 预计算 ||b||^2

        # Faiss 可选后端（比 numpy argpartition 快）；未安装则回退 numpy
        self.faiss = _try_import_faiss()
        self._faiss_index = None
        if self.faiss is not None:
            self._faiss_index = self.faiss.IndexFlatL2(self.bank.shape[1])
            self._faiss_index.add(np.ascontiguousarray(self.bank.astype(np.float32)))

        # 推理后端：优先 TensorRT（若已导出 engine 且 tensorrt+torch 可用），否则 ONNX Runtime
        # 注意：TRT 分支需要 tensorrt 和 torch 才能分配 GPU 显存（本项目未验证该分支），
        # 缺任一依赖会自动打印原因并回退 ONNX Runtime，不影响正常部署。
        self._trt = None  # dict 或 None
        self.onnx_path = deploy_dir / "backbone.onnx"
        engine_path = deploy_dir / "backbone_fp16.engine"
        if engine_path.exists():
            self._trt = self._load_trt_engine(engine_path)

        if self._trt is not None:
            # 从 engine 推导输入/输出名
            names = self._trt["names"]
            engine = self._trt["engine"]
            self.input_name, self.out_names = self._io_names(engine, names)
            print(f"  推理后端: TensorRT (engine: {engine_path.name})")
        else:
            # ONNX Runtime：优先 GPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self.sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
            except Exception:
                self.sess = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.out_names = [o.name for o in self.sess.get_outputs()]
            print(f"  推理后端: ONNX Runtime ({self.sess.get_providers()[0]})")

        self.size = self.cfg["image_size"]
        self.fh, self.fw = self.cfg["feature_shape"]  # 特征图尺寸 (H, W)
        self.k = self.cfg["n_neighbors"]

    # ---- 预处理 ----
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """BGR 图像 -> RGB [0,255] float32 NCHW，resize 到 image_size。

        与训练端 transforms.Compose([Resize, ToTensor]) 保持一致：
        PIL BILINEAR 插值 + 像素值 [0,255]（Normalize 已固化在 ONNX 图内）。
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = pil.resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(pil, dtype=np.float32)  # (H, W, 3) [0,255]
        x = np.transpose(x, (2, 0, 1))[None]  # (1, 3, H, W)
        return np.ascontiguousarray(x)

    # ---- 特征提取 ----
    def extract(self, x: np.ndarray) -> np.ndarray:
        """推理 -> 对齐尺寸 concat 的 patch 特征 (P, C)。"""
        if self._trt is not None:
            f2, f3 = self._extract_trt(x)
        else:
            f2, f3 = self._extract_onnx(x)

        # 对齐到最大空间尺寸（同训练端 _stack_patch_features）
        th, tw = f2.shape[2], f2.shape[3]
        if f3.shape[2] != th or f3.shape[3] != tw:
            f3 = np_bilinear_resize(f3[0], th, tw)[None]  # (1, C, th, tw)

        cat = np.concatenate([f2, f3], axis=1)  # (1, C, H, W)
        B, C, H, W = cat.shape
        patches = cat.transpose(0, 2, 3, 1).reshape(B, H * W, C)[0]  # (P, C)
        # L2 归一化（与训练一致）
        norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-10
        patches = patches / norms
        return patches

    def _extract_onnx(self, x: np.ndarray):
        outs = self.sess.run(self.out_names, {self.input_name: x})
        return outs  # (f2, f3)

    def _extract_trt(self, x: np.ndarray):
        import torch

        trt = self._trt
        buffers = trt["buffers"]
        ctx = trt["ctx"]
        names = trt["names"]
        buffers[self.input_name].copy_(
            torch.from_numpy(np.ascontiguousarray(x)).cuda()
        )
        bindings = [int(buffers[n].data_ptr()) for n in names]
        ctx.execute_v2(bindings)
        f2 = buffers[self.out_names[0]].cpu().numpy()
        f3 = buffers[self.out_names[1]].cpu().numpy()
        return f2, f3

    # ---- TensorRT 引擎加载 ----
    def _load_trt_engine(self, engine_path: Path):
        """加载 TensorRT engine 并预分配 GPU buffer，失败返回 None（自动回退 ONNX）。"""
        try:
            import tensorrt as trt
            import torch  # 用于 GPU 内存分配（deploy 端可选）
        except ImportError:
            return None
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            ctx = engine.create_execution_context()

            # I/O 名称（兼容 TRT 8/10）
            try:
                names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            except AttributeError:
                names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]

            # 预分配 GPU buffer（输入输出均为 float32）
            buffers = {}
            for name in names:
                try:
                    shape = tuple(ctx.get_tensor_shape(name))
                except AttributeError:
                    shape = tuple(ctx.get_binding_shape(name))
                buffers[name] = torch.empty(shape, dtype=torch.float32, device="cuda")

            return {"engine": engine, "ctx": ctx, "buffers": buffers, "names": names}
        except Exception as e:
            print(f"  [TRT 加载失败，回退 ONNX] {type(e).__name__}: {str(e)[:80]}")
            return None

    @staticmethod
    def _io_names(engine, names):
        """从 engine 区分输入/输出 tensor 名，返回 (input_name, [output_names])。"""
        import tensorrt as trt

        inputs, outputs = [], []
        for name in names:
            try:
                mode = engine.get_tensor_mode(name)
                is_input = mode == trt.TensorIOMode.INPUT
            except AttributeError:
                is_input = engine.binding_is_input(name)
            (inputs if is_input else outputs).append(name)
        return inputs[0], outputs

    # ---- KNN 异常分数 ----
    def knn_scores(self, patches: np.ndarray, block: int = 2048) -> np.ndarray:
        """计算每个 patch 到 memory bank 的 k 近邻平均距离。

        优先用 Faiss（精确 L2），未安装则回退 numpy 分块 argpartition。
        """
        P = patches.shape[0]
        q = np.ascontiguousarray(patches.astype(np.float32))

        if self._faiss_index is not None:
            # Faiss 返回 L2 距离的平方，开方后与 numpy 实现一致
            d2, _ = self._faiss_index.search(q, self.k)  # (P, k)
            return np.sqrt(np.maximum(d2, 0.0)).mean(axis=1).astype(np.float32)

        # ---- numpy 回退实现 ----
        M = self.bank.shape[0]
        p_norm = (patches ** 2).sum(axis=1)  # (P,)
        scores = np.empty(P, dtype=np.float32)

        for s in range(0, P, block):
            e = min(s + block, P)
            pb = patches[s:e]  # (B, D)
            d2 = p_norm[s:e, None] + self.bank_norm[None, :] - 2.0 * (pb @ self.bank.T)
            np.maximum(d2, 0, out=d2)
            d = np.sqrt(d2)  # (B, M)
            # 取前 k 个最小距离求平均
            idx = np.argpartition(d, kth=min(self.k, M) - 1, axis=1)[:, : self.k]
            scores[s:e] = np.take_along_axis(d, idx, axis=1).mean(axis=1)
        return scores

    # ---- 推理 ----
    def predict(self, img: np.ndarray) -> tuple[float, np.ndarray]:
        """返回 (image_score, anomaly_map (h,w) float32)。"""
        x = self.preprocess(img)
        patches = self.extract(x)  # (P, C)
        patch_scores = self.knn_scores(patches)  # (P,)

        h, w = self.fh, self.fw
        anomaly_map = patch_scores.reshape(h, w)

        if self.cfg["anomaly_dir"] == "max":
            image_score = float(patch_scores.max())
        else:  # avg: top-1% 平均
            kk = max(1, int(0.01 * len(patch_scores)))
            image_score = float(np.sort(patch_scores)[-kk:].mean())
        return image_score, anomaly_map

    # ---- 可视化 ----
    @staticmethod
    def overlay(img: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """热力图叠加到原图（BGR）。

        直接复用 src/visualize.overlay_heatmap，避免训练端/部署端两份实现漂移。
        """
        return overlay_heatmap(img, anomaly_map, alpha=alpha, color_order="bgr")


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def infer_single(model: DeployPatchCore, img_path: str, out_path: str | None = None,
                 threshold: float = 0.5, save_vis: bool = True) -> dict:
    img = imread_unicode(str(img_path))
    if img is None:  # 兜底：OpenCV 解不了时用 PIL（例如某些 CMYK/EXIF 异常图）
        with Image.open(str(img_path)) as pil:
            img = np.asarray(pil.convert("RGB"))[:, :, ::-1].copy()
    if img is None or img.size == 0:
        raise ValueError(f"无法读取图片: {img_path}")

    t0 = time.time()
    score, amap = model.predict(img)
    ms = (time.time() - t0) * 1000

    label = "缺陷" if score > threshold else "正常"
    print(f"  [{Path(img_path).name}] score={score:.4f} -> {label}  (推理 {ms:.0f}ms)")

    if save_vis and out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        vis = model.overlay(img, amap)
        # 加文字标注
        text = f"score={score:.3f} {label}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if label == "正常" else (0, 0, 255), 2)
        cv2.imwrite(out_path, vis)
    return {"path": str(img_path), "score": score, "label": label, "ms": ms}


def infer_folder(model: DeployPatchCore, folder: str, out_dir: str, threshold: float = 0.5) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for p in sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg")):
        out = os.path.join(out_dir, p.stem + "_vis.png")
        r = infer_single(model, str(p), out, threshold)
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="类别名（对应 deploy/<category>）")
    parser.add_argument("--image", default=None, help="单张图片路径")
    parser.add_argument("--input", default=None, help="批量图片文件夹")
    parser.add_argument("--out", default=None, help="可视化输出目录/文件")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值；不填则用 config.json 里 calibrate.py 标定好的阈值")
    parser.add_argument("--deploy_root", default="deploy")
    args = parser.parse_args()

    model = DeployPatchCore(args.category, deploy_root=args.deploy_root)
    print(f"已加载部署包: {args.category}（bank={model.cfg['memory_bank_size']:,} × dim {model.cfg['feature_dim']}，"
          f"输入 {model.size}×{model.size}，k={model.k}）")

    # 阈值：显式传入 > 部署包内标定值 > 默认 0.5
    threshold = args.threshold
    if threshold is None and "threshold" in model.cfg:
        threshold = float(model.cfg["threshold"])
        print(f"判定阈值: {threshold:.4f}（来源: config.json 标定值，basis={model.cfg.get('threshold_basis', '-')}）")
    elif threshold is None:
        threshold = 0.5
        print(f"警告: 部署包内无标定阈值，使用默认 {threshold:.4f}；建议先运行 python calibrate.py")

    if args.image:
        out = args.out or os.path.join("outputs", "deploy_demo", f"{Path(args.image).stem}_vis.png")
        infer_single(model, args.image, out, threshold)
        print(f"可视化已保存: {out}")
    elif args.input:
        out_dir = args.out or os.path.join("outputs", "deploy_demo", args.category)
        results = infer_folder(model, args.input, out_dir, threshold)
        good = sum(1 for r in results if r["label"] == "正常")
        print(f"\n批量完成: {len(results)} 张（正常 {good} / 缺陷 {len(results) - good}）")
        print(f"可视化已保存: {out_dir}")
    else:
        parser.error("必须提供 --image 或 --input")


if __name__ == "__main__":
    main()
