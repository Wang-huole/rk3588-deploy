"""缺陷类型分类器部署推理（不依赖 PyTorch）。

仅需: numpy + onnxruntime + opencv-python + Pillow
数据: deploy/<category>/classify_fold*.onnx + classify_config.json

输入为 4 通道：RGB(3) + PatchCore 异常热力图(1)。
推理时把多折模型的 softmax 概率取平均（集成），更稳。

用法:
    from classify_infer import DeployClassifier
    clf = DeployClassifier("pill", deploy_root="deploy")
    res = clf.predict(img_bgr, amap)      # {"probs": {...}, "top": "scratch", "conf": 0.87}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeployClassifier:
    """缺陷类型分类器（多折 ONNX 集成）。"""

    def __init__(self, category: str, deploy_root: str = "deploy"):
        deploy_dir = Path(deploy_root) / category
        self.deploy_dir = str(deploy_dir)
        cfg_path = deploy_dir / "classify_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"缺少分类器配置: {cfg_path}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.classes: list[str] = list(self.cfg["classes"])
        self.class_zh: dict[str, str] = dict(self.cfg.get("class_zh", {}))
        self.size = int(self.cfg["image_size"])
        self.amap_mean = float(self.cfg.get("amap_mean", 0.0))
        self.amap_std = float(self.cfg.get("amap_std", 1.0))

        self.sessions = []
        self.input_names = []
        for fn in self.cfg["files"]:
            p = deploy_dir / fn
            if not p.exists():
                raise FileNotFoundError(f"缺少分类器模型: {p}")
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            self.sessions.append(sess)
            self.input_names.append(sess.get_inputs()[0].name)
        self.n_models = len(self.sessions)

    # ---- 预处理 ----
    def preprocess(self, img_bgr: np.ndarray, amap: np.ndarray) -> np.ndarray:
        """BGR 原图 + 异常图 -> (1,4,H,W) float32 NCHW。

        与训练端 (src/train_classifier.py) 的 val 分支严格一致：
        PIL BILINEAR resize -> /255 -> ImageNet 归一化；异常图 cv2.resize -> (x-mean)/std。
        """
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(pil, dtype=np.float32) / 255.0          # (H,W,3)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))                           # (3,H,W)

        m = cv2.resize(amap.astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        m = (m - self.amap_mean) / max(self.amap_std, 1e-6)
        x = np.concatenate([x, m[None]], axis=0)                 # (4,H,W)

        return np.ascontiguousarray(x[None], dtype=np.float32)

    # ---- 推理 ----
    def predict(self, img_bgr: np.ndarray, amap: np.ndarray) -> dict:
        """返回各类别概率、top-1 类型与置信度。"""
        x = self.preprocess(img_bgr, amap)
        probs = np.zeros(len(self.classes), dtype=np.float64)
        for sess, name in zip(self.sessions, self.input_names):
            logits = sess.run(None, {name: x})[0][0].astype(np.float64)
            logits -= logits.max()
            e = np.exp(logits)
            probs += e / e.sum()
        probs /= max(self.n_models, 1)

        top = int(np.argmax(probs))
        return {
            "probs": {c: float(probs[i]) for i, c in enumerate(self.classes)},
            "top": self.classes[top],
            "conf": float(probs[top]),
            "top3": [
                {"cls": self.classes[i], "zh": self.class_zh.get(self.classes[i], ""), "prob": float(probs[i])}
                for i in np.argsort(-probs)[:3]
            ],
        }

    def predict_defect_type(self, img_bgr: np.ndarray, amap: np.ndarray) -> dict:
        """两阶段用：只在“已判定为缺陷”后调用，且只在 7 个缺陷类里选。

        返回 {"type","zh","conf","source":"model","probs","top3"}
        """
        out = self.predict(img_bgr, amap)
        probs = out["probs"]
        defect_probs = {c: p for c, p in probs.items() if c != "good"}
        total = sum(defect_probs.values())
        if total <= 1e-9:
            # 模型认为几乎全是 good，与 PatchCore 的缺陷判定冲突，退回 top-1
            top = out["top"]
        else:
            top = max(defect_probs, key=defect_probs.get)
        conf = defect_probs.get(top, 0.0)
        return {
            "type": top,
            "zh": self.class_zh.get(top, ""),
            "conf": round(float(conf), 4),
            "source": "model",
            "probs": {c: round(float(p), 4) for c, p in probs.items()},
            "top3": out["top3"],
        }
