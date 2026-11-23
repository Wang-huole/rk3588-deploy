"""PatchCore 核心算法：Memory Bank 构建 + Coreset 稀疏采样 + 最近邻异常分数。"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def _try_import_faiss():
    """尝试导入 faiss（可选依赖），失败返回 None。"""
    try:
        import faiss
        return faiss
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 特征处理
# ---------------------------------------------------------------------------
def _stack_patch_features(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """把多层 feature map 对齐到同一空间尺寸后 concat，再 flatten 成 patch 特征。

    Args:
        feats: {层名: (B, C, H, W)}

    Returns:
        (B, H*W, C_concat) 的 patch 特征
    """
    # 以最大的 H/W 为目标（通常是 layer2），把其他层上采样到该尺寸
    target_h = max(f.shape[2] for f in feats.values())
    target_w = max(f.shape[3] for f in feats.values())

    aligned = []
    for f in feats.values():
        if f.shape[2] != target_h or f.shape[3] != target_w:
            f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
        aligned.append(f)

    # (B, C1+C2, H, W)
    cat = torch.cat(aligned, dim=1)
    B, C, H, W = cat.shape
    # (B, H*W, C)
    patches = cat.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return patches, (H, W)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-10)


# ---------------------------------------------------------------------------
# Coreset 稀疏采样（greedy k-center）
# ---------------------------------------------------------------------------
def coreset_sampling(embeddings: np.ndarray, ratio: float, replace: bool = False) -> np.ndarray:
    """Greedy k-center coreset 采样（分块计算，避免大数组爆内存）。

    反复选择"距离已选集合最远"的点，直到选到目标数量。
    """
    n = embeddings.shape[0]
    m = max(1, int(n * ratio))
    if m >= n:
        return np.arange(n)

    rng = np.random.default_rng(0)
    first = int(rng.integers(n))
    selected = [first]

    emb = embeddings.astype(np.float32)
    # 分块计算每个点到 first 的距离，避免一次性分配 (N, D) 的减法结果
    min_dist = _blocked_l2_dist(emb, emb[first])

    for _ in tqdm(range(1, m), desc="Coreset 采样", leave=False):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = _blocked_l2_dist(emb, emb[idx])
        min_dist = np.minimum(min_dist, new_dist)

    return np.array(selected, dtype=np.int64)


def _blocked_l2_dist(emb: np.ndarray, ref: np.ndarray, block: int = 50000) -> np.ndarray:
    """分块计算 emb 中每个点到 ref 的 L2 距离，避免一次性分配大数组。"""
    n = emb.shape[0]
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        chunk = emb[s:e]
        # ||x - ref||^2 = ||x||^2 + ||ref||^2 - 2*x·ref
        d = np.sum(chunk * chunk, axis=1) + np.dot(ref, ref) - 2.0 * chunk @ ref
        np.maximum(d, 0, out=d)
        out[s:e] = np.sqrt(d)
    return out


# ---------------------------------------------------------------------------
# PatchCore 模型
# ---------------------------------------------------------------------------
class PatchCore:
    """PatchCore 推理器。

    使用方式：
        model = PatchCore(...)
        model.build_memory_bank(loader, extractor, device)   # 训练（建 bank）
        score, anomaly_map = model.predict(img_feats)         # 推理
        model.save("memory.pt") / model.load("memory.pt")
    """

    def __init__(
        self,
        coreset_ratio: float = 0.1,
        n_neighbors: int = 3,
        anomaly_dir: str = "avg",
        device: str = "cuda",
    ) -> None:
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.anomaly_dir = anomaly_dir
        self.device = device
        self.memory_bank: np.ndarray | None = None
        self.nn: NearestNeighbors | None = None
        self.feature_shape: Tuple[int, int] | None = None  # (H, W) 用于上采样
        # Faiss 可选后端（比 sklearn 快）；未安装则回退 sklearn
        self.faiss = _try_import_faiss()
        self._faiss_index = None

    # ---- 训练 ----
    @torch.no_grad()
    def build_memory_bank(self, loader, extractor) -> None:
        """遍历正常样本，收集 patch 特征并做 coreset 采样。"""
        extractor = extractor.to(self.device).eval()
        all_patches = []

        for batch in tqdm(loader, desc="构建 Memory Bank"):
            imgs = batch["image"].to(self.device)
            feats = extractor(imgs)
            patches, shape = _stack_patch_features(feats)  # (B, HW, C)
            patches = _l2_normalize(patches)
            all_patches.append(patches.cpu().numpy())
            if self.feature_shape is None:
                self.feature_shape = shape  # (H, W)，来自实际特征图尺寸（不再假设正方形）

        # (N_total, C)
        bank = np.concatenate(all_patches, axis=0).reshape(-1, all_patches[0].shape[-1])
        print(f"  原始 patch 数: {bank.shape[0]:,}")

        # 如果 patch 总数太大，先随机预采样到上限，避免 coreset 过慢/爆内存
        max_pre_sample = 200000
        if bank.shape[0] > max_pre_sample:
            rng = np.random.default_rng(42)
            sel = rng.choice(bank.shape[0], max_pre_sample, replace=False)
            bank = bank[sel]
            print(f"  预采样到: {bank.shape[0]:,}")

        idx = coreset_sampling(bank, self.coreset_ratio)
        self.memory_bank = bank[idx]
        print(f"  Coreset 后 patch 数: {self.memory_bank.shape[0]:,}")

        # 建立最近邻索引（优先 Faiss，回退 sklearn）
        self._build_index()
        print(f"  最近邻后端: {'Faiss' if self._faiss_index is not None else 'sklearn'}")

    def _build_index(self) -> None:
        """根据 memory bank 构建最近邻索引（Faiss 优先）。"""
        bank = np.ascontiguousarray(self.memory_bank.astype(np.float32))
        if self.faiss is not None:
            dim = bank.shape[1]
            self._faiss_index = self.faiss.IndexFlatL2(dim)
            self._faiss_index.add(bank)
            self.nn = None
        else:
            self._faiss_index = None
            self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, n_jobs=-1)
            self.nn.fit(bank)

    # ---- 推理 ----
    @torch.no_grad()
    def predict(self, feats: Dict[str, torch.Tensor]) -> Tuple[float, np.ndarray]:
        """对单批次 feature map 计算异常分数和像素图。

        Args:
            feats: extractor 输出的 {层名: (B, C, H, W)}，B=1

        Returns:
            image_score: 图像级异常分数
            anomaly_map: (H_orig, W_orig) 之前的 (h, w) 距离图（numpy）
        """
        patches, _ = _stack_patch_features(feats)  # (1, HW, C)
        patches = _l2_normalize(patches).cpu().numpy().reshape(-1, patches.shape[-1])

        # 最近邻距离（Faiss 优先）
        q = np.ascontiguousarray(patches.astype(np.float32))
        if self._faiss_index is not None:
            dists, _ = self._faiss_index.search(q, self.n_neighbors)  # (HW, k)
            # Faiss 返回的是 L2 距离的平方，开方后与 sklearn 一致
            dists = np.sqrt(np.maximum(dists, 0.0))
        else:
            dists, _ = self.nn.kneighbors(q)  # (HW, n_neighbors)
        # PatchCore: 用最近邻的平均距离
        patch_scores = dists.mean(axis=1)  # (HW,)

        # 像素级距离图
        h, w = self.feature_shape
        anomaly_map = patch_scores.reshape(h, w)

        # 图像级分数
        if self.anomaly_dir == "max":
            image_score = float(patch_scores.max())
        else:  # avg: 取 top-k patch 平均（PatchCore 原论文）
            k = max(1, int(0.01 * len(patch_scores)))
            topk = np.sort(patch_scores)[-k:]
            image_score = float(topk.mean())

        return image_score, anomaly_map

    # ---- 序列化 ----
    def save(self, path: str) -> None:
        torch.save(
            {
                "memory_bank": self.memory_bank,
                "feature_shape": self.feature_shape,
                "coreset_ratio": self.coreset_ratio,
                "n_neighbors": self.n_neighbors,
                "anomaly_dir": self.anomaly_dir,
            },
            path,
        )

    def load(self, path: str) -> None:
        # weights_only=False: memory bank 里存的是 numpy 数组，PyTorch 2.6+ 默认
        # weights_only=True 会拒绝加载 numpy 对象；文件是本项目自己生成的，安全。
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.memory_bank = ckpt["memory_bank"]
        self.feature_shape = ckpt["feature_shape"]
        self.coreset_ratio = ckpt.get("coreset_ratio", 0.1)
        self.n_neighbors = ckpt.get("n_neighbors", 3)
        self.anomaly_dir = ckpt.get("anomaly_dir", "avg")
        self._build_index()
