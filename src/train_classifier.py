"""缺陷类型分类器训练（迁移学习微调，方案A）。

与 PatchCore 的关系（两阶段串联）：
    阶段1  PatchCore 判定 "是不是缺陷"（异常分数 + 标定阈值，很准）
    阶段2  本分类器只在判为缺陷时给出 "是哪种缺陷"

输入为 4 通道：RGB(3) + PatchCore 异常热力图(1)。
异常图由部署包里的 ONNX backbone 推理得到，与线上完全一致，不额外增加推理成本。

数据（MVTec AD / pill）：
    good            -> train/good (267 张)          ← 故意排除 test/good（阈值标定集）
    color/combined/contamination/crack/faulty_imprint/pill_type/scratch -> test/<类>/*
输出：
    deploy/<category>/classify_fold{0..4}.onnx   5 折模型（推理时概率平均）
    deploy/<category>/classify_config.json        类别表 / 图像尺寸 / 归一化参数

用法：
    python -m src.train_classifier --category pill
    python -m src.train_classifier --category pill --folds 5 --epochs 18
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from deploy_infer import DeployPatchCore, imread_unicode

# 类别顺序即标签索引，必须与 classify_config.json 保持一致
CLASSES = ["good", "color", "combined", "contamination", "crack",
           "faulty_imprint", "pill_type", "scratch"]
CLASS_ZH = {
    "good": "正常", "color": "颜色异常", "combined": "复合缺陷", "contamination": "污染",
    "crack": "裂纹", "faulty_imprint": "刻印错误", "pill_type": "药品类型错误", "scratch": "划痕",
}

DATASET_ROOT = Path("C:/Users/1/Desktop/素材/mvtec_anomaly_detection")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "outputs" / "classifier_cache"


# ---------------------------------------------------------------------------
# 数据收集
# ---------------------------------------------------------------------------
def collect_samples(category: str) -> tuple[list[str], list[int]]:
    """收集带标签样本路径。

    good 类只用 train/good；test/good 是阈值标定集，排除以免污染。
    其余 7 类缺陷样本 MVTec 只在 test/ 下提供，直接使用。
    """
    root = DATASET_ROOT / category
    paths: list[str] = []
    labels: list[int] = []
    for idx, cls in enumerate(CLASSES):
        if cls == "good":
            files = sorted((root / "train" / "good").glob("*.png"))
        else:
            files = sorted((root / "test" / cls).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"{category}/{cls} 没有找到图片（路径 {root}）")
        paths += [str(p) for p in files]
        labels += [idx] * len(files)
        print(f"  {cls:16s} {len(files):4d} 张")
    return paths, labels


# ---------------------------------------------------------------------------
# 异常热力图缓存
# ---------------------------------------------------------------------------
def _norm_key(p: str) -> str:
    """路径键归一化：Windows 反斜杠 -> 正斜杠。

    np.savez 用 zip 存储时会把键名里的反斜杠规范化成正斜杠，
    所以存与取必须用同一种写法，否则 KeyError。
    """
    return str(p).replace("\\", "/")


def cache_anomaly_maps(category: str, paths: list[str]) -> dict[str, np.ndarray]:
    """用部署包的 ONNX backbone 预计算异常图（56×56），与线上推理完全一致。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{category}_amap.npz"
    if cache_file.exists():
        print(f"  复用缓存 {cache_file}")
        with np.load(cache_file, allow_pickle=True) as z:
            return {_norm_key(k): z[k] for k in z.files}

    model = DeployPatchCore(category, deploy_root=str(PROJECT_ROOT / "deploy"))
    maps: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = imread_unicode(p)
        if img is None:
            raise ValueError(f"无法读取图片: {p}")
        _, amap = model.predict(img)
        maps[_norm_key(p)] = amap.astype(np.float16)
        if i % 50 == 0 or i == len(paths):
            print(f"  异常图 {i}/{len(paths)}  ({time.time() - t0:.0f}s)")
    np.savez_compressed(cache_file, **maps)
    print(f"  缓存已保存 {cache_file}（{time.time() - t0:.0f}s）")
    return maps


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def preload_images(paths: list[str], img_size: int) -> dict[str, np.ndarray]:
    """把图片一次性解码并缩放到 img_size，缓存成 uint8 数组。

    好处：训练时不再反复读 1000×1000 PNG；也避免 Windows 下 DataLoader
    多进程 worker 各自加载 torch/CUDA DLL 触发 "页面文件太小"(WinError 1455)。
    """
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(paths, 1):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            out[p] = np.asarray(im, dtype=np.uint8)
        if i % 100 == 0 or i == len(paths):
            print(f"  预载图片 {i}/{len(paths)}")
    return out


class PillDataset(Dataset):
    def __init__(self, paths, labels, maps, img_cache, img_size, train: bool,
                 amap_mean: float = 0.0, amap_std: float = 1.0):
        self.paths = paths
        self.labels = labels
        self.maps = maps
        self.img_cache = img_cache
        self.train = train
        self.amap_mean, self.amap_std = amap_mean, amap_std
        # 注意：Resize 已在预载阶段完成，这里只做增广与归一化
        if train:
            self.tf = transforms.Compose([
                transforms.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def _load_amap(self, path: str, size: int) -> torch.Tensor:
        m = self.maps[_norm_key(path)].astype(np.float32)
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
        m = (m - self.amap_mean) / max(self.amap_std, 1e-6)
        return torch.from_numpy(m)[None]  # (1,H,W)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = self.tf(Image.fromarray(self.img_cache[path]))
        amap = self._load_amap(path, img.shape[-1])
        x = torch.cat([img, amap], dim=0)  # (4,H,W)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(num_classes: int, in_ch: int = 4) -> nn.Module:
    """ImageNet 预训练 ResNet18，首层改为 4 通道（第 4 通道零初始化）。"""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    old = m.conv1
    new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = 0.0
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
        all_pred += out.argmax(1).cpu().tolist()
        all_true += y.cpu().tolist()
    return total_loss / max(seen, 1), correct / max(seen, 1), all_pred, all_true


def make_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """类别不均衡加权（good 267 张 vs pill_type 9 张）。"""
    cnt = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = cnt.sum() / (num_classes * np.maximum(cnt, 1))
    print("  类别权重: " + ", ".join(f"{c}={w[i]:.2f}" for i, c in enumerate(CLASSES)))
    return torch.tensor(w, dtype=torch.float32)


def train_one(model, train_ds, val_ds, epochs, lr, batch_size, device, num_workers, class_w):
    # num_workers 固定为 0：Windows 下多进程 worker 需各自加载 torch/CUDA DLL，
    # 会触发 "页面文件太小"(WinError 1455)；数据已预载进内存，单进程足够快。
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss(weight=class_w.to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, best_state, best_ep = 0.0, None, -1
    for ep in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        with torch.no_grad():
            va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        scheduler.step()
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    ep{ep + 1:02d}/{epochs}  train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} "
              f"val_loss={va_loss:.3f} val_acc={va_acc:.3f}  ({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    return model, best_acc, best_ep


def export_onnx(model, path: Path, img_size: int):
    model = model.cpu().eval()   # 导出必须在 CPU 上做（模型可能还挂在 GPU）
    dummy = torch.randn(1, 4, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18, do_constant_folding=True,
    )
    data_file = path.with_suffix(".onnx.data")
    extra = f" + {data_file.name}" if data_file.exists() else ""
    print(f"  已导出 {path}  ({path.stat().st_size / 1e6:.1f} MB{extra})")


def verify_onnx(onnx_path: Path, model, img_size: int, device):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = model.cpu().eval()
    x = torch.randn(1, 4, img_size, img_size)
    with torch.no_grad():
        ref = model(x).cpu().numpy()
    got = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"  ONNX 校验: max|Δlogits| = {diff:.2e}  {'OK' if diff < 1e-3 else '警告：数值不一致'}")
    return diff


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="pill")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  图像尺寸: {args.img_size}  折数: {args.folds}  轮数: {args.epochs}")

    print("\n[1/6] 收集样本")
    paths, labels = collect_samples(args.category)
    print(f"  合计 {len(paths)} 张")

    print("\n[2/6] 缓存异常热力图")
    maps = cache_anomaly_maps(args.category, paths)
    amap_mean = float(np.mean([v.astype(np.float32).mean() for v in maps.values()]))
    amap_std = float(np.mean([v.astype(np.float32).std() for v in maps.values()]))
    print(f"  异常图归一化参数 mean={amap_mean:.4f} std={amap_std:.4f}")

    print("\n[3/5] 预载图片到内存")
    img_cache = preload_images(paths, args.img_size)

    class_w = make_weights(labels, len(CLASSES))

    # 划分（分层 K 折）
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(paths)), labels))
    print(f"\n[4/5] {args.folds} 折交叉验证")
    fold_accs = []
    deploy_dir = PROJECT_ROOT / "deploy" / args.category
    deploy_dir.mkdir(parents=True, exist_ok=True)

    for k, (tr_idx, va_idx) in enumerate(folds):
        print(f"  --- fold {k + 1}/{args.folds} ---")
        tr_ds = PillDataset([paths[i] for i in tr_idx], [labels[i] for i in tr_idx], maps,
                            img_cache, args.img_size, True, amap_mean, amap_std)
        va_ds = PillDataset([paths[i] for i in va_idx], [labels[i] for i in va_idx], maps,
                            img_cache, args.img_size, False, amap_mean, amap_std)
        model = build_model(len(CLASSES)).to(device)
        model, best_acc, best_ep = train_one(model, tr_ds, va_ds, args.epochs, args.lr,
                                              args.batch_size, device, args.workers, class_w)
        print(f"  fold {k + 1} 最佳 val_acc={best_acc:.4f} (ep{best_ep + 1})")
        fold_accs.append(best_acc)

        # 逐类混淆统计
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, num_workers=0)
        _, _, pred, true = run_epoch(model, va_loader, nn.CrossEntropyLoss(), None, None, device, False)
        for ci, c in enumerate(CLASSES):
            idx = [i for i, t in enumerate(true) if t == ci]
            if idx:
                acc = sum(1 for i in idx if pred[i] == ci) / len(idx)
                print(f"      {c:16s} {acc:.2f}  ({len(idx)} 张)")

        onnx_path = deploy_dir / f"classify_fold{k}.onnx"
        export_onnx(model, onnx_path, args.img_size)
        verify_onnx(onnx_path, model, args.img_size, device)

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"\n  {args.folds} 折交叉验证: 平均准确率 {mean_acc:.4f} ± {std_acc:.4f}  (各折 {['%.3f' % a for a in fold_accs]})")

    print("\n[5/5] 写配置文件")
    cfg = {
        "category": args.category,
        "classes": CLASSES,
        "class_zh": CLASS_ZH,
        "image_size": args.img_size,
        "in_channels": 4,
        "amap_mean": amap_mean,
        "amap_std": amap_std,
        "files": [f"classify_fold{k}.onnx" for k in range(args.folds)],
        "cv_accuracy_mean": round(mean_acc, 4),
        "cv_accuracy_std": round(std_acc, 4),
        "cv_fold_accuracies": [round(a, 4) for a in fold_accs],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("输入为 4 通道：RGB(3, ImageNet 归一化) + PatchCore 异常图(1)。"
                 "两阶段串联：PatchCore 先判是否缺陷，判为缺陷时才用本模型细分类型。"),
    }
    cfg_path = deploy_dir / "classify_config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已写入 {cfg_path}")

    print("\n[6/6] 完成")


if __name__ == "__main__":
    main()
