"""生成 RKNN INT8 量化校准清单（PatchCore backbone 用正常样本）。

用法（在训练/PC 环境执行）:
    python rknn_deploy/prepare_calib.py

输出:
    rknn_deploy/calib/<category>_calib.txt   每行一张正常图的绝对路径
    rknn_deploy/calib/<category>/            可选：把校准图复制到该目录，便于整包拷到转换机

说明:
    - 校准集应尽量覆盖产线真实光照/角度/背景；默认取 MVTec train/good，
      上线前建议换成相机实拍正常样本。
    - backbone 输入是 RGB [0,255]，归一化已固化在 ONNX 图内；RKNN 转换时
      mean/std 填 0，由图内 Sub/Div 完成归一化（见 onnx2rknn.py）。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path("C:/Users/1/Desktop/素材/mvtec_anomaly_detection")


def collect(category: str, data_root: Path, copy: bool) -> list[Path]:
    # 优先 train/good（训练分布），补充 test/good 扩覆盖；只取正常样本
    picked: list[Path] = []
    for sub in ("train/good", "test/good"):
        d = data_root / category / sub
        if d.exists():
            picked += sorted([p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not picked:
        raise FileNotFoundError(f"未找到 {category} 正常样本: {data_root / category}")
    if copy:
        out_dir = PROJECT_ROOT / "rknn_deploy" / "calib" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in picked:
            shutil.copy2(p, out_dir / p.name)
        picked = [out_dir / p.name for p in picked]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--categories", nargs="+", default=["toothbrush", "pill"])
    ap.add_argument("--copy", action="store_true", help="把校准图复制到 rknn_deploy/calib/<category>/")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    calib_dir = PROJECT_ROOT / "rknn_deploy" / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    for cat in args.categories:
        imgs = collect(cat, data_root, args.copy)
        list_file = calib_dir / f"{cat}_calib.txt"
        list_file.write_text("\n".join(str(p) for p in imgs) + "\n", encoding="utf-8")
        print(f"{cat}: {len(imgs)} 张 -> {list_file}")


if __name__ == "__main__":
    main()
