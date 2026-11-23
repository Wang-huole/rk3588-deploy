"""ONNX -> RKNN (RK3588) 转换 + INT8 量化。

在装有 RKNN-Toolkit2（x86 Linux 或 ARM64 Linux）的转换机上执行，
不要直接在 Windows 上跑（Toolkit2 不提供 Windows wheel）。

前置:
    pip install rknn-toolkit2        # 版本需与板端 librknnrt 匹配（建议 2.3.x / 2.4.x）
    确认 python -c "from rknn.api.rknn import RKNN" 能导入

用法:
    # 转换单个类别 backbone（PatchCore 特征提取器）
    python rknn_deploy/onnx2rknn.py --category toothbrush
    python rknn_deploy/onnx2rknn.py --category pill

    # 关闭量化（只出 FP16 RKNN，用于精度对照/算子兜底）
    python rknn_deploy/onnx2rknn.py --category pill --no-quant

产物:
    rknn_deploy/rknn/<category>/backbone_int8.rknn   （量化加速）
    rknn_deploy/rknn/<category>/backbone_fp16.rknn   （--no-quant 时）

关键说明:
    1. backbone.onnx 输入为 RGB [0,255] float32，图内已含 /255 与 ImageNet 归一化，
       所以此处 mean_value/std_value 全部填 0，避免重复归一化。
    2. 固定 batch=1、固定 448×448 输入，动态轴会给 NPU 转换带来额外不确定性。
    3. 若某层不支持 NPU，rknn.build 会回退 CPU；看转换日志的 "unsupported/CPU layer"
       统计，再决定是否换 resnet50 或保留 ONNX Runtime 兜底。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from rknn.api.rknn import RKNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_ROOT = PROJECT_ROOT / "deploy"
RKNN_OUT = PROJECT_ROOT / "rknn_deploy" / "rknn"
CALIB_DIR = PROJECT_ROOT / "rknn_deploy" / "calib"

TARGET = "rk3588"


def convert(category: str, quant: bool, target_platform: str) -> Path:
    onnx_path = DEPLOY_ROOT / category / "backbone.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"缺少 ONNX: {onnx_path}")

    out_dir = RKNN_OUT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "int8" if quant else "fp16"
    rknn_path = out_dir / f"backbone_{tag}.rknn"

    calib_list = CALIB_DIR / f"{category}_calib.txt"
    if quant and not calib_list.exists():
        raise FileNotFoundError(f"缺少校准清单: {calib_list}，先运行 prepare_calib.py")

    # mean/std 填 0：归一化在 ONNX 图内完成
    rknn = RKNN(verbose=True)
    config = {
        "target_platform": target_platform,
        "mean_values": [[0, 0, 0]],
        "std_values": [[1, 1, 1]],
        "quantized_dtype": "asymmetric_quantized-8",
        "quantized_method": "channel",
        # 特征图输出对量化敏感，强制保留输出层精度，避免异常分数整体偏移
        "quantized_hybrid": True,
    }
    if not quant:
        config["quantized_dtype"] = "w16a16"

    ret = rknn.config(**config)
    if ret != 0:
        raise RuntimeError(f"rknn.config 失败: {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx 失败: {ret}")

    ret = rknn.build(do_quantization=quant, dataset=str(calib_list) if quant else None)
    if ret != 0:
        raise RuntimeError(f"rknn.build 失败（量化/编译）: {ret}")

    ret = rknn.export_rknn_model(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"导出 RKNN 失败: {ret}")

    size_mb = rknn_path.stat().st_size / 1e6
    print(f"[OK] {category} -> {rknn_path} ({size_mb:.1f} MB, {tag})")
    rknn.release()
    return rknn_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=["toothbrush", "pill"])
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--no-quant", action="store_true", help="不做 INT8 量化，导出 FP16")
    args = ap.parse_args()
    convert(args.category, quant=not args.no_quant, target_platform=args.target)


if __name__ == "__main__":
    main()
