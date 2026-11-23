"""转换前核查 ONNX：输入/输出形状、算子、外部权重，定位 RKNN 可能不支持的层。

用法（PC 训练环境）:
    python rknn_deploy/check_onnx.py
"""
from __future__ import annotations

import os
from pathlib import Path

import onnx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY = PROJECT_ROOT / "deploy"

SUPPORTED = {
    # RKNN 常见支持算子（非穷举，仅用于粗筛）
    "Conv", "Add", "Sub", "Div", "Mul", "Gemm", "Relu", "MaxPool",
    "AveragePool", "ReduceMean", "Reshape", "Flatten", "Concat",
    "Sigmoid", "Softmax", "Transpose", "Resize", "Clip", "Identity",
}


def shape_str(tensor, graph):
    try:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in tensor.type.tensor_type.shape.dim]
        return str(dims)
    except Exception:
        return "?"


def inspect(onnx_path: Path):
    print("=" * 70)
    print(onnx_path.relative_to(PROJECT_ROOT))
    m = onnx.load(str(onnx_path))
    for i in m.graph.input:
        print(f"  input : {i.name}  {shape_str(i, m.graph)}")
    for o in m.graph.output:
        print(f"  output: {o.name}  {shape_str(o, m.graph)}")
    ops = sorted({n.op_type for n in m.graph.node})
    unsupported = [o for o in ops if o not in SUPPORTED]
    print(f"  ops({len(ops)}): {ops}")
    if unsupported:
        print(f"  [注意] 可能需 CPU 回退的算子: {unsupported}")
    ext = [init.name for init in m.graph.initializer
           if init.HasField("data_location") and init.data_location == 1]
    if ext:
        print(f"  [注意] 外部权重文件 {len(ext)} 个（.onnx.data），转换时需与 .onnx 同目录")
    print()


def main():
    for onnx_path in sorted(DEPLOY.rglob("*.onnx")):
        inspect(onnx_path)


if __name__ == "__main__":
    main()
