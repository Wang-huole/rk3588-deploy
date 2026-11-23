"""把 deploy/<category>/backbone.onnx 导出为 TensorRT FP16 engine。

用法（训练/部署环境执行，需安装 tensorrt）:
    python -m src.export_tensorrt --category toothbrush
    python -m src.export_tensorrt                       # 导出 config 里全部类别

产物:
    deploy/<category>/backbone_fp16.engine   TensorRT 序列化引擎（设备/GPU 相关，不可跨机器复用）

deploy_infer.DeployPatchCore 会自动检测该文件，存在则优先用 TRT 推理，否则回退 ONNX Runtime。

注意:
    - engine 与 GPU 架构强绑定（RTX 5060 = Blackwell sm_120），换卡需重新导出。
    - 需要 tensorrt 包: pip install tensorrt（或从 NVIDIA 官网下载对应 CUDA 版本）。
"""
from __future__ import annotations

import argparse
import os
import sys

from .utils import load_config


def export_category(cfg: dict, category: str, workspace_gb: float = 2.0,
                    fp16: bool = True) -> str:
    """将 deploy/<category>/backbone.onnx 编译为 TensorRT engine。"""
    import tensorrt as trt

    deploy_dir = os.path.join("deploy", category)
    onnx_path = os.path.join(deploy_dir, "backbone.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"未找到 {onnx_path}，请先运行 python -m src.export_onnx --category {category}"
        )

    trt_v = tuple(int(x) for x in trt.__version__.split(".")[:2])
    print(f"  TensorRT 版本: {trt.__version__}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [ONNX 解析错误] {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  启用 FP16")
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
    )

    print("  正在编译 engine（首次较慢，请耐心等待）...")
    # TRT 10+ 用 build_serialized_network，旧版用 build_engine + serialize
    if trt_v >= (10, 0):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("build_serialized_network 返回 None")
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("build_engine 返回 None")
        serialized = engine.serialize()

    engine_path = os.path.join(deploy_dir, "backbone_fp16.engine")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine 已保存: {engine_path}  ({size_mb:.1f} MB)")
    return engine_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--workspace_gb", type=float, default=2.0,
                        help="TRT workspace 显存上限（GB）")
    parser.add_argument("--no_fp16", action="store_true", help="禁用 FP16（用 FP32）")
    args = parser.parse_args()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        print("错误: 未安装 tensorrt。请先 pip install tensorrt（需匹配 CUDA 版本）。")
        sys.exit(1)

    cfg = load_config(args.config)
    categories = [args.category] if args.category else cfg["dataset"]["categories"]
    for cat in categories:
        print(f"\n=== 导出 TensorRT: {cat} ===")
        export_category(cfg, cat, workspace_gb=args.workspace_gb, fp16=not args.no_fp16)


if __name__ == "__main__":
    main()
