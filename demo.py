"""缺陷检测演示脚本：批量推理 + 生成 HTML 结果报告页。

在训练/演示环境运行（依赖 onnxruntime + numpy + opencv，不需要 PyTorch）。
先确保部署包已导出（python -m src.export_onnx --category toothbrush）。

用法:
    # 演示测试集全部图片，生成可分享的 HTML 报告
    python demo.py --category toothbrush

    # 指定输入文件夹 + 输出目录
    python demo.py --category pill --input <文件夹> --out outputs/demo

产物:
    outputs/demo/<category>/result.html   单文件报告（图片 base64 内嵌，可直接发给别人看）
    outputs/demo/<category>/vis/          每张图的热力图叠加
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from deploy_infer import DeployPatchCore, imread_unicode, infer_single


def img_to_base64(img: np.ndarray) -> str:
    """numpy BGR 图 -> base64 PNG。"""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_html(results: list[dict], category: str, threshold: float) -> str:
    """把批量结果渲染成单个 HTML 文件（图片内嵌）。"""
    cards = []
    # 按分数从高到低排序
    results = sorted(results, key=lambda r: r["score"], reverse=True)
    for r in results:
        # 用 imread_unicode 读，避免中文/特殊字符路径下 cv2.imread 静默返回 None
        vis = imread_unicode(r["vis_path"])
        if vis is None:
            print(f"  [跳过] 无法读取可视化图: {r['vis_path']}")
            continue
        b64 = img_to_base64(vis)
        label_cls = "defect" if r["label"] == "缺陷" else "good"
        cards.append(f"""
        <div class="card {label_cls}">
          <img src="data:image/png;base64,{b64}" />
          <div class="info">
            <span class="badge {label_cls}">{r['label']}</span>
            <span class="score">score: {r['score']:.4f}</span>
            <span class="ms">{r['ms']:.0f} ms</span>
            <div class="name">{Path(r['path']).name}</div>
          </div>
        </div>""")

    n_good = sum(1 for r in results if r["label"] == "正常")
    n_bad = len(results) - n_good
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>PatchCore 缺陷检测演示 - {category}</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; margin: 24px; background: #f5f6fa; }}
  h1 {{ font-size: 22px; }}
  .summary {{ background: #fff; padding: 14px 18px; border-radius: 8px; margin-bottom: 18px;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .summary b {{ color: #c0392b; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
  .card.defect {{ outline: 3px solid #e74c3c; }}
  .card.good {{ outline: 3px solid #27ae60; }}
  .card img {{ width: 100%; display: block; }}
  .info {{ padding: 8px 12px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px; color: #fff; font-size: 12px; }}
  .badge.defect {{ background: #e74c3c; }}
  .badge.good {{ background: #27ae60; }}
  .score {{ margin-left: 8px; font-weight: 600; }}
  .ms {{ color: #888; font-size: 12px; }}
  .name {{ color: #666; font-size: 12px; margin-top: 4px; word-break: break-all; }}
</style>
</head>
<body>
  <h1>PatchCore 缺陷检测演示 — {category}</h1>
  <div class="summary">
    共 <b>{len(results)}</b> 张 | 正常 <b style="color:#27ae60">{n_good}</b> 张 | 缺陷 <b>{n_bad}</b> 张
    | 阈值 threshold = {threshold} | 红色边框=判为缺陷，绿色=正常
  </div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>"""


def calibrate_threshold(model: DeployPatchCore, category: str, n: int = 30,
                        percentile: float = 99.0,
                        calibrate_dir: str | None = None) -> float:
    """用正常样本标定判定阈值，并写入部署包 config.json。

    阈值 = 正常样本分数的 P99（默认），即正常样本只有 1% 会被误判为缺陷。
    标定目录优先取 --calibrate_dir（产线现场采集的正常样本）；否则用训练集 good。

    Returns:
        threshold: 标定出的阈值
    """
    if calibrate_dir:
        good_dir = Path(calibrate_dir)
    else:
        import yaml
        with open("configs/patchcore_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        good_dir = Path(cfg["dataset"]["root"]) / category / "train" / "good"
    if not good_dir.exists():
        raise FileNotFoundError(f"标定目录不存在: {good_dir}")

    files = sorted(good_dir.glob("*.png"))[:n]
    if not files:
        raise FileNotFoundError(f"标定目录无正常样本: {good_dir}")

    scores = []
    for p in files:
        try:
            r = infer_single(model, str(p), None, threshold=0.0, save_vis=False)
            scores.append(r["score"])
        except Exception as e:
            print(f"  [跳过] {p.name}: {str(e)[:60]}")
    if not scores:
        raise RuntimeError("标定失败：没有可用正常样本")

    scores = np.array(scores)
    threshold = float(np.percentile(scores, percentile))

    # 写入部署包 config.json，部署推理默认使用
    cfg_path = Path(model.deploy_dir) / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    cfg_json["threshold"] = threshold
    cfg_json["threshold_source"] = str(good_dir)
    cfg_json["threshold_percentile"] = percentile
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2, ensure_ascii=False)

    print(f"\n标定结果（{len(scores)} 张正常样本, 目录 {good_dir}）:")
    print(f"  分数 min={scores.min():.4f}  mean={scores.mean():.4f}  max={scores.max():.4f}")
    print(f"  建议阈值（P{percentile:.0f}）= {threshold:.4f}（已写入 {cfg_path}）")
    return threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--input", default=None, help="输入文件夹；默认用 MVTec 测试集对应类别")
    parser.add_argument("--out", default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值；不填时用 --calibrate 标定，或读 deploy config 里的阈值")
    parser.add_argument("--calibrate", action="store_true",
                        help="先用正常样本标定阈值（默认 P99）")
    parser.add_argument("--calibrate_n", type=int, default=30, help="标定使用的正常样本数")
    parser.add_argument("--calibrate_percentile", type=float, default=99.0)
    parser.add_argument("--calibrate_dir", default=None,
                        help="标定用正常样本目录（产线现场采集）；默认用训练集 good")
    parser.add_argument("--deploy_root", default="deploy")
    args = parser.parse_args()

    model = DeployPatchCore(args.category, deploy_root=args.deploy_root)

    # 阈值：显式 > 标定 > 部署包内已存阈值 > 默认 0.5
    threshold = args.threshold
    if threshold is None:
        if args.calibrate:
            threshold = calibrate_threshold(model, args.category, args.calibrate_n,
                                            args.calibrate_percentile, args.calibrate_dir)
        elif "threshold" in model.cfg:
            threshold = float(model.cfg["threshold"])
            print(f"使用部署包内标定的阈值: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"警告: 未标定阈值，使用默认 {threshold}（正常样本分数可能高于它，建议 --calibrate）")
    print(f"判定阈值: {threshold:.4f}")

    # 确定输入
    if args.input:
        in_dir = Path(args.input)
    else:
        # 默认：MVTec 测试集（从 config 读路径）
        import yaml
        with open("configs/patchcore_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        in_dir = Path(cfg["dataset"]["root"]) / args.category / "test"

    if not in_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {in_dir}")

    out_root = Path(args.out or os.path.join("outputs", "demo", args.category))
    vis_dir = out_root / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入: {in_dir}")
    print(f"输出: {out_root}")

    results = []
    files = sorted(in_dir.glob("*.png")) + sorted(in_dir.glob("*.jpg"))
    if not files:
        # 可能是子目录结构（good/defective/...）
        for sub in sorted(in_dir.iterdir()):
            if sub.is_dir():
                files += sorted(sub.glob("*.png")) + sorted(sub.glob("*.jpg"))
    files = sorted(set(files))
    print(f"共 {len(files)} 张图片")

    for p in files:
        vis_path = vis_dir / f"{p.stem}_vis.png"
        try:
            r = infer_single(model, str(p), str(vis_path), threshold)
            r["vis_path"] = str(vis_path)
            results.append(r)
        except Exception as e:
            print(f"  [跳过] {p.name}: {type(e).__name__}: {str(e)[:60]}")

    html = build_html(results, args.category, threshold)
    html_path = out_root / "result.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"\nHTML 报告已生成: {html_path}")


if __name__ == "__main__":
    main()
