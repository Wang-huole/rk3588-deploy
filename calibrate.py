"""统一阈值标定：对所有已导出类别，用同一口径标定判定阈值并写回部署包。

为什么需要单独一步：
    异常分数是 L2 距离，正常样本也有基础分数，不能拿固定值（0.5 / 0.85）当阈值。
    不同类别的分数尺度不同，必须"按类别、按同一口径"分别标定，否则 Web 页面切
    类别时同一个分数含义会变，产线上也无法解释判定依据。

口径（全类别一致）：
    threshold = 正常样本分数的 P{percentile}（默认 P99）
    - 正常样本来源优先级：--normal_dir（产线现场采集）> test/good > train/good
    - 正常情况下 P99 表示"误报率约 1%"；样本数较少时 P99 接近最大值，
      等价于"标定集内没有正常样本超过阈值"，对上线是偏保守（宁可误报不漏检）。

指标说明：
    AUROC / AUPRO 与阈值无关，因此用 test/good 标定阈值不会让检测精度指标虚高；
    但"正常样本误报率"是在标定集上算的，属于自身口径，报告中会明确标注。

用法:
    python calibrate.py                              # 全部类别，自动选正常样本来源
    python calibrate.py --category toothbrush
    python calibrate.py --percentile 99.5            # 改分位
    python calibrate.py --normal_dir D:\\normal      # 用产线现场正常样本（推荐）
    python calibrate.py --dry_run                    # 只计算不写回 config.json

产物:
    deploy/<category>/config.json                    # 写入 threshold 及来源元信息
    outputs/calibration_report.md                    # 可读的标定报告
    outputs/calibration_summary.csv                  # 机器可读汇总
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from deploy_infer import DeployPatchCore, imread_unicode

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFECT_DIRS_NAME = "test"  # MVTec 结构：test/good + test/<defect_type>


def iter_images(folder: Path) -> list[Path]:
    """列出目录下的图片文件（非递归）。"""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def score_files(model: DeployPatchCore, files: list[Path]) -> np.ndarray:
    """逐张打分，读不出来的文件跳过并告警。"""
    scores = []
    for p in files:
        img = imread_unicode(str(p))
        if img is None:
            print(f"  [跳过] 无法读取: {p.name}")
            continue
        score, _ = model.predict(img)
        scores.append(float(score))
    return np.asarray(scores, dtype=np.float64)


def resolve_normal_dir(root: Path, category: str, normal_dir: str | None) -> tuple[Path, str]:
    """决定标定用的正常样本目录，返回 (目录, 口径标识)。"""
    if normal_dir:
        d = Path(normal_dir)
        if d.name == category or (d / category).is_dir():
            d = d if d.name == category else d / category
        return d, "site_normal"          # 产线现场采集
    test_good = root / category / "test" / "good"
    if iter_images(test_good):
        return test_good, "test_good"
    return root / category / "train" / "good", "train_good"


def split_roles(root: Path, category: str) -> dict[str, list[Path]]:
    """返回 {good: [...], defect: [...]}（仅测试集，用于误报/召回诊断）。"""
    test_dir = root / category / DEFECT_DIRS_NAME
    out: dict[str, list[Path]] = {"good": [], "defect": []}
    if not test_dir.is_dir():
        return out
    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        files = iter_images(sub)
        out["good" if sub.name == "good" else "defect"].extend(files)
    return out


def summarize(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {}
    return {
        "n": int(scores.size),
        "min": round(float(scores.min()), 4),
        "mean": round(float(scores.mean()), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p99": round(float(np.percentile(scores, 99)), 4),
        "max": round(float(scores.max()), 4),
    }


def calibrate_category(
    category: str,
    deploy_root: Path,
    data_root: Path,
    normal_dir: str | None,
    percentile: float,
    dry_run: bool,
) -> dict:
    print(f"\n=== 标定类别: {category} ===")
    model = DeployPatchCore(category, deploy_root=str(deploy_root))

    basis_dir, basis = resolve_normal_dir(data_root, category, normal_dir)
    normal_files = iter_images(basis_dir)
    if not normal_files:
        raise FileNotFoundError(f"标定目录无正常样本: {basis_dir}")

    normal_scores = score_files(model, normal_files)
    if normal_scores.size == 0:
        raise RuntimeError("标定失败：没有可用正常样本")
    threshold = float(np.percentile(normal_scores, percentile))
    stats = summarize(normal_scores)

    # 域差异参考：另一份正常样本的 P99（不用于定阈值，只做对比展示）
    alt_dir = data_root / category / ("train" if "test" in basis else "test") / "good"
    alt_scores = score_files(model, iter_images(alt_dir)) if iter_images(alt_dir) else np.asarray([])

    # 诊断：测试集上的误报率与缺陷召回率
    roles = split_roles(data_root, category)
    good_scores = score_files(model, roles["good"]) if roles["good"] else np.asarray([])
    defect_scores = score_files(model, roles["defect"]) if roles["defect"] else np.asarray([])
    fpr = float((good_scores > threshold).mean()) if good_scores.size else float("nan")
    recall = float((defect_scores > threshold).mean()) if defect_scores.size else float("nan")

    print(f"  正常样本来源: {basis} ({basis_dir})")
    print(f"  正常分数: min={stats['min']} mean={stats['mean']} max={stats['max']} (n={stats['n']})")
    print(f"  阈值 P{percentile:g} = {threshold:.4f}")
    if alt_scores.size:
        print(f"  参考域 {alt_dir.parent.name}/good P99 = {np.percentile(alt_scores, 99):.4f} "
              f"(min={alt_scores.min():.4f} max={alt_scores.max():.4f})")
    if good_scores.size:
        print(f"  测试集正常误报率 FPR = {fpr:.1%}")
    if defect_scores.size:
        print(f"  测试集缺陷召回率 Recall = {recall:.1%}")

    cfg_path = deploy_root / category / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        deploy_cfg = json.load(f)

    deploy_cfg.update({
        "threshold": round(threshold, 4),
        "threshold_basis": basis,
        "threshold_source": str(basis_dir),
        "threshold_percentile": percentile,
        "threshold_n": stats["n"],
        "threshold_normal_stats": {k: v for k, v in stats.items() if k != "n"},
        "threshold_diagnostics": {
            "test_good_fpr": None if np.isnan(fpr) else round(fpr, 4),
            "test_defect_recall": None if np.isnan(recall) else round(recall, 4),
        },
        "calibrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "threshold_note": (
            f"正常样本 P{percentile:g} 标定（{basis}，n={stats['n']}）；"
            "样本数较少时 P99 接近最大值，等价于标定集内无误报。"
            "正式上线请用产线相机实拍正常样本重新标定（--normal_dir）。"
        ),
    })
    # 清掉历史字段，避免旧口径残留
    deploy_cfg.pop("threshold_percentile_raw", None)

    if not dry_run:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(deploy_cfg, f, indent=2, ensure_ascii=False)
        print(f"  已写回: {cfg_path}")

    return {
        "category": category,
        "basis": basis,
        "basis_dir": str(basis_dir),
        "n_normal": stats["n"],
        "threshold": round(threshold, 4),
        "normal_min": stats["min"],
        "normal_mean": stats["mean"],
        "normal_max": stats["max"],
        "alt_p99": (round(float(np.percentile(alt_scores, 99)), 4) if alt_scores.size else None),
        "test_good_fpr": (None if np.isnan(fpr) else round(fpr, 4)),
        "test_defect_recall": (None if np.isnan(recall) else round(recall, 4)),
        "image_auroc": deploy_cfg.get("image_auroc"),
    }


def write_report(rows: list[dict], percentile: float, path: Path) -> None:
    """生成可读报告（Markdown）与机器可读汇总（CSV）。"""
    lines = [
        "# 阈值标定报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 统一口径：threshold = 正常样本分数的 P{percentile:g}（误报率约 {100 - percentile:.1f}%）",
        "- 正常样本来源优先级：产线现场样本 > test/good > train/good",
        "- AUROC/AUPRO 与阈值无关，用 test/good 标定不会让精度指标虚高；"
        "表中 FPR 是标定集自身口径。",
        "",
        "| 类别 | 正常样本来源 | n | 正常分数 mean / max | 阈值 P99 | 正常误报率 | 缺陷召回率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        fpr = "-" if r["test_good_fpr"] is None else f"{r['test_good_fpr']:.1%}"
        rec = "-" if r["test_defect_recall"] is None else f"{r['test_defect_recall']:.1%}"
        lines.append(
            f"| {r['category']} | {r['basis']} | {r['n_normal']} | "
            f"{r['normal_mean']:.3f} / {r['normal_max']:.3f} | {r['threshold']:.3f} | {fpr} | {rec} |"
        )
    lines += [
        "",
        "> 说明：样本数较少（n<100）时 P99 非常接近最大值，标定集内误报率为 0 属于正常现象；"
        "上线后建议持续用现场数据复核阈值。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n报告已生成: {path}")
    print(f"汇总已生成: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="统一阈值标定（写回 deploy/<category>/config.json）")
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--category", default=None, help="只标定一个类别；默认 deploy/ 下全部")
    parser.add_argument("--deploy_root", default="deploy")
    parser.add_argument("--normal_dir", default=None,
                        help="正常样本目录（产线现场采集，推荐）；不填则自动选 test/good 或 train/good")
    parser.add_argument("--percentile", type=float, default=99.0, help="正常分数分位，默认 99")
    parser.add_argument("--dry_run", action="store_true", help="只计算不写回 config.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        app_cfg = yaml.safe_load(f)
    data_root = Path(app_cfg["dataset"]["root"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {data_root}（可用 --normal_dir 指定现场样本）")

    deploy_root = Path(args.deploy_root)
    categories = (
        [args.category] if args.category
        else sorted(d.name for d in deploy_root.iterdir()
                    if d.is_dir() and (d / "config.json").exists())
    )

    rows = []
    for cat in categories:
        rows.append(calibrate_category(
            cat, deploy_root, data_root, args.normal_dir, args.percentile, args.dry_run
        ))

    write_report(rows, args.percentile, Path("outputs") / "calibration_report.md")
    if args.dry_run:
        print("\n[dry_run] 未写回任何 config.json")


if __name__ == "__main__":
    main()
