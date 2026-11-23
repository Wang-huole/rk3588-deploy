"""一键入口：对配置里的每个类别执行 训练 -> 评估 -> 汇总。

用法:
    python run.py --config configs/patchcore_config.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from src.evaluate import evaluate_category
from src.train import train_category
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patchcore_config.yaml")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练（已有 memory bank）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["dir"], exist_ok=True)

    rows = []
    for cat in cfg["dataset"]["categories"]:
        if not args.skip_train:
            train_category(cfg, cat)
        metrics = evaluate_category(cfg, cat)
        rows.append({"category": cat, **metrics})

    df = pd.DataFrame(rows)
    summary_path = os.path.join(cfg["output"]["dir"], "summary.csv")
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
