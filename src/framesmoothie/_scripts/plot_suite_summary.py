from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", type=Path, required=True)
    args = ap.parse_args()

    csv_path = args.suite_dir / "suite_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing suite summary CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "preset" not in df.columns:
        raise ValueError("suite_summary.csv must contain a 'preset' column")

    plotted = 0
    for metric in [
        "final_loss",
        "best_metric",
        "final_loss_src_sem",
        "final_loss_src_inst",
        "final_loss_tgt_sem",
        "final_loss_tgt_inst",
    ]:
        if metric not in df.columns:
            continue
        series = df[metric]
        if series.isna().all():
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(df["preset"].astype(str).tolist(), series.tolist(), marker="o")
        plt.xlabel("preset")
        plt.ylabel(metric)
        plt.title(f"Suite summary: {metric}")
        plt.tight_layout()
        plt.savefig(args.suite_dir / f"plot_{metric}.png", dpi=160)
        plt.close()
        plotted += 1

    print(f"Generated {plotted} plot(s) in {args.suite_dir}")


if __name__ == "__main__":
    main()
