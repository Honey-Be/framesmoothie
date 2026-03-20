
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framesmoothie._scripts.run_tiny_overfit import run_overfit
from framesmoothie._scripts.presets import list_presets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--presets", nargs="+", default=list(list_presets()))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/suites"))
    ap.add_argument("--suite-name", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--select-metric", type=str, default="loss")
    ap.add_argument("--select-mode", type=str, default="min", choices=["min", "max"])
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--early-stop-metric", type=str, default=None)
    ap.add_argument("--early-stop-mode", type=str, default=None, choices=[None, "min", "max"])
    ap.add_argument("--early-stop-patience", type=int, default=0)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--scheduler-kind", type=str, default="none", choices=["none", "cosine", "step"])
    ap.add_argument("--scheduler-step-size", type=int, default=50)
    ap.add_argument("--scheduler-gamma", type=float, default=0.5)
    ap.add_argument("--ema-schedule-kind", type=str, default="constant", choices=["constant", "linear", "cosine"])
    ap.add_argument("--ema-beta-start", type=float, default=0.99)
    ap.add_argument("--ema-beta-end", type=float, default=0.999)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite_name = args.suite_name or f"suite_{ts}"
    suite_dir = args.output_root / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    for preset in args.presets:
        summary = run_overfit(
            dataset=args.dataset,
            preset=preset,
            steps=args.steps,
            lr=args.lr,
            device=args.device,
            output_root=suite_dir,
            run_name=None,
            log_every=args.log_every,
            save_checkpoints=args.save_checkpoints,
            select_metric=args.select_metric,
            select_mode=args.select_mode,
            resume_from=None,
            topk=args.topk,
            early_stop_metric=args.early_stop_metric,
            early_stop_mode=args.early_stop_mode,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            scheduler_kind=args.scheduler_kind,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
            ema_schedule_kind=args.ema_schedule_kind,
            ema_beta_start=args.ema_beta_start,
            ema_beta_end=args.ema_beta_end,
        )
        summaries.append(summary)

    (suite_dir / "suite_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = sorted({k for s in summaries for k in s.keys()})
    with (suite_dir / "suite_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    # Plot quick summary over presets
    df = pd.DataFrame(summaries)
    if not df.empty and "preset" in df.columns:
        for metric in [
            "final_loss",
            "best_metric",
            "final_loss_src_sem",
            "final_loss_src_inst",
            "final_loss_tgt_sem",
            "final_loss_tgt_inst",
            "steps_executed",
        ]:
            if metric in df.columns:
                plt.figure(figsize=(8, 4))
                xs = df["preset"].astype(str).tolist()
                ys = df[metric].tolist()
                plt.plot(xs, ys, marker="o")
                plt.xlabel("preset")
                plt.ylabel(metric)
                plt.title(f"Suite summary: {metric}")
                plt.tight_layout()
                plt.savefig(suite_dir / f"plot_{metric}.png", dpi=160)
                plt.close()

    print(f"Suite written to: {suite_dir}")


if __name__ == "__main__":
    main()
