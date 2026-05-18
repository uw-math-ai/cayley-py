#!/usr/bin/env python3
"""Quick inspector for a finished Koltsov3 GB pipeline run directory.

Usage (from the pipeline folder):
    python inspect_results.py koltsov3_gb_pipeline_35265614

Prints three views designed to diagnose whether the model is learning:
  1. Per-config final metrics (train / val / test RMSE, R2, Spearman)
  2. Label statistics and unique-state fractions per n
  3. One representative training curve (val RMSE / R2 over iterations)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_results.py <output_dir>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    summary_csv = out_dir / "summary_results.csv"
    iters_csv = out_dir / "iteration_results.csv"
    if not summary_csv.is_file():
        print(f"ERROR: {summary_csv} not found")
        sys.exit(1)
    if not iters_csv.is_file():
        print(f"ERROR: {iters_csv} not found")
        sys.exit(1)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    summary = pd.read_csv(summary_csv)
    iters = pd.read_csv(iters_csv)

    print("=" * 80)
    print(f"  Sweep: {out_dir.name}  -  {len(summary)} configs, {len(iters)} iteration rows")
    print("=" * 80)

    # ----------------------------------------------------------------
    # View 1 -- final metrics per config
    # ----------------------------------------------------------------
    print("\n[1] Final per-config metrics (sorted by n, max_depth):")
    cols1 = [
        "n", "max_depth", "best_iteration",
        "train_rmse", "train_r2",
        "val_rmse", "val_r2",
        "test_rmse", "test_r2", "test_spearman",
    ]
    print(summary[cols1].sort_values(["n", "max_depth"]).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    # ----------------------------------------------------------------
    # View 2 -- labels & unique-state fractions (one row per n)
    # ----------------------------------------------------------------
    print("\n[2] Label stats & uniqueness per n (one config per n):")
    cols2 = [
        "n", "label_mean", "label_std",
        "n_train_states", "num_unique_train_states", "unique_train_fraction",
        "diameter",
    ]
    per_n = summary[cols2].drop_duplicates("n").sort_values("n")
    print(per_n.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    # ----------------------------------------------------------------
    # View 3 -- training curve for the largest config
    # ----------------------------------------------------------------
    largest_n = int(summary["n"].max())
    largest_depth = int(summary[summary["n"] == largest_n]["max_depth"].max())
    print(f"\n[3] Training curve for n={largest_n}, max_depth={largest_depth} (every 50 iter):")
    sub = iters[(iters["n"] == largest_n) & (iters["max_depth"] == largest_depth)].sort_values("iteration")
    if sub.empty:
        print("  (no iteration rows for this config)")
    else:
        sample = sub.iloc[::50]
        # Always include the last iteration too
        if sub.iloc[[-1]].iloc[0]["iteration"] not in set(sample["iteration"]):
            sample = pd.concat([sample, sub.iloc[[-1]]])
        cols3 = ["iteration", "train_rmse", "val_rmse", "val_r2", "val_spearman"]
        print(sample[cols3].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    # ----------------------------------------------------------------
    # Quick verdicts
    # ----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Quick verdicts")
    print("=" * 80)

    # Underfit vs overfit
    gap = (summary["val_rmse"] - summary["train_rmse"]).mean()
    print(f"  mean(val_rmse - train_rmse) = {gap:+.4f}")
    if abs(gap) < 0.01:
        print("    -> train and val are essentially equal: UNDERFITTING (model can't learn from the data)")
    elif gap > 0.05:
        print("    -> val much worse than train: OVERFITTING")
    else:
        print("    -> small gap: roughly balanced")

    # Capacity effect
    by_depth = summary.groupby("max_depth")["val_rmse"].mean()
    spread = by_depth.max() - by_depth.min()
    print(f"  val_rmse spread across max_depth = {spread:.5f}  (per-depth means: {dict(round(by_depth, 5))})")
    if spread < 0.001:
        print("    -> max_depth has no effect: capacity is not the bottleneck")

    # Predicting the mean?
    test_r2_mean = summary["test_r2"].mean()
    if test_r2_mean < 0.05:
        print(f"  mean test_r2 = {test_r2_mean:.4f}: model is essentially predicting the LABEL MEAN")

    # Label-collision check
    if "unique_train_fraction" in summary.columns:
        worst_unique = per_n["unique_train_fraction"].min()
        print(f"  min unique_train_fraction across n = {worst_unique:.4f}")
        if worst_unique < 0.5:
            print("    -> many duplicate states with potentially conflicting labels (LABEL COLLISION)")


if __name__ == "__main__":
    main()
