#!/usr/bin/env python3
"""
Run wids_datathon.py multiple times with different seeds and average outputs.
This reduces variance and improves leaderboard generalization.

Usage:
    /opt/anaconda3/bin/python3 run_multi_seed.py            # 3 seeds
    /opt/anaconda3/bin/python3 run_multi_seed.py --runs 5   # 5 seeds
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 123, 456, 789, 2024]
DATA_DIR = Path(__file__).resolve().parent
PYTHON = "/opt/anaconda3/bin/python3"
SCRIPT = str(DATA_DIR / "wids_datathon.py")

VARIANTS = [
    "submission_stack.csv",
    "submission_stack_blend.csv",
    "submission_stack_reg.csv",
    "submission_blend.csv",
    "submission_jh.csv",
]


def enforce_mono_clip(df):
    cols = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
    for a, b in zip(cols, cols[1:]):
        df[b] = np.maximum(df[a], df[b])
    for c in cols:
        df[c] = df[c].clip(0.003, 0.997)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    seeds = SEEDS[:args.runs]
    print(f"Running {len(seeds)} seeds: {seeds}")

    collected = {v: [] for v in VARIANTS}

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{len(seeds)} — seed={seed}")
        print(f"{'='*60}")
        result = subprocess.run(
            [PYTHON, SCRIPT, "--seed", str(seed)],
            cwd=str(DATA_DIR),
        )
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            continue

        for v in VARIANTS:
            fp = DATA_DIR / v
            if fp.exists():
                df = pd.read_csv(fp)
                collected[v].append(df)
                print(f"  Collected {v}")
            else:
                print(f"  Missing {v}")

    print(f"\n{'='*60}")
    print("Averaging …")
    print(f"{'='*60}")

    cols = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
    for v, dfs in collected.items():
        if len(dfs) < 2:
            print(f"  {v}: only {len(dfs)} runs, skipping average")
            continue

        avg = dfs[0].copy()
        for c in cols:
            avg[c] = np.mean([df[c].values for df in dfs], axis=0)
        avg = enforce_mono_clip(avg)

        name = v.replace(".csv", f"_avg{len(dfs)}.csv")
        out_path = DATA_DIR / name
        avg.to_csv(out_path, index=False)
        print(f"  {name} (from {len(dfs)} runs)")
        for c in cols:
            print(f"    {c}: min={avg[c].min():.4f} mean={avg[c].mean():.4f} max={avg[c].max():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
