"""Diagnose whether the model produces any usable tornness at all.

Experiment 1 returned a null. Before concluding anything about the hypothesis we
need to know which of these is true:

  (a) the model is saturated -- p(cross) is ~0 or ~1 everywhere, so no torn
      predictions exist and the experiment never ran;
  (b) torn predictions exist but co-activation carries no signal (a real null);
  (c) the similarity features themselves are degenerate (no spread to correlate).

These need completely different responses, and the Experiment 1 table cannot
distinguish them.

Usage:
    python src/diag_tornness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["p_cross", "margin", "entropy", "coact_min", "coact_prod",
         "max_sim_0", "max_sim_1", "global_max_sim", "sim_gap",
         "typing_score", "mc_dropout_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--arch", default="resnet34")
    a = ap.parse_args()

    files = sorted(a.runs.glob(f"{a.arch}_fold*/tornness_fold*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    win = df[df.in_exp_window] if "in_exp_window" in df else df
    print(f"{len(files)} folds | {len(df):,} crops | {df.ped_id.nunique():,} peds"
          f" | in-window {len(win):,}\n")

    print("=== (a) is the model saturated? ===")
    p = win.p_cross.to_numpy()
    for lo, hi in [(0.0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.7),
                   (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]:
        n = ((p >= lo) & (p < hi)).sum()
        bar = "#" * int(60 * n / len(p))
        print(f"  p_cross [{lo:.2f},{hi:.2f}) {n:>8,} {100*n/len(p):5.1f}% {bar}")
    print(f"\n  CROP-level  margin<0.5: {(win.margin<0.5).mean():.4f}"
          f" | <0.2: {(win.margin<0.2).mean():.4f}"
          f" | <0.05: {(win.margin<0.05).mean():.4f}")

    print("\n=== (c) do the similarity features have spread? ===")
    d = win[[c for c in FEATS if c in win.columns]].describe(
        percentiles=[.01, .25, .5, .75, .99]).T
    print(d[["mean", "std", "min", "1%", "50%", "99%", "max"]].round(4).to_string())
    for c in ("coact_min", "global_max_sim", "max_sim_0", "max_sim_1"):
        if c in win and win[c].std() < 1e-6:
            print(f"  !! {c} is constant -- nothing to correlate")

    print("\n=== (b) signal at pedestrian level, various torn definitions ===")
    agg = {c: "mean" for c in FEATS if c in win.columns}
    agg["human_disagreement"] = "first"
    ped = win.groupby("ped_id").agg(agg)
    # a genuinely torn pedestrian: mean p(cross) near 0.5 across the window,
    # which survives per-frame saturation as long as frames disagree
    ped["p_dist_from_half"] = (ped.p_cross - 0.5).abs()

    for name, col, q in [("mean-margin lowest 25%", "margin", 0.25),
                         ("mean-margin lowest 10%", "margin", 0.10),
                         ("|p-0.5| lowest 25%", "p_dist_from_half", 0.25),
                         ("|p-0.5| lowest 10%", "p_dist_from_half", 0.10),
                         ("ALL pedestrians", None, 1.0)]:
        sub = ped if col is None else ped[ped[col] <= ped[col].quantile(q)]
        y = sub.human_disagreement.to_numpy()
        out = []
        for f in ("coact_min", "coact_prod", "typing_score", "entropy",
                  "margin", "mc_dropout_std"):
            if f not in sub or sub[f].std() == 0:
                continue
            r = stats.spearmanr(sub[f], y)
            star = "*" if r.pvalue < 0.05 else " "
            out.append(f"{f}={r.statistic:+.3f}{star}")
        print(f"  {name:24s} n={len(sub):>5,}  " + "  ".join(out))
    print("\n  (* = p<0.05)")

    print("\n=== reference: is human_disagreement itself well spread? ===")
    hd = ped.human_disagreement
    print(f"  mean {hd.mean():.3f} std {hd.std():.3f} | "
          f">0.5: {(hd>0.5).mean():.3f} | >0.8: {(hd>0.8).mean():.3f}")


if __name__ == "__main__":
    main()
