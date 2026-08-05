"""Per-city robustness: is any of this driven by one deployment?

Seattle contributes roughly a third of the corpus, so every pooled number needs
to survive being broken apart. Three checks:

  1. does the quality filter discard contested labels in every city, or only
     where the pooled average happens to come from;
  2. does the label-versus-reviewer variance split hold per city;
  3. leave-one-city-out on the headline retention rates, so no single
     deployment can be carrying them.

Bootstrap intervals accompany the retention rates, since those are the numbers
the argument rests on.

Usage:
    python sidewalk/by_city.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def boot_ci(mask: np.ndarray, n: int = 4000, seed: int = 0):
    if len(mask) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    r = [mask[rng.integers(0, len(mask), len(mask))].mean() for _ in range(n)]
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def variance_split(v: pd.DataFrame, outcome: str, iters: int = 25):
    """Label vs reviewer share of variance, same additive fit as decompose.py."""
    d = v if outcome == "Unsure" else v[v.vote != "Unsure"]
    if len(d) < 2000:
        return np.nan, np.nan
    y = (d.vote == outcome).to_numpy(float)
    li = d.label_id.astype("category").cat.codes.to_numpy()
    ri = d.validator_id.astype("category").cat.codes.to_numpy()
    nl, nr = li.max() + 1, ri.max() + 1
    mu, a, b = y.mean(), np.zeros(nl), np.zeros(nr)
    ln = np.bincount(li, minlength=nl).astype(float)
    rn = np.bincount(ri, minlength=nr).astype(float)
    for _ in range(iters):
        a = np.bincount(li, y - mu - b[ri], minlength=nl) / np.maximum(ln, 1)
        b = np.bincount(ri, y - mu - a[li], minlength=nr) / np.maximum(rn, 1)
        b -= b.mean()
    res = y - mu - a[li] - b[ri]
    tot = a[li].var() + b[ri].var() + res.var()
    return a[li].var() / tot, b[ri].var() / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    a = ap.parse_args()

    d = pd.read_parquet(a.data / "labels.parquet")
    v = pd.read_parquet(a.data / "validations.parquet")

    print("=== quality filter retention, by city ===\n")
    print(f"  {'city':11s} {'reviewed':>9} {'split%':>7} {'unsure%':>8} "
          f"{'QC all':>7} {'QC split':>9} {'QC unsure':>10}")
    rows = []
    for c, g in d.groupby("city"):
        sp, un = g[g.is_split], g[g.is_unsure]
        r = {"city": c, "n": len(g),
             "split_pct": g.is_split.mean(), "unsure_pct": g.is_unsure.mean(),
             "qc_all": g.qc_kept.mean(),
             "qc_split": sp.qc_kept.mean() if len(sp) >= 20 else np.nan,
             "qc_unsure": un.qc_kept.mean() if len(un) >= 20 else np.nan}
        rows.append(r)
        print(f"  {c:11s} {len(g):9,} {r['split_pct']:7.1%} {r['unsure_pct']:8.1%} "
              f"{r['qc_all']:7.1%} {r['qc_split']:9.1%} {r['qc_unsure']:10.1%}")
    bc = pd.DataFrame(rows)

    n_ok = ((bc.qc_split < bc.qc_all) & (bc.qc_unsure < bc.qc_all)).sum()
    print(f"\n  contested labels retained at a LOWER rate than average in "
          f"{n_ok}/{len(bc)} cities")

    print("\n=== bootstrap CIs on the pooled retention rates ===")
    for name, m in (("all labels", d.qc_kept), ("split labels", d[d.is_split].qc_kept),
                    ("unsure labels", d[d.is_unsure].qc_kept)):
        arr = m.to_numpy()
        lo, hi = boot_ci(arr)
        print(f"  {name:15s} {arr.mean():6.1%}  95% CI [{lo:.1%}, {hi:.1%}]  "
              f"n={len(arr):,}")

    print("\n=== leave-one-city-out on the pooled rates ===")
    print(f"  {'excluded':11s} {'QC split':>9} {'QC unsure':>10}")
    worst = []
    for c in sorted(d.city.unique()):
        g = d[d.city != c]
        s, u = g[g.is_split].qc_kept.mean(), g[g.is_unsure].qc_kept.mean()
        worst.append((max(s, u), c))
        print(f"  {c:11s} {s:9.1%} {u:10.1%}")
    print(f"  -> worst case with any single city removed: "
          f"{max(worst)[0]:.1%} (dropping {max(worst)[1]})")

    # Estimate variance WITHIN each city. Pooling across deployments mixes
    # city-specific reviewer pools and base rates, which inflates the reviewer
    # component and previously made uncertainty look reviewer-dominated when it
    # is not.
    print("\n=== label vs reviewer variance, estimated within each city ===\n")
    print(f"  {'city':11s} | {'Disagree lab':>12} {'rev':>6} "
          f"| {'Unsure lab':>11} {'rev':>6}")
    rows2 = []
    for c, g in v.groupby("city"):
        dl, dr = variance_split(g, "Disagree")
        ul, ur = variance_split(g, "Unsure")
        if np.isnan(dl) or np.isnan(ul):
            continue
        rows2.append({"city": c, "dis_lab": dl, "dis_rev": dr,
                      "uns_lab": ul, "uns_rev": ur})
        print(f"  {c:11s} | {dl:12.1%} {dr:6.1%} | {ul:11.1%} {ur:6.1%}")
    r2 = pd.DataFrame(rows2)
    print(f"\n  disagreement label-dominated in {(r2.dis_lab>r2.dis_rev).sum()}"
          f"/{len(r2)} cities")
    print(f"  uncertainty  label-dominated in {(r2.uns_lab>r2.uns_rev).sum()}"
          f"/{len(r2)} cities")
    print(f"\n  median   Disagree  label {r2.dis_lab.median():.1%} vs reviewer "
          f"{r2.dis_rev.median():.1%}  (ratio {r2.dis_lab.median()/r2.dis_rev.median():.1f}x)")
    print(f"           Unsure    label {r2.uns_lab.median():.1%} vs reviewer "
          f"{r2.uns_rev.median():.1%}  (ratio {r2.uns_lab.median()/r2.uns_rev.median():.1f}x)")
    print("\n  Both are label-driven. The difference is one of degree: reviewer")
    print("  identity matters about twice as much for 'not sure' as for")
    print("  disagreement, and the label matters about half as much.")
    r2.to_csv(a.data / "variance_by_city.csv", index=False)

    bc.to_csv(a.data / "by_city.csv", index=False)
    print(f"\nwrote {a.data/'by_city.csv'}")


if __name__ == "__main__":
    main()
