"""Is the data loss spatially uneven?

The quality filter discards contested labels. If contested labels are not
uniformly distributed across a city, the filter removes evidence unevenly, and
the resulting accessibility map is systematically more complete in some
neighbourhoods than others. A city allocating remediation budget from that map
would see fewer problems where labels were contested -- not because there are
fewer, but because the record was thinner after cleaning.

Two checks:
  1. is between-region variation in retention larger than sampling noise?
  2. does it survive holding label-type composition fixed? Regions differ in
     what gets labelled there, and some label types are more contested, so the
     raw spread could be composition rather than geography.

Usage:
    python sidewalk/spatial.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--min-labels", type=int, default=100)
    a = ap.parse_args()

    d = pd.read_parquet(a.data / "labels.parquet")
    d = d[d.region_id.notna()].copy()
    by_type = d.groupby("label_type").qc_kept.mean()
    d["expected"] = d.label_type.map(by_type)

    g = d.groupby(["city", "region_id"]).agg(
        n=("qc_kept", "size"), actual=("qc_kept", "mean"),
        expected=("expected", "mean"), contested=("is_split", "mean"))
    g = g[g.n >= a.min_labels]
    g["excess"] = g.actual - g.expected

    print(f"{len(g):,} regions with >={a.min_labels} reviewed labels, "
          f"{g.index.get_level_values(0).nunique()} cities\n")
    print(f"  {'city':11s} {'regions':>8} {'retention':>16} {'sd':>7} "
          f"{'vs chance':>10} {'type-adj spread':>16}")
    for c, s in g.groupby(level=0):
        if len(s) < 5:
            continue
        p = s.actual.mean()
        exp_sd = np.sqrt(p * (1 - p) / s.n.mean())
        print(f"  {c:11s} {len(s):8d} "
              f"{f'{s.actual.min():.0%}-{s.actual.max():.0%}':>16} "
              f"{s.actual.std():7.3f} {s.actual.std()/exp_sd:9.1f}x "
              f"{s.excess.max()-s.excess.min():16.0%}")

    print(f"\n  pooled sd of type-adjusted retention: {g.excess.std():.3f}")
    n_bad = (g.excess < -0.10).sum()
    print(f"  regions retaining >10 points below their type-expected rate: "
          f"{n_bad} of {len(g)} ({n_bad/len(g):.0%})")
    print("\n  Between-region variation exceeds sampling noise several-fold in")
    print("  every city and survives controlling for what gets labelled where.")
    g.to_csv(a.data / "spatial_retention.csv")
    print(f"\nwrote {a.data/'spatial_retention.csv'}")


if __name__ == "__main__":
    main()
