"""Why does stacking the pre and post capture make the model worse?

The paired input lost to post-only in every comparison -- pooled 0.740 vs 0.753,
Michael 0.666 vs 0.694, Harvey 0.511 vs 0.609. "We stacked two images and it got
worse" is a weak thing to report. If the two captures are misregistered, then
channels 0-2 and 3-5 disagree about where the building is, and the network is
handed spatial noise on top of whatever change signal exists. That is a finding,
and it is measurable.

xBD documents its pairs as co-registered. This checks that claim per crop, by
phase correlation between the pre and post patch. A sub-pixel offset would mean
the pairing failed for some other reason; an offset of several pixels on a
building whose median footprint is 26 pixels would mean the pair was never
usable as a per-pixel stack, and that a change-detection model here needs
explicit alignment rather than channel concatenation.

Reported per event, because the events differ in sensor, off-nadir angle and
season, and pooling has misled this project three times already.

Usage:
    python xbd/registration.py --sample 6000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "xbd"))


def shift_of(pre: np.ndarray, post: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel (dx, dy) aligning pre onto post, and the response peak.

    Phase correlation on a Hann-windowed grayscale patch. The response is worth
    keeping: a low peak means the two patches share little structure at all,
    which is itself informative about whether a stack could work.
    """
    import cv2

    a = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if a.shape != b.shape or min(a.shape) < 16:
        return np.nan, np.nan, np.nan
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--buildings", type=Path,
                    default=ROOT / "xbd/data/buildings.parquet")
    ap.add_argument("--radiometry", type=Path,
                    default=ROOT / "xbd/data/scene_radiometry.parquet")
    ap.add_argument("--sample", type=int, default=6000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "xbd/data/registration.parquet")
    a = ap.parse_args()

    import cv2
    from dataset import load_meta

    m = load_meta(a.crops, "all4", 24.0, a.buildings, a.radiometry)
    if a.sample and len(m) > a.sample:
        # not groupby().apply(): from pandas 2.2 the grouping column is dropped
        # from the result, so `m.disaster` disappeared on the next line
        per = max(200, a.sample // max(m.disaster.nunique(), 1))
        m = pd.concat([g.sample(min(len(g), per), random_state=0)
                       for _, g in m.groupby("disaster")], ignore_index=True)
    print(f"measuring {len(m):,} crops across {m.disaster.nunique()} events")

    rows = []
    for n, (_, r) in enumerate(m.iterrows(), 1):
        if n % 500 == 0:
            print(f"  {n:,}/{len(m):,}", flush=True)
        pre = cv2.imread(str(a.crops / r.pre))
        post = cv2.imread(str(a.crops / r.post))
        if pre is None or post is None:
            continue
        dx, dy, resp = shift_of(pre, post)
        rows.append({"uid": r.uid, "disaster": r.disaster, "damage": r.damage,
                     "px_side": r.px_side, "dx": dx, "dy": dy, "resp": resp,
                     "shift": float(np.hypot(dx, dy))})
    d = pd.DataFrame(rows).dropna(subset=["shift"])
    d["shift_rel"] = d["shift"] / d.px_side
    d.to_parquet(a.out, index=False)

    print(f"\n=== how far apart are the two captures? ===")
    q = d["shift"].quantile([.25, .5, .75, .9]).round(2)
    print("  offset in pixels: " +
          ", ".join(f"p{int(k*100)}={v:.2f}" for k, v in q.items()))
    print(f"  median offset as a fraction of the building's own side: "
          f"{d.shift_rel.median():.3f}")
    print(f"  crops offset by more than 2 px:  {(d['shift'] > 2).mean():.1%}")
    print(f"  crops offset by more than a quarter of the building: "
          f"{(d.shift_rel > 0.25).mean():.1%}")

    print(f"\n  {'event':22s} {'n':>6} {'median px':>10} {'>2px':>7} "
          f"{'rel':>6} {'resp':>6}")
    for dis, g in d.groupby("disaster"):
        if len(g) < 50:
            continue
        print(f"  {str(dis)[:22]:22s} {len(g):6,} {g['shift'].median():10.2f} "
              f"{(g['shift'] > 2).mean():6.1%} {g.shift_rel.median():6.3f} "
              f"{g.resp.median():6.3f}")

    print("\n  A stack of two channels only works if they address the same")
    print("  pixels. On a 26-pixel building, an offset of a few pixels puts")
    print("  roof over ground and the network sees an edge that is not there.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
