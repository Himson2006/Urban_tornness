"""Look at the paired crops before training anything on them.

The whole premise is that damage is legible as a *difference* between the pre-
and post-disaster crop. That is an empirical claim about 30-pixel buildings, not
a given, and it is cheap to check by eye. If the pre and post columns are
indistinguishable for major-damage, the premise is wrong and no amount of
training will rescue it.

Rows are buildings, sampled per damage class. Columns are pre, post, and the
absolute difference -- the difference panel is what the 6-channel model has
access to that a post-only model does not.

Usage:
    python xbd/contact_sheet.py --per-class 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CELL = 96


def load(p: Path) -> np.ndarray:
    import cv2
    im = cv2.imread(str(p))
    if im is None:
        return np.zeros((CELL, CELL, 3), np.uint8)
    return cv2.resize(im, (CELL, CELL), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--min-side", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "xbd/data/contact_sheet.png")
    a = ap.parse_args()

    import cv2

    m = pd.read_parquet(a.crops / "crop_meta.parquet")
    m = m[m.px_side >= a.min_side]
    print(f"{len(m):,} crops >= {a.min_side:.0f} px")

    blocks = []
    for cls in ORDER:
        s = m[m.damage == cls]
        if s.empty:
            print(f"  {cls:14s} none")
            continue
        s = s.sample(min(a.per_class, len(s)), random_state=a.seed)
        print(f"  {cls:14s} {len(s)} shown of {(m.damage == cls).sum():,}")
        for _, r in s.iterrows():
            pre, post = load(a.crops / r.pre), load(a.crops / r.post)
            diff = cv2.absdiff(pre, post)
            # stretch the difference so it is visible at all; the model sees the
            # raw channels, this is presentation only
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            row = np.hstack([pre, post, diff])
            cv2.putText(row, cls.split("-")[0], (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(row, f"{r.px_side:.0f}px", (3, CELL - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
            blocks.append(row)
        blocks.append(np.full((3, CELL * 3, 3), 60, np.uint8))

    if not blocks:
        raise SystemExit("nothing to show")
    sheet = np.vstack(blocks)
    hdr = np.full((18, CELL * 3, 3), 30, np.uint8)
    for i, t in enumerate(["pre", "post", "|diff|"]):
        cv2.putText(hdr, t, (i * CELL + 4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1)
    cv2.imwrite(str(a.out), np.vstack([hdr, sheet]))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
