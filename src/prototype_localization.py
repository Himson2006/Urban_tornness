"""Where do the prototypes actually look?

The hand-off figures suggested the model anchors on the pedestrian-ground
interface -- feet, curb line -- rather than on the body or head. That is a claim
about every prototype, not about the two that happened to be retrieved, so it
should be counted rather than eyeballed.

Crops are centred on the bounding box at `crop_scale` x bbox, so the pedestrian
occupies the central 1/crop_scale fraction of the frame. For each prototype we
take its saved self-activation map, find the peak, and place it:

  inside / outside the pedestrian box
  and, within the box, head / torso / legs-feet / below-feet

If the peaks cluster at legs-feet and below, the model is reading foot placement
against the kerb -- a genuine cue, but one that cannot separate "about to step
out" from "staying put", which is exactly the case humans split on.

Usage:
    python src/prototype_localization.py --run runs_tight/resnet34_fold4 --crop-scale 1.3
    python src/prototype_localization.py --run runs/resnet34_fold4 --crop-scale 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# vertical bands within the pedestrian box, top-down
BANDS = [("head", 0.00, 0.20), ("torso", 0.20, 0.55),
         ("legs", 0.55, 0.85), ("feet", 0.85, 1.00)]


def find_proto_dir(run_dir: Path) -> Path:
    eps = sorted(run_dir.glob("img/epoch-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    if not eps:
        raise SystemExit(f"no pushed prototypes under {run_dir}/img/")
    return eps[-1]


def locate(act: np.ndarray) -> tuple[float, float]:
    """Peak of the self-activation map, in normalised (x, y) of the crop."""
    if act.ndim == 3:
        act = act[0]
    iy, ix = np.unravel_index(np.argmax(act), act.shape)
    return (ix + 0.5) / act.shape[1], (iy + 0.5) / act.shape[0]


def classify(x: float, y: float, crop_scale: float) -> tuple[bool, str]:
    """Inside the pedestrian box? And which vertical band of it?"""
    frac = 1.0 / crop_scale            # box side as a fraction of the crop
    lo, hi = 0.5 - frac / 2, 0.5 + frac / 2
    inside = (lo <= x <= hi) and (lo <= y <= hi)
    if not inside:
        return False, ("below-feet" if y > hi else
                       "above-head" if y < lo else "beside")
    ry = (y - lo) / max(hi - lo, 1e-9)   # 0 at head, 1 at feet
    for name, a, b in BANDS:
        if a <= ry < b or (name == "feet" and ry >= b):
            return True, name
    return True, "torso"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    pdir = find_proto_dir(a.run)
    acts = sorted(pdir.glob("prototype-self-act*.npy"),
                  key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    if not acts:
        raise SystemExit(
            f"no prototype-self-act*.npy in {pdir}\n"
            f"push must be run with prototype_self_act_filename_prefix set")

    rows = []
    for f in acts:
        j = int("".join(c for c in f.stem if c.isdigit()))
        x, y = locate(np.load(f))
        inside, band = classify(x, y, a.crop_scale)
        rows.append({"proto": j, "x": x, "y": y,
                     "inside_box": inside, "band": band})
    df = pd.DataFrame(rows).sort_values("proto")
    dest = a.out / f"proto_localization_{a.run.name}_s{a.crop_scale}.csv"
    df.to_csv(dest, index=False)

    frac = 1.0 / a.crop_scale
    print(f"{a.run.name} | crop_scale {a.crop_scale} -> pedestrian box occupies "
          f"the central {frac:.0%} of the frame")
    print(f"{len(df)} prototypes\n")
    print(f"  peaks INSIDE the pedestrian box : {df.inside_box.mean():.1%}")
    print(f"  peaks OUTSIDE                   : {1-df.inside_box.mean():.1%}\n")
    print("  band distribution:")
    vc = df.band.value_counts()
    for k, v in vc.items():
        print(f"    {k:12s} {v:>3}  {v/len(df):6.1%}  " + "#" * int(40*v/len(df)))

    low = df[(df.band.isin(["legs", "feet"])) | (df.band == "below-feet")]
    print(f"\n  at legs / feet / below the feet: {len(low)}/{len(df)} "
          f"({len(low)/len(df):.1%})")
    print(f"  mean normalised height of peak (0=top, 1=bottom): {df.y.mean():.3f}")
    if len(low) / len(df) > 0.5:
        print("\n  => The model anchors on the pedestrian-ground interface.")
        print("     Foot placement against the kerb is a real cue, but it cannot")
        print("     separate 'about to step out' from 'staying put' -- the exact")
        print("     case annotators split on.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
