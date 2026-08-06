"""Build the building-level manifest and answer the go/no-go questions.

Three things decide whether this study is worth running, and all three are
answerable from labels alone:

  1. Does the object fill the frame? The pedestrian work failed because a
     pedestrian occupied a quarter of its crop and the prototypes drifted onto
     the road. A building crop is only useful here if the building is large
     enough in pixels to carry the evidence.

  2. Is there a contested boundary with enough mass on it? The damage scale is
     ordinal, so minor-vs-major is a threshold judgement in the way that
     no-damage-vs-destroyed is not. If the middle classes are rare, there is
     nothing to be torn about.

  3. Is ambiguity spatially structured? Every building carries a longitude and
     latitude, so "where is ambiguity concentrated" is answerable rather than
     aspirational.

Usage:
    python xbd/manifest.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAMAGE_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed",
                "un-classified"]


def parse_polygon(wkt: str) -> np.ndarray | None:
    """POLYGON ((x y, x y, ...)) -> Nx2 array."""
    m = re.search(r"\(\((.*?)\)\)", wkt or "")
    if not m:
        return None
    pts = [p.strip().split() for p in m.group(1).split(",")]
    try:
        return np.array([[float(a), float(b)] for a, b in pts])
    except ValueError:
        return None


def poly_area(p: np.ndarray) -> float:
    """Shoelace."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def parse_file(path: Path) -> list[dict]:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return []
    meta = d.get("metadata", {})
    xy = {f["properties"]["uid"]: f for f in d.get("features", {}).get("xy", [])
          if "properties" in f}
    rows = []
    for f in d.get("features", {}).get("lng_lat", []):
        pr = f.get("properties", {})
        uid = pr.get("uid")
        if pr.get("feature_type") != "building" or uid is None:
            continue
        g = parse_polygon(f.get("wkt"))
        px = parse_polygon(xy.get(uid, {}).get("wkt", ""))
        if g is None or px is None:
            continue
        w, h = np.ptp(px[:, 0]), np.ptp(px[:, 1])   # ndarray.ptp gone in numpy 2
        rows.append({
            "uid": uid,
            "scene": path.stem.split("__")[-1].replace("_post_disaster", ""),
            "split": path.stem.split("__")[0],
            "disaster": meta.get("disaster"),
            "damage": pr.get("subtype"),
            "lon": g[:, 0].mean(), "lat": g[:, 1].mean(),
            "px_w": w, "px_h": h,
            "px_area": poly_area(px),
            "px_side": float(np.sqrt(poly_area(px))),
            "gsd": meta.get("gsd"),
            "off_nadir": meta.get("off_nadir_angle"),
            "sun_elev": meta.get("sun_elevation"),
            "capture_date": meta.get("capture_date"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=ROOT / "xbd/data/labels")
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/data")
    ap.add_argument("--crop-scale", type=float, default=1.5,
                    help="crop side as a multiple of the building's side")
    a = ap.parse_args()

    files = sorted(a.labels.glob("*.json"))
    if not files:
        raise SystemExit(f"no label files in {a.labels}; run xbd/fetch_labels.py")
    rows = [r for f in files for r in parse_file(f)]
    d = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out / "buildings.parquet", index=False)

    print(f"{len(files):,} scenes -> {len(d):,} buildings, "
          f"{d.disaster.nunique()} disasters\n")

    print("=== 2. is there a contested boundary with mass on it? ===")
    vc = d.damage.value_counts()
    for k in DAMAGE_ORDER:
        if k in vc:
            print(f"  {k:16s} {vc[k]:>8,}  {vc[k]/len(d):6.1%}")
    mid = d.damage.isin(["minor-damage", "major-damage"]).sum()
    print(f"\n  minor + major (the ordinal middle): {mid:,} ({mid/len(d):.1%})")
    print("  these are the threshold judgements -- the analogue of the")
    print("  melanoma/nevus boundary that carried the medical work\n")

    print("=== 1. does the building fill the frame? ===")
    q = d.px_side.quantile([.1, .25, .5, .75, .9]).round(1)
    print("  building side in pixels: " +
          ", ".join(f"p{int(k*100)}={v:.0f}" for k, v in q.items()))
    frac = 1.0 / (a.crop_scale ** 2)
    print(f"  at crop_scale {a.crop_scale}, the building occupies "
          f"{frac:.0%} of the crop area")
    big = (d.px_side >= 40).mean()
    print(f"  buildings with side >= 40 px: {big:.1%}")
    print("  (PIE pedestrians occupied 25% of their crop and prototypes")
    print("   drifted off them; anything at or above that is workable)\n")
    print("  building size by damage class (median px side):")
    for k in DAMAGE_ORDER:
        s = d[d.damage == k]
        if len(s) > 50:
            print(f"    {k:16s} {s.px_side.median():5.0f} px   n={len(s):,}")

    print("\n=== 3. is ambiguity spatially structured? ===")
    print(f"  {'disaster':26s} {'buildings':>10} {'minor+major':>12} {'destroyed':>10}")
    for dis, g in d.groupby("disaster"):
        if len(g) < 200:
            continue
        m = g.damage.isin(["minor-damage", "major-damage"]).mean()
        z = (g.damage == "destroyed").mean()
        print(f"  {str(dis)[:26]:26s} {len(g):10,} {m:11.1%} {z:9.1%}")
    print("\n  every building carries lon/lat, so this can be taken down to")
    print("  neighbourhood scale once the population is fixed")

    print(f"\nwrote {a.out/'buildings.parquet'}")


if __name__ == "__main__":
    main()
