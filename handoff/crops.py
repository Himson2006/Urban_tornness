"""Fetch the image crop behind each Project Sidewalk label.

This is the piece the sidewalk study never needed and this one cannot do
without: paper/main.tex argues about labels and reviewers, but a prototype
model has to look at pixels. The exports carry no imagery, only a pointer --
`pano_id` plus the label's position inside the Google Street View panorama.

**Where the view comes from.** `pano_x, pano_y` locate the label in the full
equirectangular panorama, so a request centred there puts the label in the
middle of the returned image:

    heading = camera_heading + pano_x / pano_width * 360 - 180   (mod 360)
    pitch   = 90 - pano_y / pano_height * 180

The two correction terms in the heading were derived empirically against the
`heading` column and are not optional. The panorama's x origin sits at the
*back* of the vehicle, hence the 180 degree flip, and x is measured relative to
the camera rather than to true north, hence `camera_heading`. Without both,
`pano_x` and `heading` correlate at r=0.008 -- pure noise, and every crop would
point somewhere random. With both, the residual has median -0.8 degrees and
correlates at 0.93 with the label's horizontal offset inside the labelling
canvas, which is exactly what should be left over: the `heading` column is the
camera pose when the labeller was working, and the label sits at `canvas_x`
within that viewport.

Pitch needs no such correction; the vertical convention is plain
equirectangular (r=0.60 against the camera pitch, the gap again being the
canvas offset).

`--pose camera` derives the view the other way, from the camera pose plus the
canvas offset. It is the cross-check, not the default: it assumes the canvas is
rectilinear at the same fov as the request. The two should now agree to a few
degrees.

**The zoom -> fov map is a guess and must be checked.** The Street View JS API
relation is fov = 180 / 2**zoom, giving 90/45/22.5 degrees for the zoom levels
1-3 that appear here. Project Sidewalk may not use it directly. Run
`--dry-run --contact-sheet` first and look at the result: if labels are not
roughly centred and roughly filling the frame, fix FOV_FOR_ZOOM before
spending money on the full pull.

**This costs money and is rate-limited.** The Street View Static API is billed
per request. Direct tile scraping of panoramas is not an option -- it is
against Google's terms -- so the API with a key is the only route. Budget with
`--dry-run` before running for real.

Usage:
    export GSV_API_KEY=...
    python handoff/crops.py --task ramp --dry-run
    python handoff/crops.py --task ramp --n-per-class 12000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ENDPOINT = "https://maps.googleapis.com/maps/api/streetview"
# Verify against a contact sheet before trusting. See the module docstring.
FOV_FOR_ZOOM = {1: 90.0, 2: 45.0, 3: 22.5}
# US$ per 1000 requests, Street View Static API. Check current rates; this is
# only here so --dry-run gives a number worth reacting to.
USD_PER_1K = 7.0


def view_from_pano(d: pd.DataFrame) -> pd.DataFrame:
    """Heading/pitch/fov centred on the label's panorama position.

    The `+ camera_heading - 180` is load-bearing; see the module docstring.
    """
    d = d.copy()
    d["req_heading"] = (d.camera_heading
                        + d.pano_x / d.pano_width * 360.0 - 180.0) % 360.0
    d["req_pitch"] = (90.0 - d.pano_y / d.pano_height * 180.0).clip(-90, 90)
    z = d.zoom.round().astype("Int64")
    d["req_fov"] = z.map(FOV_FOR_ZOOM).astype(float).fillna(90.0)
    return d


def view_from_camera(d: pd.DataFrame) -> pd.DataFrame:
    """Heading/pitch from the labeller's viewport, offset by the canvas hit.

    Kept as a cross-check on view_from_pano, not as the default: it assumes the
    canvas is a rectilinear projection with the same fov the request will use,
    which is close enough to validate against but not to rely on.
    """
    d = d.copy()
    z = d.zoom.round().astype("Int64")
    fov = z.map(FOV_FOR_ZOOM).astype(float).fillna(90.0)
    # fraction of the canvas away from centre, scaled by the horizontal fov
    dx = (d.canvas_x / d.canvas_width - 0.5)
    dy = (d.canvas_y / d.canvas_height - 0.5)
    aspect = d.canvas_height / d.canvas_width
    d["req_heading"] = (d.heading + dx * fov) % 360.0
    d["req_pitch"] = (d.pitch - dy * fov * aspect).clip(-90, 90)
    d["req_fov"] = fov
    return d


def check_pose_agreement(d: pd.DataFrame) -> None:
    """Both derivations should land in the same place. Report it when they do not.

    This is the guard that caught the missing camera_heading term: a median
    |heading delta| near 90 degrees is not a small disagreement, it is the
    median of a uniform distribution, i.e. no relationship at all. Anything
    above ~20 degrees here means the geometry is wrong and the crops would be
    pictures of nothing in particular.
    """
    a, b = view_from_pano(d), view_from_camera(d)
    dh = (a.req_heading - b.req_heading + 180) % 360 - 180
    dp = a.req_pitch - b.req_pitch
    print("pose agreement (pano-derived vs camera-derived)")
    for name, x in (("heading", dh), ("pitch", dp)):
        print(f"  |{name} delta|  median {x.abs().median():6.2f}  "
              f"p90 {x.abs().quantile(.9):6.2f} deg   "
              f"within 20 deg: {(x.abs() < 20).mean():.1%}")


def sample(d: pd.DataFrame, classes: list[str], n_per_class: int,
           seed: int = 0) -> pd.DataFrame:
    """Class-balanced sample, grouped so a panorama never straddles the split.

    Two labels on the same panorama share lighting, camera and often the same
    stretch of pavement. Sampling them independently and splitting at random is
    the leak already fixed twice in this project; the group key travels with
    the sample so the dataset builder cannot forget it.
    """
    d = d[d.label_type.isin(classes)].copy()
    d = d[d.pano_id.notna() & d.pano_x.notna() & d.pano_y.notna()
          & (d.pano_width > 0) & (d.pano_height > 0)]

    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        g = d[d.label_type == c]
        take = min(n_per_class, len(g))
        if take < n_per_class:
            print(f"  warning: {c} has only {len(g):,}, wanted {n_per_class:,}")
        out.append(g.iloc[rng.permutation(len(g))[:take]])
    s = pd.concat(out, ignore_index=True)
    s["group"] = s.pano_id  # split key, never label_id
    return s


def crop_path(root: Path, row) -> Path:
    """Content-addressed by the request, so a changed fov refetches."""
    key = f"{row.pano_id}|{row.req_heading:.3f}|{row.req_pitch:.3f}|{row.req_fov:.2f}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return root / h[:2] / f"{h}.jpg"


def fetch(s: pd.DataFrame, out: Path, key: str, size: int = 448,
          sleep: float = 0.02) -> pd.DataFrame:
    import requests

    out.mkdir(parents=True, exist_ok=True)
    paths, ok = [], []
    sess = requests.Session()
    for i, row in enumerate(s.itertuples(), 1):
        p = crop_path(out, row)
        paths.append(str(p.relative_to(out)))
        if p.exists():
            ok.append(True)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        params = {"size": f"{size}x{size}", "pano": row.pano_id,
                  "heading": f"{row.req_heading:.3f}",
                  "pitch": f"{row.req_pitch:.3f}",
                  "fov": f"{row.req_fov:.2f}", "return_error_code": "true",
                  "key": key}
        try:
            r = sess.get(ENDPOINT, params=params, timeout=20)
            # A 200 can still be the grey "no imagery" placeholder; those are
            # small. Treat anything tiny as a miss rather than training on it.
            good = r.status_code == 200 and len(r.content) > 5000
            if good:
                p.write_bytes(r.content)
            ok.append(good)
        except Exception as e:
            print(f"  {row.pano_id}: {e}")
            ok.append(False)
        time.sleep(sleep)
        if i % 500 == 0:
            print(f"  {i:,}/{len(s):,}  ok={np.mean(ok):.1%}")

    s = s.assign(crop=paths, fetched=ok)
    print(f"fetched {sum(ok):,}/{len(s):,} ({np.mean(ok):.1%})")
    return s


def main():
    from targets import TASKS

    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=Path,
                    default=ROOT / "handoff/data/targets.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "handoff/data/crops")
    ap.add_argument("--task", default="ramp", choices=sorted(TASKS))
    ap.add_argument("--n-per-class", type=int, default=12000)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--pose", default="pano", choices=("pano", "camera"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the sampling plan and cost, fetch nothing")
    a = ap.parse_args()

    d = pd.read_parquet(a.targets)
    s = sample(d, TASKS[a.task], a.n_per_class, a.seed)
    s = (view_from_pano if a.pose == "pano" else view_from_camera)(s)

    print(f"\ntask={a.task}  {len(s):,} labels  "
          f"{s.pano_id.nunique():,} panoramas  {s.city.nunique()} cities")
    print(s.label_type.value_counts().to_string())
    print(f"\nfov requested: "
          f"{s.req_fov.value_counts().sort_index().to_dict()}")
    check_pose_agreement(s)

    n_new = sum(not crop_path(a.out, r).exists() for r in s.itertuples())
    print(f"\n{n_new:,} requests needed ({len(s) - n_new:,} already on disk)")
    print(f"estimated cost: ${n_new / 1000 * USD_PER_1K:,.2f} "
          f"at ${USD_PER_1K:.2f}/1k -- verify current pricing")

    if a.dry_run:
        print("\ndry run: nothing fetched")
        return

    key = os.environ.get("GSV_API_KEY")
    if not key:
        raise SystemExit("set GSV_API_KEY")
    s = fetch(s, a.out, key, a.size)
    dest = a.targets.parent / f"crops_{a.task}.parquet"
    s[s.fetched].to_parquet(dest, index=False)
    print(f"wrote {dest} ({int(s.fetched.sum()):,} rows)")


if __name__ == "__main__":
    main()
