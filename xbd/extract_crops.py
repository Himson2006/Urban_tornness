"""Step B: extract paired pre/post building crops.

The pairing is the point. A 27-pixel building is thin evidence on its own, and
that resolution is what made me wary of this dataset. But xBD's pre- and
post-disaster images are co-registered, so the same pixel polygon addresses the
same building in both. The model's input is therefore not "what does this roof
look like" but "what changed here", and change survives at low resolution far
better than texture does.

That also gives the explanation a different object. A competing-prototype
account over paired crops says: this change looks like these buildings that were
judged minor, and also like these judged major. HAM10000 had no equivalent.

Storage discipline follows the PIE work: scenes are fetched through the Hub
cache, cropped, and the crops written out small. Crops are saved at native
resolution -- a 40x40 patch is under 4 KB -- so resizing stays a decision for
training rather than something baked into disk.

Usage:
    python xbd/extract_crops.py --limit 200      # try a few scenes first
    python xbd/extract_crops.py                  # everything in the manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = "aryananand/xBD"


def scene_paths(split: str, scene: str) -> tuple[str, str]:
    return (f"{split}/images/{scene}_pre_disaster.png",
            f"{split}/images/{scene}_post_disaster.png")


def crop_box(cx: float, cy: float, side: float, scale: float,
             W: int, H: int) -> tuple[int, int, int, int] | None:
    """Square box of `scale` x the building's side, clamped to the image."""
    half = max(side * scale / 2, 8)
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = int(round(cx + half)), int(round(cy + half))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "xbd/data/buildings.parquet")
    ap.add_argument("--labels", type=Path, default=ROOT / "xbd/data/labels")
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--crop-scale", type=float, default=1.5)
    ap.add_argument("--min-side", type=float, default=20.0,
                    help="skip buildings smaller than this, in pixels")
    ap.add_argument("--limit", type=int, default=0, help="0 = all scenes")
    ap.add_argument("--disaster", default="", help="restrict to one disaster")
    ap.add_argument("--seed", type=int, default=0,
                    help="scene sample is deterministic given --limit")
    a = ap.parse_args()

    import cv2
    import json
    import re
    from huggingface_hub import hf_hub_download

    d = pd.read_parquet(a.manifest)
    d = d[d.px_side >= a.min_side]
    if a.disaster:
        d = d[d.disaster == a.disaster]
        if d.empty:
            raise SystemExit(f"no buildings for disaster {a.disaster!r}")
    a.out.mkdir(parents=True, exist_ok=True)

    # We need per-building pixel polygons, which the manifest summarises rather
    # than stores; re-read them from the label files as we go.
    scenes = d[["split", "scene"]].drop_duplicates()
    if a.limit and len(scenes) > a.limit:
        # sample rather than take a prefix: scenes are ordered by disaster, so a
        # prefix is one disaster's worth of buildings, not a cross-section
        scenes = scenes.sample(a.limit, random_state=a.seed)
    scenes = scenes.values.tolist()
    print(f"{len(scenes):,} scenes, {len(d):,} buildings >= {a.min_side:.0f} px",
          flush=True)

    meta_path = a.out / "crop_meta.parquet"
    done = set()
    rows = []
    if meta_path.exists():
        prev = pd.read_parquet(meta_path)
        rows = prev.to_dict("records")
        done = set(prev.scene.unique())
        print(f"  resuming: {len(done):,} scenes already extracted")

    for i, (split, scene) in enumerate(scenes, 1):
        if scene in done:
            continue
        lab = a.labels / f"{split}__labels__{scene}_post_disaster.json"
        if not lab.exists():
            continue
        try:
            pre_p, post_p = scene_paths(split, scene)
            pre = cv2.imread(hf_hub_download(REPO, pre_p, repo_type="dataset"))
            post = cv2.imread(hf_hub_download(REPO, post_p, repo_type="dataset"))
        except Exception as e:
            print(f"  {scene}: fetch failed ({type(e).__name__})")
            continue
        if pre is None or post is None or pre.shape != post.shape:
            print(f"  {scene}: image pair unusable")
            continue
        H, W = post.shape[:2]

        js = json.loads(lab.read_text())
        sub = (a.out / scene)
        sub.mkdir(exist_ok=True)
        n = 0
        for f in js.get("features", {}).get("xy", []):
            pr = f.get("properties", {})
            if pr.get("feature_type") != "building":
                continue
            m = re.search(r"\(\((.*?)\)\)", f.get("wkt", ""))
            if not m:
                continue
            pts = np.array([[float(x), float(y)] for x, y in
                            (p.strip().split() for p in m.group(1).split(","))])
            side = float(np.sqrt(0.5 * abs(
                np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))))
            if side < a.min_side:
                continue
            box = crop_box(pts[:, 0].mean(), pts[:, 1].mean(), side,
                           a.crop_scale, W, H)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            uid = pr["uid"]
            cv2.imwrite(str(sub / f"{uid}_pre.png"), pre[y1:y2, x1:x2])
            cv2.imwrite(str(sub / f"{uid}_post.png"), post[y1:y2, x1:x2])
            rows.append({"uid": uid, "scene": scene, "split": split,
                         "damage": pr.get("subtype"), "px_side": side,
                         "crop_w": x2 - x1, "crop_h": y2 - y1,
                         "pre": f"{scene}/{uid}_pre.png",
                         "post": f"{scene}/{uid}_post.png"})
            n += 1
        if i % 25 == 0 or n == 0:
            print(f"  ({i}/{len(scenes)}) {scene}: {n} crops", flush=True)
        if i % 50 == 0:
            pd.DataFrame(rows).to_parquet(meta_path, index=False)

    m = pd.DataFrame(rows)
    m.to_parquet(meta_path, index=False)
    size = sum(p.stat().st_size for p in a.out.rglob("*.png")) / 1e6
    print(f"\n{len(m):,} paired crops from {m.scene.nunique():,} scenes, "
          f"{size:.0f} MB")
    print(m.damage.value_counts().to_string())
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
