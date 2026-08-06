"""Per-scene radiometric alignment between the pre and post captures.

Looking at the first contact sheet turned up a confound that would have been
invisible in any accuracy number. The pre and post images of a scene come from
different satellite passes -- different sun angle, atmosphere, sometimes
different sensor and season -- so the whole frame shifts in tone between them.
On the major-damage examples the pre crop was blue-cast and the post crop nearly
white, across the entire crop including the grass. The no-damage examples showed
large differences too.

A six-channel model handed that would learn "the post image is brighter" and
score well, because capture conditions correlate with scene, and scene
correlates with disaster and therefore with damage. It would be the same class
of failure as the pedestrian prototypes landing on the curb: a real signal, and
the wrong one.

The fix is to estimate the tone shift from the *whole scene* and remove it. A
building is a small fraction of a 1024x1024 frame, so scene-wide mean and
standard deviation are set by ground, vegetation and roofs at large -- what
changed at one building barely moves them. Matching pre to post on those
statistics removes the capture difference while leaving local change intact.

Statistics are stored per scene rather than baked into the crops, so the
unaligned input stays available: `--no-align` at training time measures how much
of the model's performance was the artifact.

Usage:
    python xbd/radiometry.py            # over scenes already in the HF cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = "aryananand/xBD"


def scene_stats(pre: np.ndarray, post: np.ndarray) -> dict:
    """Per-channel mean/sd of each capture, over the whole frame."""
    out = {}
    for nm, im in (("pre", pre), ("post", post)):
        x = im.reshape(-1, 3).astype(np.float32)
        for c in range(3):
            out[f"{nm}_m{c}"] = float(x[:, c].mean())
            out[f"{nm}_s{c}"] = float(x[:, c].std())
    return out


def alignment(row) -> tuple[np.ndarray, np.ndarray]:
    """Gain and offset mapping the pre capture onto the post capture.

    pre_aligned = pre * gain + offset, per channel.
    """
    g = np.array([row[f"post_s{c}"] / max(row[f"pre_s{c}"], 1e-3)
                  for c in range(3)], np.float32)
    o = np.array([row[f"post_m{c}"] - g[c] * row[f"pre_m{c}"]
                  for c in range(3)], np.float32)
    return g, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "xbd/data/scene_radiometry.parquet")
    ap.add_argument("--only-cached", action="store_true", default=True,
                    help="skip scenes whose imagery is not already downloaded")
    a = ap.parse_args()

    import cv2
    from huggingface_hub import hf_hub_download

    m = pd.read_parquet(a.crops / "crop_meta.parquet")
    scenes = m[["split", "scene"]].drop_duplicates().values.tolist()
    rows = []
    if a.out.exists():
        prev = pd.read_parquet(a.out)
        rows = prev.to_dict("records")
        done = set(prev.scene)
        scenes = [s for s in scenes if s[1] not in done]
        print(f"resuming: {len(done):,} scenes done, {len(scenes):,} to go")

    for i, (split, scene) in enumerate(scenes, 1):
        try:
            kw = dict(repo_type="dataset",
                      **({"local_files_only": True} if a.only_cached else {}))
            pre = cv2.imread(hf_hub_download(
                REPO, f"{split}/images/{scene}_pre_disaster.png", **kw))
            post = cv2.imread(hf_hub_download(
                REPO, f"{split}/images/{scene}_post_disaster.png", **kw))
        except Exception:
            continue
        if pre is None or post is None:
            continue
        rows.append({"scene": scene, **scene_stats(pre, post)})
        if i % 100 == 0:
            print(f"  ({i}/{len(scenes)})", flush=True)
            pd.DataFrame(rows).to_parquet(a.out, index=False)

    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no scenes measured -- is the imagery cached?")
    d.to_parquet(a.out, index=False)

    # how large is the confound we are removing?
    dm = np.mean([d[f"post_m{c}"] - d[f"pre_m{c}"] for c in range(3)], axis=0)
    ds = np.mean([d[f"post_s{c}"] / d[f"pre_s{c}"].clip(1e-3)
                  for c in range(3)], axis=0)
    print(f"\n{len(d):,} scenes measured")
    print(f"  post minus pre mean brightness: median {np.median(dm):+.1f}, "
          f"p10 {np.percentile(dm, 10):+.1f}, p90 {np.percentile(dm, 90):+.1f}"
          f"  (0-255 scale)")
    print(f"  post/pre contrast ratio:        median {np.median(ds):.2f}, "
          f"p10 {np.percentile(ds, 10):.2f}, p90 {np.percentile(ds, 90):.2f}")
    print("  a scene-constant shift of this size is visible to the model and")
    print("  carries no information about any individual building")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
