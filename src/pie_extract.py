"""Step B -- rolling crop extraction, one video set at a time.

The storage discipline: we never need more than ONE PIE video set on disk.
This script reads the Step A manifest, verifies the set's videos are present,
decodes each video once (sequentially -- seeking on these 10-minute 1080p clips
is far slower than grabbing), writes JPEG crops plus a metadata parquet, runs
integrity checks, and only then prints that the set is safe to delete.

It never deletes anything itself.

Usage:
    python src/pie_extract.py --set set01 --clips /path/to/PIE_clips
    python src/pie_extract.py --set set01 --clips ... --verify-only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- frame set

def frames_for_set(manifest_dir: Path, set_id: str, pre: int, post: int,
                   from_track_start: bool) -> pd.DataFrame:
    """Rows of the manifest we need pixels for, for one set.

    from_track_start=True  -> the full track from its first annotated frame
                              (the conservative default; largest footprint)
    from_track_start=False -> only the window humans actually saw
                              (exp_start_point .. critical_point), which is what
                              intention_prob is a judgement about
    Either way we keep `post` frames past critical_point for resolvability, and
    `pre` frames of lead-in.
    """
    f = pd.read_parquet(manifest_dir / "frames.parquet")
    f = f[f.set_id == set_id].copy()
    if from_track_start:
        keep = f.frames_to_critical <= post
    else:
        span = f.critical_point - f.exp_start_point
        keep = (f.frames_to_critical >= -(span + pre)) & (f.frames_to_critical <= post)
    return f[keep].reset_index(drop=True)


# ------------------------------------------------------------ degradation

def degradation_stats(crop: np.ndarray) -> dict:
    """Cheap per-crop confounder controls, computed while we have the pixels.

    These are the covariates Experiment 1 regresses out: contested cases in any
    driving dataset correlate with small/blurry/occluded crops, and we need to
    show the tornness signal survives that.
    """
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "crop_h": crop.shape[0],
        "crop_w": crop.shape[1],
        "blur_var_laplacian": float(cv2.Laplacian(g, cv2.CV_64F).var()),
        "mean_luma": float(g.mean()),
        "contrast_std": float(g.std()),
    }


def crop_box(frame: np.ndarray, row, scale: float) -> tuple[np.ndarray, int]:
    """Crop `scale` x the bbox, clamped to the image. Returns (crop, truncated)."""
    H, W = frame.shape[:2]
    cx, cy = (row.xtl + row.xbr) / 2, (row.ytl + row.ybr) / 2
    hw, hh = row.bbox_w * scale / 2, row.bbox_h * scale / 2
    x1, y1 = int(round(cx - hw)), int(round(cy - hh))
    x2, y2 = int(round(cx + hw)), int(round(cy + hh))
    truncated = int(x1 < 0 or y1 < 0 or x2 > W or y2 > H)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None, truncated
    return frame[y1:y2, x1:x2], truncated


# -------------------------------------------------------------- extraction

def extract_video(video_path: Path, rows: pd.DataFrame, out_dir: Path,
                  scale: float, jpeg_q: int) -> list[dict]:
    """Decode one video sequentially, writing every crop the manifest asks for."""
    wanted = rows.groupby("frame")
    frame_ids = sorted(wanted.groups.keys())
    if not frame_ids:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta, pos, target_i = [], 0, 0
    last_wanted = frame_ids[-1]

    while target_i < len(frame_ids):
        target = frame_ids[target_i]
        # grab() decodes nothing but advances -- far cheaper than retrieve()
        while pos < target:
            if not cap.grab():
                break
            pos += 1
        ok, frame = cap.read()
        if not ok:
            break
        cur, pos = pos, pos + 1
        if cur != target:
            target_i += 1
            continue

        for row in wanted.get_group(target).itertuples():
            crop, truncated = crop_box(frame, row, scale)
            if crop is None:
                continue
            name = f"{row.ped_id}_{row.frame:06d}.jpg"
            cv2.imwrite(str(out_dir / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            meta.append({
                "set_id": row.set_id, "video_id": row.video_id,
                "ped_id": row.ped_id, "frame": row.frame,
                "crop_path": f"{row.set_id}/{row.video_id}/{name}",
                "xtl": row.xtl, "ytl": row.ytl, "xbr": row.xbr, "ybr": row.ybr,
                "bbox_w": row.bbox_w, "bbox_h": row.bbox_h,
                "occluded_flag": row.occluded_flag, "occlusion": row.occlusion,
                "action": row.action, "look": row.look, "cross": row.cross,
                "gesture": row.gesture,
                "frames_to_critical": row.frames_to_critical,
                "in_exp_window": row.in_exp_window,
                "truncated": truncated,
                **degradation_stats(crop),
            })
        target_i += 1
        if cur >= last_wanted:
            break

    cap.release()
    return meta


# ------------------------------------------------------------------ checks

def verify(set_id: str, expected: pd.DataFrame, meta: pd.DataFrame,
           crops_root: Path, n_spot: int = 20) -> tuple[bool, list[str]]:
    msgs, ok = [], True

    n_exp, n_got = len(expected), len(meta)
    msgs.append(f"crops expected {n_exp:,} / written {n_got:,} "
                f"({100 * n_got / max(n_exp, 1):.2f}%)")
    if n_got < n_exp:
        missing = n_exp - n_got
        # a handful of boxes fall entirely outside frame; >1% means a real bug
        if missing / max(n_exp, 1) > 0.01:
            ok = False
            msgs.append(f"FAIL: {missing:,} crops missing (>1%)")
        else:
            msgs.append(f"note: {missing:,} crops skipped (fully out of frame)")

    got_peds = set(meta.ped_id.unique())
    exp_peds = set(expected.ped_id.unique())
    if got_peds != exp_peds:
        ok = False
        msgs.append(f"FAIL: {len(exp_peds - got_peds)} pedestrians have no crops")

    sample = meta.sample(min(n_spot, len(meta)), random_state=0)
    bad = [r.crop_path for r in sample.itertuples()
           if not (crops_root / r.crop_path).exists()
           or cv2.imread(str(crops_root / r.crop_path)) is None]
    if bad:
        ok = False
        msgs.append(f"FAIL: {len(bad)} sampled crops unreadable, e.g. {bad[:3]}")
    else:
        msgs.append(f"spot-check: {len(sample)} random crops readable")

    # contact sheet so the boxes can be eyeballed, not just counted
    tiles = []
    for r in sample.itertuples():
        im = cv2.imread(str(crops_root / r.crop_path))
        if im is None:
            continue
        im = cv2.resize(im, (128, 256))
        cv2.putText(im, f"{r.action[:4]}/{r.look[:4]}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        # frames_to_critical is float (critical_point parses as float, and is
        # NaN for tracks without attributes), so coerce before formatting
        cv2.putText(im, f"t{int(r.frames_to_critical):+d}", (3, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(im)
    if tiles:
        rows_ = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
        w = max(r.shape[1] for r in rows_)
        rows_ = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows_]
        sheet = crops_root / f"_spotcheck_{set_id}.jpg"
        cv2.imwrite(str(sheet), np.vstack(rows_))
        msgs.append(f"contact sheet: {sheet}")

    return ok, msgs


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="e.g. set01")
    ap.add_argument("--clips", type=Path, required=True,
                    help="dir containing <set>/video_XXXX.mp4")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--jpeg-q", type=int, default=90)
    ap.add_argument("--pre", type=int, default=0,
                    help="lead-in frames before exp_start_point")
    ap.add_argument("--post", type=int, default=45,
                    help="frames kept past critical_point")
    ap.add_argument("--full-track", action="store_true",
                    help="extract every annotated frame from track start "
                         "(~10.5 GB all sets) instead of the default human-"
                         "experiment window exp_start..critical+post (~3.1 GB). "
                         "The default matches the frames annotators actually "
                         "saw, so intention_prob judges the same evidence the "
                         "model sees.")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    set_id = a.set
    expected = frames_for_set(a.manifest, set_id, a.pre, a.post,
                              from_track_start=a.full_track)
    if expected.empty:
        raise SystemExit(f"no manifest rows for {set_id}")

    a.crops.mkdir(parents=True, exist_ok=True)
    meta_path = a.crops / f"_meta_{set_id}.parquet"
    state_path = a.crops / f"_state_{set_id}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": []}

    videos = sorted(expected.video_id.unique())
    print(f"[{set_id}] {len(videos)} videos, {len(expected):,} crops expected")

    if not a.verify_only:
        src = a.clips / set_id
        missing = [v for v in videos if not (src / f"{v}.mp4").exists()]
        if missing:
            raise SystemExit(
                f"missing videos in {src}: {missing}\n"
                f"download {set_id} first, then re-run")

        all_meta = []
        if meta_path.exists():
            all_meta.append(pd.read_parquet(meta_path))

        for i, vid in enumerate(videos, 1):
            if vid in state["done"]:
                print(f"  ({i}/{len(videos)}) {vid} already done, skipping")
                continue
            rows = expected[expected.video_id == vid]
            print(f"  ({i}/{len(videos)}) {vid}: {len(rows):,} crops ...", flush=True)
            m = extract_video(src / f"{vid}.mp4", rows,
                              a.crops / set_id / vid, a.scale, a.jpeg_q)
            all_meta.append(pd.DataFrame(m))
            state["done"].append(vid)
            # checkpoint after every video so an interrupt costs at most one clip
            pd.concat(all_meta, ignore_index=True).to_parquet(meta_path, index=False)
            state_path.write_text(json.dumps(state))
            print(f"      wrote {len(m):,} crops")

    meta = pd.read_parquet(meta_path)
    ok, msgs = verify(set_id, expected, meta, a.crops)
    print("\n--- integrity ---")
    for m in msgs:
        print(" ", m)

    size_gb = sum(p.stat().st_size for p in (a.crops / set_id).rglob("*.jpg")) / 1e9
    print(f"  crop store for {set_id}: {size_gb:.2f} GB")
    print("\n" + ("=" * 60))
    if ok:
        print(f"SAFE TO DELETE VIDEOS FOR {set_id}  ({a.clips / set_id})")
        print("Delete them yourself -- this script never does.")
    else:
        print(f"DO NOT DELETE {set_id} -- integrity checks failed (see above).")
    print("=" * 60)
    # exit code lets a wrapper gate video deletion on the checks passing
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
