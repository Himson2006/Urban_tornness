"""Step A -- build the PIE manifest from annotations alone (no video required).

Parses the three PIE annotation sources into two tables:

  peds.parquet    one row per pedestrian track (attributes, intention_prob,
                  outcome, experiment window)
  frames.parquet  one row per (pedestrian, frame) with bbox + behaviour tags

Everything downstream -- crop extraction, tornness features, resolvability
curves -- is driven off these two files, so the manifest fixes exactly which
pixels we will ever need to touch.

Usage:
    python src/pie_manifest.py [--raw data/pie_raw] [--out data/pie_manifest]
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

# Official PIE split (pie_data.py::_get_image_set_ids).
SPLIT_BY_SET = {
    "set01": "train", "set02": "train", "set04": "train",
    "set05": "val", "set06": "val",
    "set03": "test",
}

# Per-box textual tags we keep. 'id' is pulled out separately as the track key.
BOX_TAGS = ("gesture", "action", "cross", "look", "occlusion")

# Attribute names on <pedestrian> in annotations_attributes/.
PED_ATTR_NUM = ("num_lanes", "crossing", "exp_start_point", "critical_point",
                "intention_prob", "crossing_point")
PED_ATTR_STR = ("age", "gender", "signalized", "traffic_direction", "intersection")


def parse_spatial(xml_path: Path) -> list[dict]:
    """Extract every annotated pedestrian bounding box from one video's XML."""
    root = ET.parse(xml_path).getroot()
    rows = []
    for track in root.findall(".//track"):
        if track.get("label") != "pedestrian":
            continue
        for box in track.findall("box"):
            tags = {a.get("name"): (a.text or "") for a in box.findall("attribute")}
            ped_id = tags.get("id")
            if ped_id is None:
                continue
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            rows.append({
                "ped_id": ped_id,
                "frame": int(box.get("frame")),
                "xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr,
                "bbox_w": xbr - xtl,
                "bbox_h": ybr - ytl,
                # box-level flags: `occluded` is binary here, the `occlusion`
                # text tag carries PIE's 3-level none/part/full scale
                "occluded_flag": int(box.get("occluded", 0)),
                "outside": int(box.get("outside", 0)),
                "keyframe": int(box.get("keyframe", 1)),
                **{t: tags.get(t, "") for t in BOX_TAGS},
            })
    return rows


def parse_attributes(xml_path: Path) -> list[dict]:
    """Extract per-track pedestrian attributes (intention_prob lives here)."""
    root = ET.parse(xml_path).getroot()
    rows = []
    for ped in root.findall(".//pedestrian"):
        row = {"ped_id": ped.get("id")}
        for k in PED_ATTR_STR:
            row[k] = ped.get(k)
        for k in PED_ATTR_NUM:
            v = ped.get(k)
            row[k] = float(v) if v not in (None, "", "n/a") else np.nan
        rows.append(row)
    return rows


def build(raw: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ann_dir, attr_dir = raw / "annotations", raw / "annotations_attributes"

    frame_rows, ped_rows = [], []
    for set_dir in sorted(ann_dir.glob("set*")):
        set_id = set_dir.name
        for xml_path in sorted(set_dir.glob("video_*_annt.xml")):
            video_id = xml_path.name.replace("_annt.xml", "")
            for r in parse_spatial(xml_path):
                frame_rows.append({"set_id": set_id, "video_id": video_id, **r})

            attr_path = attr_dir / set_id / f"{video_id}_attributes.xml"
            if attr_path.exists():
                for r in parse_attributes(attr_path):
                    ped_rows.append({"set_id": set_id, "video_id": video_id, **r})

    frames = pd.DataFrame(frame_rows)
    peds = pd.DataFrame(ped_rows)

    # Track geometry, derived from the frame table.
    g = frames.groupby("ped_id")
    track = pd.DataFrame({
        "track_start": g["frame"].min(),
        "track_end": g["frame"].max(),
        "n_frames": g["frame"].count(),
        "median_bbox_h": g["bbox_h"].median(),
        "median_bbox_w": g["bbox_w"].median(),
        "max_bbox_h": g["bbox_h"].max(),
        "frac_occluded": g["occluded_flag"].mean(),
    }).reset_index()

    peds = peds.merge(track, on="ped_id", how="left")
    peds["split"] = peds["set_id"].map(SPLIT_BY_SET)
    frames["split"] = frames["set_id"].map(SPLIT_BY_SET)

    # How much of the human-experiment window and post-critical tail we actually
    # have boxes for -- this is what bounds the resolvability experiment.
    peds["exp_window_len"] = peds["critical_point"] - peds["exp_start_point"]
    peds["frames_after_critical"] = peds["track_end"] - peds["critical_point"]
    peds["frames_before_critical"] = peds["critical_point"] - peds["track_start"]

    # Mark which frames sit inside the window humans actually saw.
    exp = peds.set_index("ped_id")[["exp_start_point", "critical_point"]]
    frames = frames.join(exp, on="ped_id")
    frames["in_exp_window"] = (
        (frames["frame"] >= frames["exp_start_point"])
        & (frames["frame"] <= frames["critical_point"])
    )
    frames["frames_to_critical"] = frames["frame"] - frames["critical_point"]
    frames["has_intent_label"] = frames["ped_id"].isin(
        peds.loc[peds["intention_prob"].notna(), "ped_id"]
    )

    out.mkdir(parents=True, exist_ok=True)
    peds.to_parquet(out / "peds.parquet", index=False)
    frames.to_parquet(out / "frames.parquet", index=False)
    return peds, frames


# ---------------------------------------------------------------- reporting

def crop_storage_estimate(frames: pd.DataFrame, peds: pd.DataFrame, scale=2.0) -> str:
    """Bytes on disk if we crop every box at `scale` x bbox and save as JPEG q90.

    ~0.35 bytes/pixel is a conservative JPEG q90 rate for photographic crops.
    """
    labelled = frames[frames["has_intent_label"]]
    lines = []
    for name, df in (("all annotated ped boxes", frames),
                     ("boxes on intent-labelled peds", labelled)):
        px = ((df["bbox_w"] * scale) * (df["bbox_h"] * scale)).sum()
        lines.append(f"  {name:32s} {len(df):>8,d} crops  ~{px * 0.35 / 1e9:6.2f} GB")
    return "\n".join(lines)


def report(peds: pd.DataFrame, frames: pd.DataFrame) -> str:
    L = []
    A = L.append
    lab = peds[peds["intention_prob"].notna()]

    A("## Step A -- PIE manifest (annotations only, no video downloaded)\n")
    A(f"- pedestrian tracks with attributes: **{len(peds):,}**")
    A(f"- tracks with `intention_prob`: **{len(lab):,}**")
    A(f"- annotated pedestrian boxes: **{len(frames):,}** "
      f"({len(frames[frames['has_intent_label']]):,} on intent-labelled tracks)\n")

    A("### Per set\n")
    A("| set | split | videos | ped tracks | w/ intent | boxes | boxes (intent peds) |")
    A("|---|---|---|---|---|---|---|")
    for s, d in peds.groupby("set_id"):
        f = frames[frames["set_id"] == s]
        A(f"| {s} | {SPLIT_BY_SET[s]} | {d['video_id'].nunique()} | {len(d)} | "
          f"{d['intention_prob'].notna().sum()} | {len(f):,} | "
          f"{int(f['has_intent_label'].sum()):,} |")

    A("\n### intention_prob distribution (the torn-case population)\n")
    ip = lab["intention_prob"]
    A(f"- mean {ip.mean():.3f}, median {ip.median():.3f}, "
      f"n unique values {ip.nunique()}")
    denom = np.unique(np.round(ip.values * 60))
    A(f"- values are multiples of 1/60 -> consistent with ~15 annotators on a "
      f"5-point scale ({len(denom)} distinct levels observed)")
    A("")
    A("| bin | n | % |")
    A("|---|---|---|")
    bins = np.arange(0, 1.0001, 0.1)
    h, _ = np.histogram(ip, bins=bins)
    for i, c in enumerate(h):
        A(f"| {bins[i]:.1f}-{bins[i+1]:.1f} | {c} | {100*c/len(ip):.1f}% |")

    A("\n**Contested population** (how many pedestrians humans were split on):\n")
    A("| band | n | % of labelled |")
    A("|---|---|---|")
    for lo, hi in [(0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8)]:
        n = int(((ip >= lo) & (ip <= hi)).sum())
        A(f"| {lo:.2f}-{hi:.2f} | {n} | {100*n/len(ip):.1f}% |")

    A("\n### Behavioural outcome vs. intent\n")
    A("| crossing | meaning | n | mean intention_prob |")
    A("|---|---|---|---|")
    meaning = {1.0: "crossed in ego path", 0.0: "did not cross",
               -1.0: "irrelevant (near road, not intending)"}
    for c, d in lab.groupby("crossing"):
        A(f"| {c:+.0f} | {meaning.get(c,'?')} | {len(d)} | "
          f"{d['intention_prob'].mean():.3f} |")

    A("\n### Track geometry (bounds the resolvability experiment)\n")
    for col in ("n_frames", "exp_window_len", "frames_before_critical",
                "frames_after_critical", "median_bbox_h"):
        q = lab[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        A(f"- `{col}`: p5={q[0.05]:.0f} p25={q[0.25]:.0f} med={q[0.5]:.0f} "
          f"p75={q[0.75]:.0f} p95={q[0.95]:.0f}")
    n_tail = int((lab["frames_after_critical"] >= 45).sum())
    A(f"- tracks with >=45 frames after critical_point: **{n_tail}** "
      f"({100*n_tail/len(lab):.1f}%)")

    A("\n### Per-frame behaviour tags (free cue labels for dual-match cases)\n")
    lf = frames[frames["has_intent_label"]]
    for t in ("action", "look", "cross", "gesture"):
        vc = lf[t].value_counts()
        A(f"- `{t}`: " + ", ".join(f"{k}={v:,}" for k, v in vc.head(6).items()))

    A("\n### Crop storage estimate (2x bbox, JPEG q90)\n```")
    A(crop_storage_estimate(frames, peds))
    A("```")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--raw", type=Path, default=root / "data/pie_raw")
    ap.add_argument("--out", type=Path, default=root / "data/pie_manifest")
    ap.add_argument("--results", type=Path, default=root / "RESULTS.md")
    a = ap.parse_args()

    peds, frames = build(a.raw, a.out)
    txt = report(peds, frames)
    print(txt)
    a.results.write_text(
        "# RESULTS\n\nTornness project -- running log.\n\n" + txt + "\n"
    )
    print(f"\nwrote {a.out}/peds.parquet, {a.out}/frames.parquet, {a.results}")


if __name__ == "__main__":
    main()
