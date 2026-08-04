"""Step C -- PyTorch datasets over the PIE crop store.

Two views of the same data:

  PIECropDataset  single crops -> for training the ProtoPNet intent classifier
  PIETrackDataset per-pedestrian frame sequences -> for the resolvability
                  experiment, where we need tornness as a function of time

Label design note (see RESULTS.md): the binary target is `intent_binary`,
defined as `crossing != -1`, NOT `crossing == 1`. PIE separates intention from
action on purpose: 855 pedestrians who never crossed still carry a mean
intention_prob of 0.838 because conditions blocked them. Training on the action
label would put clear intenders in the "won't cross" prototype set and make
every dual-match a label artefact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
SPLIT_BY_SET = {
    "set01": "train", "set02": "train", "set04": "train",
    "set05": "val", "set06": "val", "set03": "test",
}


def load_store(crops: Path = ROOT / "data/pie_crops",
               manifest: Path = ROOT / "data/pie_manifest") -> pd.DataFrame:
    """Join every extracted set's metadata with per-pedestrian labels."""
    metas = sorted(crops.glob("_meta_set*.parquet"))
    if not metas:
        raise FileNotFoundError(
            f"no extracted sets in {crops}; run src/pie_extract.py first")
    df = pd.concat([pd.read_parquet(p) for p in metas], ignore_index=True)

    peds = pd.read_parquet(manifest / "peds.parquet")
    peds["intent_binary"] = (peds.crossing != -1).astype(int)
    cols = ["ped_id", "intention_prob", "intent_binary", "crossing",
            "crossing_point", "critical_point", "exp_start_point",
            "age", "gender", "num_lanes", "signalized", "traffic_direction",
            "intersection", "split"]
    df = df.merge(peds[cols], on="ped_id", how="inner")

    # human disagreement magnitude: 0 = unanimous, 1 = maximally split
    df["human_disagreement"] = 1.0 - 2.0 * (df.intention_prob - 0.5).abs()
    return df


def kfold_assign(peds: pd.DataFrame, n_folds: int = 5, group: str = "video",
                 seed: int = 0) -> pd.DataFrame:
    """Assign each pedestrian a fold, grouped so no scene spans the split.

    PIE's own `_get_kfold_pedestrian_ids` uses a plain shuffled KFold over
    pedestrian ids, which puts pedestrians from the *same video* in both train
    and test. For a prototype network that is a live leak: a projected prototype
    is a real training patch, and it can come from the test fold's own scene,
    same lighting, same crowd. We group instead.

    group='video' -- one 10-min clip never spans folds (default)
    group='set'   -- whole recording sessions never span folds (strictest;
                     only 6 groups, so fold balance is coarse)

    Strata keep both the intent balance and the *contested* cases (the ones the
    deferral experiment actually runs on) spread evenly across folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    p = peds.copy()
    p["intent_binary"] = (p.crossing != -1).astype(int)
    contested = p.intention_prob.between(0.3, 0.7).astype(int)
    stratum = p.intent_binary.astype(str) + "_" + contested.astype(str)
    groups = p.set_id if group == "set" else p.set_id + "/" + p.video_id

    p["fold"] = -1
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(skf.split(p, stratum, groups)):
        p.iloc[test_idx, p.columns.get_loc("fold")] = k
    return p[["ped_id", "fold", "intent_binary"]]


def filter_split(df: pd.DataFrame, split: str | None,
                 in_window_only: bool = False,
                 min_bbox_h: float = 0.0) -> pd.DataFrame:
    out = df
    if split is not None:
        out = out[out.split == split]
    if in_window_only:
        out = out[out.in_exp_window]
    if min_bbox_h > 0:
        out = out[out.bbox_h >= min_bbox_h]
    return out.reset_index(drop=True)


class CenterFraction:
    """Center-crop to a fraction of the stored crop.

    Crops were extracted at 2x the bbox, so roughly 75% of each image's area is
    background. With a 7x7 feature grid there are far more background patches
    than pedestrian patches, and road texture is an easier, more stable signal
    than posture -- prototypes localise to the ground plane instead of the
    person. Tightening recovers an effective crop of `target x bbox` without
    re-extracting anything from video.
    """

    def __init__(self, target_mult: float, stored_mult: float = 2.0):
        self.frac = max(min(target_mult / stored_mult, 1.0), 0.05)

    def __call__(self, img):
        import torchvision.transforms.functional as TF
        w, h = img.size
        return TF.center_crop(img, [max(int(round(h * self.frac)), 8),
                                    max(int(round(w * self.frac)), 8)])

    def __repr__(self):
        return f"CenterFraction(frac={self.frac:.3f})"


def crop_transform(crop_scale: float, stored_mult: float = 2.0):
    """None when no tightening is requested, so existing behaviour is unchanged."""
    if crop_scale is None or crop_scale >= stored_mult:
        return None
    return CenterFraction(crop_scale, stored_mult)


META_COLS = ["ped_id", "frame", "frames_to_critical", "in_exp_window",
             "bbox_h", "bbox_w", "occluded_flag", "truncated",
             "blur_var_laplacian", "mean_luma", "contrast_std",
             "action", "look", "cross", "gesture",
             "intention_prob", "human_disagreement", "crossing"]


class PIECropDataset(Dataset):
    """One crop per item. Yields (image, intent_binary, meta)."""

    def __init__(self, df: pd.DataFrame, crops_root: Path = ROOT / "data/pie_crops",
                 transform=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(crops_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r.crop_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        meta = {c: r[c] for c in META_COLS if c in r}
        return img, int(r.intent_binary), meta


class PIETrackDataset(Dataset):
    """One pedestrian per item: an ordered frame sequence.

    Yields (images[T], intent_binary, intention_prob, meta) where meta carries
    per-frame `frames_to_critical` so resolvability curves can be aligned on the
    critical point across pedestrians of different track lengths.
    """

    def __init__(self, df: pd.DataFrame, crops_root: Path = ROOT / "data/pie_crops",
                 transform=None, stride: int = 1, max_frames: int | None = None):
        self.root = Path(crops_root)
        self.transform = transform
        d = df.sort_values(["ped_id", "frame"])
        self.groups = {p: g.reset_index(drop=True) for p, g in d.groupby("ped_id")}
        self.ped_ids = sorted(self.groups)
        self.stride = stride
        self.max_frames = max_frames

    def __len__(self):
        return len(self.ped_ids)

    def __getitem__(self, i):
        pid = self.ped_ids[i]
        g = self.groups[pid].iloc[:: self.stride]
        if self.max_frames is not None:
            g = g.iloc[-self.max_frames:]
        imgs = []
        for p in g.crop_path:
            im = Image.open(self.root / p).convert("RGB")
            imgs.append(self.transform(im) if self.transform is not None else im)
        if self.transform is not None:
            imgs = torch.stack(imgs)
        first = g.iloc[0]
        meta = {
            "ped_id": pid,
            "frames_to_critical": g.frames_to_critical.to_numpy(),
            "frame": g.frame.to_numpy(),
            "in_exp_window": g.in_exp_window.to_numpy(),
            "bbox_h": g.bbox_h.to_numpy(),
            "blur_var_laplacian": g.blur_var_laplacian.to_numpy(),
            "occluded_flag": g.occluded_flag.to_numpy(),
            "action": g.action.tolist(),
            "look": g.look.tolist(),
            "crossing": int(first.crossing),
            "human_disagreement": float(first.human_disagreement),
        }
        return imgs, int(first.intent_binary), float(first.intention_prob), meta


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """PIE intent is ~75/25 imbalanced; ProtoPNet's CE needs weighting."""
    n = df.groupby("intent_binary").size()
    w = len(df) / (2.0 * n)
    return torch.tensor([w.get(0, 1.0), w.get(1, 1.0)], dtype=torch.float32)


if __name__ == "__main__":
    df = load_store()
    print(f"crops: {len(df):,}  pedestrians: {df.ped_id.nunique():,}")
    print(df.groupby("split").agg(crops=("crop_path", "size"),
                                  peds=("ped_id", "nunique")))
    print("\nintent_binary balance:", df.groupby("intent_binary").ped_id.nunique().to_dict())
    print("class weights:", class_weights(df).tolist())
    contested = df[df.human_disagreement > 0.8].ped_id.nunique()
    print(f"contested pedestrians (intention_prob in 0.4-0.6): {contested}")
