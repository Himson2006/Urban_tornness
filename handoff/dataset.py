"""Sidewalk label crops as a torch dataset.

The task is the one Project Sidewalk's own reviewers perform: shown a crop
centred on a crowdsourced accessibility label, decide whether the label is
*correct*. That is the hand-off this paper is about -- an inspector deciding
whether to trust a volunteer -- so the model is trained on exactly the decision
whose uncertainty is later compared against reviewer disagreement.

Two design points are load-bearing.

**Train on consensus, evaluate on contest.** A label whose reviewers were split
has no trustworthy target, and training on it teaches the model the noise it is
supposed to be uncertain about. `consensus_only` keeps training to labels the
reviewers settled, holding the contested ones back as the evaluation set. This
is the LIDC arrangement -- train on the confident nodules, hold out the
indeterminate ones -- and it is what keeps Role B from being circular.

**Splits are grouped by panorama, never by label.** Two labels on one panorama
share lighting, camera and often the same stretch of pavement. The released
splits do not group, and 24.4% of crops sit on a panorama that straddles two of
them; `handoff/data.py` regenerates them. The group column travels with the
manifest so the trainer cannot forget it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# A label is "contested" when reviewers took opposing positions, and "unclear"
# when they individually declined to judge. The thresholds match
# sidewalk/manifest.py so the two studies speak the same language.
CONTESTED = 0.5      # split_adj above this
UNCLEAR = 0.15       # unsure_adj above this


class CropDataset(Dataset):
    """Returns (image, label, uid). ProtoPNet's loops want the first two."""

    def __init__(self, meta: pd.DataFrame, root: Path, size: int = 224,
                 augment: bool = False):
        self.meta = meta.reset_index(drop=True)
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.mean = torch.tensor(MEAN)[:, None, None]
        self.std = torch.tensor(STD)[:, None, None]

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i):
        r = self.meta.iloc[i]
        im = Image.open(self.root / r.path).convert("RGB")

        if self.augment:
            # Crops are already centred on the label, so anything that shifts
            # the frame far risks pushing the label out of it. Flips and mild
            # colour jitter only; no random-resized-crop.
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            im = im.resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
            x = x.permute(2, 0, 1)
            x = (x * np.random.uniform(0.9, 1.1)).clamp(0, 1)
        else:
            im = im.resize((self.size, self.size), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
            x = x.permute(2, 0, 1)

        x = (x - self.mean) / self.std
        return x, int(r.y), str(r.uid)


class TwoTuple(Dataset):
    """ProtoPNet's loops unpack (image, label)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return x, y


class UnNormalized(Dataset):
    """Push needs images in [0,1]; rescale rather than reload from disk."""

    def __init__(self, ds):
        self.ds = ds
        self.m = torch.tensor(MEAN)[:, None, None]
        self.s = torch.tensor(STD)[:, None, None]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return (x * self.s + self.m).clamp(0, 1), y


def load_manifest(path: Path, consensus_only: bool = True) -> pd.DataFrame:
    """Manifest with the contested/unclear flags attached.

    `consensus_only` affects the TRAIN split alone. Val and test keep every
    label, because the contested ones are the evaluation set Role B needs.
    """
    m = pd.read_parquet(path)
    m = m[m.split.notna()].copy()
    m["contested"] = m.split_adj > CONTESTED
    m["unclear"] = m.unsure_adj > UNCLEAR
    if consensus_only:
        drop = (m.split == "train") & (m.contested | m.unclear)
        m = m[~drop].copy()
    return m


def class_weights(tr: pd.DataFrame, n_cls: int = 2) -> torch.Tensor:
    n = np.bincount(tr.y.values, minlength=n_cls).astype(float)
    w = n.sum() / np.maximum(n, 1)
    return torch.as_tensor(w / w.mean(), dtype=torch.float)


def held_out_city(m: pd.DataFrame, city: str) -> pd.DataFrame:
    """Re-split so one city is the test set.

    The harder generalisation test, and the analogue of xBD's held-out-event
    runs. Per that work: these are not pooled runs and must be reported apart.
    """
    m = m.copy()
    is_city = m.city == city
    if not is_city.any():
        raise SystemExit(f"no crops for city {city!r}")
    m.loc[is_city, "split"] = "test"
    # carve a validation set out of the remaining panoramas, still grouped
    rest = m.loc[~is_city, "pano_id"].unique()
    rng = np.random.default_rng(0)
    val = set(rest[rng.permutation(len(rest))[:max(1, len(rest) // 6)]])
    m.loc[~is_city, "split"] = np.where(
        m.loc[~is_city, "pano_id"].isin(val), "val", "train")
    return m
