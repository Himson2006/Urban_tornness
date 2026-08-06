"""Paired pre/post building crops as a torch dataset.

Two things here are load-bearing for the study.

**Six channels, not three.** A sample is the pre-disaster crop stacked on the
post-disaster crop. The prototypes a model learns over that input are prototypes
of *change*, which is the object the explanation is supposed to be about. The
three-channel post-only mode exists so that the ablation -- does the pairing
actually buy anything -- is a flag rather than a second codebase.

**Scene-grouped splits.** Buildings in one scene share a satellite pass: the same
lighting, the same off-nadir angle, the same roof material, often the same
construction. Splitting buildings at random puts near-duplicates on both sides
and inflates every number. This is the leak that had to be fixed in the
pedestrian work, where the dataset's own k-fold shuffled over pedestrians and
let the same video appear in train and test. Group by scene; optionally hold out
whole disasters, which is the harder and more honest generalisation test.

The primary task is minor-damage vs major-damage. That is the contested
boundary: both are standing buildings with visible damage, and the split between
them is a threshold judgement rather than a category difference. no-damage vs
destroyed is not a judgement call, and a model that only has to make that
distinction has nothing to be torn about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ImageNet statistics, applied per 3-channel block
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

TASKS = {
    # the contested boundary -- the analogue of melanoma vs nevus
    "middle": ["minor-damage", "major-damage"],
    # the easy control: if co-activation means anything, it should be far
    # smaller here than on `middle` for the same architecture
    "extremes": ["no-damage", "destroyed"],
    "all4": ["no-damage", "minor-damage", "major-damage", "destroyed"],
}


def load_meta(crops: Path, task: str = "middle", min_side: float = 24.0,
              buildings: Path | None = None) -> pd.DataFrame:
    """Crop metadata filtered to a task, with disaster attached."""
    m = pd.read_parquet(Path(crops) / "crop_meta.parquet")
    classes = TASKS[task]
    m = m[m.damage.isin(classes) & (m.px_side >= min_side)].copy()
    m["label"] = m.damage.map({c: i for i, c in enumerate(classes)})
    if buildings is not None and Path(buildings).exists():
        b = pd.read_parquet(buildings)[
            ["uid", "disaster", "lon", "lat", "gsd", "off_nadir", "sun_elev"]]
        m = m.merge(b, on="uid", how="left")
    else:
        m["disaster"] = m.scene.str.rsplit("_", n=1).str[0]
    return m.reset_index(drop=True)


def assign_folds(m: pd.DataFrame, k: int = 5, by: str = "scene",
                 seed: int = 0) -> pd.Series:
    """Group-aware fold labels.

    `by="scene"` is the default and the minimum defensible grouping.
    `by="disaster"` holds out entire events, which is what a deployment
    actually faces: the next hurricane is not in the training set.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = m[by].values
    if by == "disaster" and m.disaster.nunique() < k:
        k = max(2, m.disaster.nunique())
    fold = np.full(len(m), -1, dtype=int)
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgk.split(m, m.label, groups)):
        fold[te] = i
    return pd.Series(fold, index=m.index, name="fold")


class PairedCropDataset(Dataset):
    """Yields (C, S, S) float tensor, label, uid.

    C is 6 when paired (pre stacked on post) and 3 when post-only.
    """

    def __init__(self, meta: pd.DataFrame, crops: Path, size: int = 96,
                 paired: bool = True, augment: bool = False,
                 normalize: bool = True):
        self.m = meta.reset_index(drop=True)
        self.root = Path(crops)
        self.size = size
        self.paired = paired
        self.augment = augment
        self.normalize = normalize
        n = 2 if paired else 1
        self._mean = torch.tensor(MEAN * n).view(-1, 1, 1)
        self._std = torch.tensor(STD * n).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.m)

    def _read(self, rel: str) -> np.ndarray:
        import cv2
        im = cv2.imread(str(self.root / rel))
        if im is None:
            im = np.zeros((self.size, self.size, 3), np.uint8)
        im = cv2.resize(im, (self.size, self.size),
                        interpolation=cv2.INTER_LINEAR)
        return im[:, :, ::-1]                        # BGR -> RGB

    def __getitem__(self, i: int):
        r = self.m.iloc[i]
        post = self._read(r.post)
        ims = [self._read(r.pre), post] if self.paired else [post]

        if self.augment:
            # geometry must be applied identically to both members of the pair,
            # or the model is handed a misregistration it never sees at test
            if np.random.rand() < 0.5:
                ims = [im[:, ::-1] for im in ims]
            if np.random.rand() < 0.5:
                ims = [im[::-1, :] for im in ims]
            rot = np.random.randint(4)
            if rot:
                ims = [np.rot90(im, rot) for im in ims]

        x = np.concatenate([np.ascontiguousarray(im) for im in ims], axis=2)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0
        if self.normalize:
            x = (x - self._mean) / self._std
        return x, int(r.label), str(r.uid)


def class_weights(m: pd.DataFrame, n_classes: int) -> torch.Tensor:
    c = np.bincount(m.label.values, minlength=n_classes).astype(float)
    w = c.sum() / (n_classes * np.maximum(c, 1))
    return torch.tensor(w, dtype=torch.float32)
