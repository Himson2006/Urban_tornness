"""Multi-frame input for PIE: give the model the cue it cannot currently see.

The diagnosis says the failure is temporal. Human disagreement is driven by
stance -- standing vs walking, rho=+0.220, stronger than any model signal -- and
a single crop cannot distinguish "paused mid-stride, about to go" from "standing
still, staying put". So the model resolves the ambiguity confidently and
arbitrarily, and its confidence runs backwards to human uncertainty.

This module stacks T crops from the same pedestrian track, spaced `gap` frames
apart, into a 3T-channel input and inflates the backbone's first convolution to
accept it. Everything downstream -- prototypes, push, tornness -- is unchanged,
so the comparison against the single-frame model is clean.

Success criteria, fixed BEFORE looking (two of three):
  1. entropy vs frac_standing moves from -0.116 toward zero or positive
  2. silent-failure rate drops
  3. coact_min vs frac_standing significant POOLED over five folds
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_lookup(df: pd.DataFrame) -> dict:
    """(ped_id, frame) -> crop_path, over every extracted crop."""
    return dict(zip(zip(df.ped_id, df.frame), df.crop_path))


def inflate_conv1(features: nn.Module, n_frames: int) -> nn.Module:
    """Widen the stem conv from 3 to 3*T channels, reusing pretrained weights.

    Each frame's copy is the original kernel divided by T, so a static clip
    (all frames identical) produces exactly the activations the pretrained
    single-frame stem would have produced. Training starts from a sane place
    instead of random.
    """
    if n_frames == 1:
        return features
    old = features.conv1
    new = nn.Conv2d(3 * n_frames, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, n_frames, 1, 1) / n_frames)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    features.conv1 = new
    return features


class PIESeqDataset(Dataset):
    """Yields (3T, H, W), label, meta. Frame t plus t-gap, t-2gap, ...

    Frames before a track starts fall back to the earliest available crop, so
    the clip is padded by repetition rather than by black frames -- repetition
    reads as "not moving", which is a real state, whereas black is not.
    """

    def __init__(self, df: pd.DataFrame, crops_root: Path, lookup: dict,
                 n_frames: int = 3, gap: int = 5, img_size: int = 224,
                 crop_scale: float = 2.0, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(crops_root)
        self.lookup = lookup
        self.n_frames = n_frames
        self.gap = gap
        self.img_size = img_size
        self.train = train
        self.frac = min(crop_scale / 2.0, 1.0)   # stored crops are 2x bbox

    def __len__(self):
        return len(self.df)

    def _load(self, ped_id, frame, flip: bool, jitter):
        path = None
        # walk backwards to the nearest earlier frame that exists
        for f in range(frame, frame - self.gap * self.n_frames - 1, -1):
            if (ped_id, f) in self.lookup:
                path = self.lookup[(ped_id, f)]
                break
        if path is None:
            path = self.df.crop_path.iloc[0]
        img = Image.open(self.root / path).convert("RGB")
        if self.frac < 1.0:
            w, h = img.size
            img = TF.center_crop(img, [max(int(round(h * self.frac)), 8),
                                       max(int(round(w * self.frac)), 8)])
        img = TF.resize(img, [self.img_size, self.img_size])
        if flip:
            img = TF.hflip(img)
        x = TF.to_tensor(img)
        if jitter is not None:
            x = (x * jitter).clamp(0, 1)
        return TF.normalize(x, MEAN, STD)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # one augmentation decision per clip, so frames stay registered
        flip = self.train and torch.rand(1).item() < 0.5
        jitter = (0.8 + 0.4 * torch.rand(1)).item() if self.train else None
        frames = [int(r.frame) - k * self.gap for k in range(self.n_frames)]
        x = torch.cat([self._load(r.ped_id, f, flip, jitter) for f in frames], 0)
        meta = {"ped_id": r.ped_id, "frame": int(r.frame)}
        return x, int(r.intent_binary), meta


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
    """Push needs images in [0,1]; re-scale rather than reload."""

    def __init__(self, ds):
        self.ds = ds
        self.m = torch.tensor(MEAN).repeat(ds.n_frames)[:, None, None]
        self.s = torch.tensor(STD).repeat(ds.n_frames)[:, None, None]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return (x * self.s + self.m).clamp(0, 1), y


def preprocess_stack(n_frames: int):
    """Normalisation function for push, matched to the stacked input."""
    m = torch.tensor(MEAN).repeat(n_frames).view(1, -1, 1, 1)
    s = torch.tensor(STD).repeat(n_frames).view(1, -1, 1, 1)

    def f(x):
        return (x - m.to(x.device)) / s.to(x.device)
    return f
