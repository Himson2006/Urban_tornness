"""Extract 10-class tornness features on the CIFAR-10 test set.

With 10 classes the co-activation idea sharpens. Per image we compute each
class's max prototype similarity, sort them, and read off:

  s1, s2            best and second-best class similarity
  coact_top2 = s2   dual-match evidence: how strongly the runner-up reading is
                    also supported. In binary PIE this was min(m0, m1).
  coact_ratio       s2 / s1 -- how evenly matched the two readings are
  global_max = s1   weak-match evidence: low s1 means nothing matched at all
  top2_share        (s1+s2) / sum(all class sims) -- is the competition confined
                    to two readings, or smeared across many?
  n_eff_sim         exp(entropy of the class-similarity distribution)

And crucially, which two: `model_top1` / `model_top2`. Humans have their own top
two (cat vs dog), so we can ask whether the model's competing pair IS the humans'
competing pair. That test is impossible in binary PIE and is the strongest
version of "the exemplars name the two readings".

Usage:
    python src/cifar_tornness.py --ckpt runs_cifar/resnet34_cifar10/best.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "src"))

from preprocess import mean, std      # noqa: E402

IMG_SIZE = 224


@torch.no_grad()
def extract(ppnet, loader, proto_class, n_classes, mc_passes, mc_p=0.2):
    rows = {k: [] for k in
            ("p_top1", "p_top2", "margin", "entropy", "pred",
             "s1", "s2", "coact_top2", "coact_ratio", "global_max",
             "top2_share", "n_eff_sim", "model_top1", "model_top2",
             "mc_std")}
    onehot = F.one_hot(proto_class, n_classes).float()          # (n_proto, C)

    for x, _ in loader:
        x = x.cuda(non_blocking=True)
        logits, min_dist = ppnet(x)
        sims = ppnet.distance_2_similarity(min_dist)            # (B, n_proto)
        # per-class max similarity: mask out other classes' prototypes
        masked = sims.unsqueeze(2) * onehot.unsqueeze(0)        # (B, n_proto, C)
        masked = masked.masked_fill(onehot.unsqueeze(0) == 0, float("-inf"))
        cls_sim = masked.max(1).values                          # (B, C)

        srt, idx = cls_sim.sort(dim=1, descending=True)
        s1, s2 = srt[:, 0], srt[:, 1]
        pos = cls_sim.clamp_min(1e-12)
        share = pos / pos.sum(1, keepdim=True)
        ent_sim = -(share * share.log()).sum(1)

        probs = F.softmax(logits, 1)
        psrt = probs.sort(dim=1, descending=True).values

        vals = {
            "p_top1": psrt[:, 0], "p_top2": psrt[:, 1],
            "margin": psrt[:, 0] - psrt[:, 1],
            "entropy": -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(1),
            "pred": probs.argmax(1),
            "s1": s1, "s2": s2, "coact_top2": s2,
            "coact_ratio": s2 / s1.clamp_min(1e-12),
            "global_max": s1,
            "top2_share": (s1 + s2) / pos.sum(1),
            "n_eff_sim": ent_sim.exp(),
            "model_top1": idx[:, 0], "model_top2": idx[:, 1],
        }
        if mc_passes > 0:
            ps = []
            for _ in range(mc_passes):
                conv = F.dropout(ppnet.conv_features(x), p=mc_p, training=True)
                d = ppnet._l2_convolution(conv)
                md = (-F.max_pool2d(-d, kernel_size=(d.size(2), d.size(3)))
                      ).view(-1, ppnet.num_prototypes)
                ps.append(F.softmax(ppnet.last_layer(
                    ppnet.distance_2_similarity(md)), 1))
            st = torch.stack(ps)
            vals["mc_std"] = st.std(0).max(1).values
        else:
            vals["mc_std"] = torch.full_like(s1, float("nan"))

        for k, v in vals.items():
            rows[k].append(v.cpu())
    return {k: torch.cat(v).numpy() for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs_cifar/resnet34_cifar10/best.pth")
    ap.add_argument("--data", type=Path, default=ROOT / "data/cifar10")
    ap.add_argument("--human", type=Path,
                    default=ROOT / "data/cifar10h/human_labels.parquet")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mc-passes", type=int, default=20)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    tf = T.Compose([T.Resize(IMG_SIZE), T.ToTensor(),
                    T.Normalize(mean=mean, std=std)])
    ds = CIFAR10(a.data, train=False, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
        pin_memory=True)

    ppnet = torch.load(a.ckpt, map_location="cuda",
                       weights_only=False).cuda().eval()
    proto_class = ppnet.prototype_class_identity.argmax(1).cuda()
    n_classes = ppnet.prototype_class_identity.shape[1]
    print(f"{ppnet.num_prototypes} prototypes over {n_classes} classes; "
          f"per class {torch.bincount(proto_class).tolist()}")

    feats = extract(ppnet, loader, proto_class, n_classes, a.mc_passes)
    df = pd.DataFrame(feats)
    df["idx"] = np.arange(len(df))
    df["true_label"] = np.array(ds.targets)

    human = pd.read_parquet(a.human)
    df = df.merge(human, on="idx", how="inner")
    # does the model's competing pair match the humans' competing pair?
    mt = df[["model_top1", "model_top2"]].to_numpy()
    ht = df[["human_label", "runner_up"]].to_numpy()
    df["pair_match"] = [set(m) == set(h) for m, h in zip(mt, ht)]
    df["top1_match"] = df.model_top1 == df.human_label

    dest = a.out or (a.ckpt.parent / "tornness_cifar10h.parquet")
    df.to_parquet(dest, index=False)
    print(f"\nwrote {dest} ({len(df):,} images)")
    print(f"  model top-1 == human top-1 : {df.top1_match.mean():.4f}")
    print(f"  model top-2 pair == human pair: {df.pair_match.mean():.4f}")
    con = df[df.top1 < 0.90]
    print(f"  on the {len(con):,} contested images: "
          f"pair match {con.pair_match.mean():.4f}")


if __name__ == "__main__":
    main()
