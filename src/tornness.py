"""Extract tornness features from a trained ProtoPNet: the paper's measurement.

For every crop we record not just *how* uncertain the model is but *what shape*
the uncertainty has, read off the prototype layer:

  max_sim_0, max_sim_1   strongest similarity to each class's prototype set
  coact_min              min(max_sim_0, max_sim_1)  -- dual-match evidence
  coact_prod             max_sim_0 * max_sim_1
  global_max_sim         max similarity to any prototype -- weak-match evidence
  margin                 |p1 - p0|; small margin = torn
  entropy                scalar baseline that cannot tell the two types apart

Typing, within torn cases: high coact_min => dual-match (both readings
supported); low coact_min *and* low global_max_sim => weak-match (no evidence).
Reported as a spectrum -- `typing_score` -- never a hard binary.

Per-prototype similarity vectors are saved alongside as the fallback feature if
co-activation over 2 classes proves too coarse.

Usage:
    python src/tornness.py --ckpt runs/resnet34_fold0/best.pth --fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "src"))

from preprocess import mean, std          # noqa: E402
from pie_dataset import (PIECropDataset, crop_transform, kfold_assign,
                         load_store)  # noqa: E402

IMG_SIZE = 224


@torch.no_grad()
def forward_features(ppnet, x, proto_class, mc_passes=0, mc_p=0.2):
    """Logits + per-prototype similarities for one batch.

    ProtoPNet's `forward` returns (logits, min_distances); similarities come from
    the model's own `distance_2_similarity`, so nothing here re-implements the
    scoring -- we just stop throwing the vector away.
    """
    logits, min_dist = ppnet(x)
    sims = ppnet.distance_2_similarity(min_dist)      # (B, n_proto)
    probs = F.softmax(logits, dim=1)

    mc_std = None
    if mc_passes > 0:
        # MC-dropout baseline. ProtoPNet has no dropout layer, so we inject it on
        # the conv feature map -- this is feature-level MC-dropout, not the
        # original formulation, and is labelled as such in the paper.
        ps = []
        for _ in range(mc_passes):
            conv = ppnet.conv_features(x)
            conv = F.dropout(conv, p=mc_p, training=True)
            d = ppnet._l2_convolution(conv)
            md = -F.max_pool2d(-d, kernel_size=(d.size(2), d.size(3)))
            md = md.view(-1, ppnet.num_prototypes)
            ps.append(F.softmax(ppnet.last_layer(
                ppnet.distance_2_similarity(md)), dim=1)[:, 1])
        mc_std = torch.stack(ps).std(0)

    return probs, sims, mc_std


def summarize(probs, sims, proto_class):
    """Collapse per-prototype similarities into the tornness feature set."""
    p1 = probs[:, 1]
    eps = 1e-12
    ent = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(1)

    m0 = sims[:, proto_class == 0].max(1).values
    m1 = sims[:, proto_class == 1].max(1).values
    gmax = sims.max(1).values
    return {
        "p_cross": p1,
        "margin": (p1 - (1 - p1)).abs(),
        "entropy": ent,
        "max_sim_0": m0,
        "max_sim_1": m1,
        "coact_min": torch.minimum(m0, m1),
        "coact_prod": m0 * m1,
        "global_max_sim": gmax,
        "sim_gap": (m1 - m0).abs(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--split", choices=["kfold", "official"], default="kfold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mc-passes", type=int, default=20,
                    help="0 disables the MC-dropout baseline")
    ap.add_argument("--include-tail", action="store_true", default=True,
                    help="also score frames past critical_point (resolvability)")
    ap.add_argument("--crop-scale", type=float, default=2.0,
                    help="effective crop as a multiple of the bbox; "
                         "stored crops are 2.0. Lower values center-crop "
                         "so the pedestrian dominates the frame.")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    df = load_store(crops=a.crops, manifest=a.manifest)
    if not a.include_tail:
        df = df[df.in_exp_window]

    # score exactly the held-out pedestrians -- never ones the model trained on
    if a.split == "official":
        df = df[df.split == "test"]
        tag = "official"
    else:
        peds = pd.read_parquet(a.manifest / "peds.parquet")
        folds = kfold_assign(peds, n_folds=a.n_folds, group="video", seed=0)
        df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
        df = df[df.fold == a.fold]
        tag = f"fold{a.fold}"
    df = df.reset_index(drop=True)
    print(f"scoring {len(df):,} crops / {df.ped_id.nunique()} held-out pedestrians")

    ppnet = torch.load(a.ckpt, map_location="cuda", weights_only=False).cuda().eval()
    proto_class = ppnet.prototype_class_identity.argmax(1).cuda()  # (n_proto,)
    print(f"prototypes per class: {torch.bincount(proto_class).tolist()}")

    _t = crop_transform(a.crop_scale)
    tf = T.Compose(([_t] if _t else []) +
                   [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize(mean=mean, std=std)])
    ds = PIECropDataset(df, a.crops, tf)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, shuffle=False,
        num_workers=a.workers, pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              torch.tensor([x[1] for x in b])))

    cols, mc_all, sim_all = {}, [], []
    for bi, (x, _) in enumerate(loader):
        probs, sims, mc = forward_features(
            ppnet, x.cuda(non_blocking=True), proto_class, a.mc_passes)
        for k, v in summarize(probs, sims, proto_class).items():
            cols.setdefault(k, []).append(v.cpu())
        sim_all.append(sims.cpu())
        if mc is not None:
            mc_all.append(mc.cpu())
        if bi % 50 == 0:
            print(f"  batch {bi}/{len(loader)}", flush=True)

    out = df.copy()
    for k, v in cols.items():
        out[k] = torch.cat(v).float().numpy()
    if mc_all:
        out["mc_dropout_std"] = torch.cat(mc_all).float().numpy()

    # typing score: within torn cases, high = dual-match, low = weak-match.
    # Rank-normalised so the threshold is reported as a spectrum, not a magic number.
    r = lambda s: s.rank(pct=True)
    out["typing_score"] = r(out.coact_min) - r(1.0 - out.global_max_sim)
    out["is_torn"] = out.margin < out.margin.quantile(0.25)

    dest = a.out or (a.ckpt.parent / f"tornness_{tag}.parquet")
    out.to_parquet(dest, index=False)
    np.save(dest.with_suffix(".protosims.npy"),
            torch.cat(sim_all).float().numpy())   # fallback per-prototype feature
    print(f"\nwrote {dest} ({len(out):,} rows) + per-prototype similarities")

    torn = out[out.is_torn]
    print(f"\ntorn crops: {len(torn):,} ({100*len(torn)/len(out):.1f}%), "
          f"{torn.ped_id.nunique()} pedestrians")
    print("correlations with human_disagreement, torn cases only "
          "(the Experiment 1 headline):")
    for c in ["coact_min", "coact_prod", "global_max_sim", "typing_score",
              "entropy", "margin"] + (["mc_dropout_std"] if mc_all else []):
        print(f"  {c:16s} spearman {torn[c].corr(torn.human_disagreement, method='spearman'):+.3f}")


if __name__ == "__main__":
    main()
