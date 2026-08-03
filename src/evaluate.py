"""Proper evaluation of a trained PIE ProtoPNet.

Raw accuracy is the wrong headline metric here and comparing it to a majority
baseline is close to meaningless: training uses a WeightedRandomSampler, so the
model is optimised for *balanced* accuracy while the test split is 82% positive.
A balanced-trained model trades positive recall for negative recall and will sit
below the majority rate by construction, even when it has learned a great deal.

So we report balanced accuracy, AUROC and AP -- which are what the tornness
experiments actually depend on, since those need calibrated-ish p(cross) on
contested cases, not argmax accuracy.

Also reports pedestrian-level metrics (crops of one pedestrian are near
duplicates, so crop-level numbers overstate the effective sample size).

Usage:
    python src/evaluate.py --ckpt runs/resnet34_official/best.pth --split official
    python src/evaluate.py --ckpt runs/resnet34_fold0/best.pth --split kfold --fold 0
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

from preprocess import mean, std                                   # noqa: E402
from pie_dataset import PIECropDataset, kfold_assign, load_store    # noqa: E402

IMG_SIZE = 224


def metrics(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score, average_precision_score,
                                 confusion_matrix)
    yhat = (p >= 0.5).astype(int)
    majority = max(y.mean(), 1 - y.mean())
    out = {
        "level": label, "n": len(y), "pos_rate": y.mean(),
        "majority_baseline": majority,
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "ap": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
    }
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out.update(tn=tn, fp=fp, fn=fn, tp=tp)
    return out


@torch.no_grad()
def score(ppnet, loader) -> tuple[np.ndarray, np.ndarray]:
    ps = []
    for x, _ in loader:
        logits, _ = ppnet(x.cuda(non_blocking=True))
        ps.append(F.softmax(logits, 1)[:, 1].cpu())
    return torch.cat(ps).numpy()


def prototype_health(ppnet) -> str:
    """Is the model still using distinct, class-separated prototypes?

    `p dist pair` collapsing toward 0 means the two classes' prototype sets have
    converged, so co-activation would measure redundancy rather than genuine
    cue conflict -- which would invalidate the whole premise.
    """
    P = ppnet.prototype_vectors.detach().flatten(1)                # (n_proto, d)
    cls = ppnet.prototype_class_identity.argmax(1).cpu().numpy()
    D = torch.cdist(P, P).cpu().numpy()
    iu = np.triu_indices(len(P), k=1)
    same = (cls[iu[0]] == cls[iu[1]])
    return (f"  mean pairwise prototype distance : {D[iu].mean():.4f}\n"
            f"    within-class                   : {D[iu][same].mean():.4f}\n"
            f"    between-class                  : {D[iu][~same].mean():.4f}\n"
            f"    between/within ratio           : "
            f"{D[iu][~same].mean() / max(D[iu][same].mean(), 1e-9):.3f}"
            f"   (~1.0 => classes share prototypes => collapse)\n"
            f"  duplicate prototypes (dist<0.01) : "
            f"{int((D[iu] < 0.01).sum())} of {len(iu[0])} pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--split", choices=["kfold", "official"], default="official")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-bbox-h", type=float, default=40.0)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    df = load_store(crops=a.crops, manifest=a.manifest)
    df = df[df.in_exp_window & (df.bbox_h >= a.min_bbox_h)]
    if a.split == "official":
        df = df[df.split == "test"]
    else:
        peds = pd.read_parquet(a.manifest / "peds.parquet")
        folds = kfold_assign(peds, n_folds=a.n_folds, group="video", seed=0)
        df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
        df = df[df.fold == a.fold]
    df = df.reset_index(drop=True)

    ppnet = torch.load(a.ckpt, map_location="cuda", weights_only=False).cuda().eval()
    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize(mean=mean, std=std)])
    loader = torch.utils.data.DataLoader(
        PIECropDataset(df, a.crops, tf), batch_size=a.batch, shuffle=False,
        num_workers=a.workers, pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              torch.tensor([x[1] for x in b])))

    p = score(ppnet, loader)
    df["p_cross"] = p

    rows = [metrics(df.intent_binary.to_numpy(), p, "crop")]
    per_ped = df.groupby("ped_id").agg(y=("intent_binary", "first"),
                                       p=("p_cross", "mean"))
    rows.append(metrics(per_ped.y.to_numpy(), per_ped.p.to_numpy(), "pedestrian"))

    print(f"\n=== {a.ckpt} | split={a.split}"
          f"{'' if a.split=='official' else f' fold{a.fold}'} ===\n")
    m = pd.DataFrame(rows).set_index("level")
    with pd.option_context("display.width", 200):
        print(m[["n", "pos_rate", "majority_baseline", "accuracy",
                 "balanced_accuracy", "auroc", "ap"]].round(4).to_string())
        print()
        print(m[["tn", "fp", "fn", "tp"]].to_string())

    print("\n=== prototype health ===")
    print(prototype_health(ppnet))

    ped = rows[1]
    print("\n=== read ===")
    if ped["auroc"] > 0.70 and ped["balanced_accuracy"] > 0.65:
        print("  Model has real signal. Raw accuracy below the majority baseline is")
        print("  expected with balanced training -- judge it on AUROC/balanced acc.")
    elif ped["auroc"] > 0.60:
        print("  Weak but non-trivial signal. Usable for tornness experiments only")
        print("  if prototype health below looks sane.")
    else:
        print("  NOT LEARNING. Fix training before running any tornness analysis.")


if __name__ == "__main__":
    main()
