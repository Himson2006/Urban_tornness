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
from pie_dataset import (PIECropDataset, crop_transform, kfold_assign,
                         load_store)    # noqa: E402

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


def prototype_health(ppnet) -> tuple[str, int]:
    """Are the prototypes distinct, and are they distinct *within* each class?

    Two different failure modes, and they pull the summary statistics in
    opposite directions:

      between-class collapse -- the two classes share prototypes. Shows up as
        between/within ratio near 1.0. Co-activation would then measure
        redundancy rather than cue conflict.

      within-class collapse -- each class keeps one prototype, duplicated N
        times. Shows up as a *high* ratio, which naively looks healthy, but the
        effective vocabulary is 1 exemplar per class. Co-activation still works
        (it only needs the per-class max) but the per-prototype profile fallback
        is dead, and "the exemplars name the two readings" loses most of its
        force.

    Returns (report, n_effective_prototypes).
    """
    P = ppnet.prototype_vectors.detach().flatten(1)                # (n_proto, d)
    cls = ppnet.prototype_class_identity.argmax(1).cpu().numpy()
    D = torch.cdist(P, P).cpu().numpy()
    iu = np.triu_indices(len(P), k=1)
    same = (cls[iu[0]] == cls[iu[1]])
    w, b = D[iu][same].mean(), D[iu][~same].mean()

    # effective vocabulary: single-linkage clusters at a tight threshold
    thr = 0.01 * max(D[iu].mean(), 1e-9) / max(D[iu].mean(), 1e-9) + 0.01
    n = len(P)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < thr:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    eff = len(groups)
    eff_by_cls = {c: len({find(i) for i in range(n) if cls[i] == c}) for c in (0, 1)}

    lines = [
        f"  mean pairwise prototype distance : {D[iu].mean():.4f}",
        f"    within-class                   : {w:.4f}",
        f"    between-class                  : {b:.4f}",
        f"    between/within ratio           : {b / max(w, 1e-9):.3f}"
        f"   (~1.0 => classes share prototypes)",
        f"  duplicate pairs (dist<0.01)      : {int((D[iu] < 0.01).sum())} of {len(iu[0])}",
        f"  EFFECTIVE prototypes             : {eff} of {n}"
        f"   (class0: {eff_by_cls[0]}/{int((cls==0).sum())},"
        f" class1: {eff_by_cls[1]}/{int((cls==1).sum())})",
    ]
    if eff < 0.5 * n:
        lines.append(f"  !! WITHIN-CLASS COLLAPSE: {n - eff} of {n} prototypes are "
                     f"duplicates of another.")
    if b / max(w, 1e-9) < 1.5:
        lines.append("  !! BETWEEN-CLASS COLLAPSE: classes are not using distinct "
                     "prototypes.")
    return "\n".join(lines), eff


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
    ap.add_argument("--crop-scale", type=float, default=2.0,
                    help="effective crop as a multiple of the bbox; "
                         "stored crops are 2.0. Lower values center-crop "
                         "so the pedestrian dominates the frame.")
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
    _t = crop_transform(a.crop_scale)
    tf = T.Compose(([_t] if _t else []) +
                   [T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
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
    health, eff = prototype_health(ppnet)
    print(health)

    ped = rows[1]
    n_proto = ppnet.prototype_vectors.shape[0]
    print("\n=== read ===")
    if ped["auroc"] < 0.60:
        print("  NOT LEARNING. Fix training before any tornness analysis.")
    elif eff < 0.5 * n_proto:
        print(f"  Discriminative signal is fine (AUROC {ped['auroc']:.3f}) but the")
        print(f"  prototype layer collapsed to {eff} effective exemplars. Co-activation")
        print("  still works; per-prototype profiles and the exemplar story do not.")
        print("  Retrain with fewer prototypes/class before the analysis.")
    elif ped["auroc"] > 0.70:
        print(f"  Usable. AUROC {ped['auroc']:.3f}, {eff}/{n_proto} effective prototypes.")
        print("  Accuracy below the majority baseline is expected under balanced")
        print("  training -- judge on AUROC / balanced accuracy.")
    else:
        print("  Weak but non-trivial. Proceed with caution.")


if __name__ == "__main__":
    main()
