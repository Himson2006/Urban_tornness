"""Plain (non-prototype) CNN baseline on PIE intent.

The urban claim is that pedestrian intent models are most confident where humans
most disagree. Measured on a single ProtoPNet, that claim cannot be separated
from "this particular ProtoPNet was badly calibrated" -- which is the objection
that would sink the paper.

So we need the same analysis on ordinary classifiers: several backbones, no
prototype layer, same splits, same sampler, same crops. If the inversion holds
here too, the finding is about single-frame intent models in general.

Writes per-crop held-out predictions in the schema stance_inversion.py expects.

Usage:
    python src/train_plain.py --arch resnet18 --split kfold --fold 0 --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pie_dataset import PIECropDataset, kfold_assign, load_store  # noqa: E402

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
BUILDERS = {"resnet18": models.resnet18, "resnet34": models.resnet34,
            "resnet50": models.resnet50, "vgg16": models.vgg16_bn,
            "densenet121": models.densenet121}


def make_model(arch: str) -> nn.Module:
    m = BUILDERS[arch](weights="DEFAULT")
    if arch.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif arch.startswith("vgg"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    else:
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m


def splits(args):
    df = load_store(crops=args.crops, manifest=args.manifest)
    df = df[df.in_exp_window & (df.bbox_h >= args.min_bbox_h)]
    if args.split == "official":
        tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
        tag = "official"
    else:
        peds = pd.read_parquet(args.manifest / "peds.parquet")
        folds = kfold_assign(peds, n_folds=args.n_folds, group="video", seed=0)
        df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
        vf = (args.fold + 1) % args.n_folds
        tr = df[(df.fold != args.fold) & (df.fold != vf)]
        va, te = df[df.fold == vf], df[df.fold == args.fold]
        tag = f"fold{args.fold}"
    # thin the training set only; evaluation keeps every frame
    if args.frame_stride > 1:
        tr = tr[tr.frame % args.frame_stride == 0]
    return tr, va, te, tag


@torch.no_grad()
def predict(model, df, crops, bs, workers):
    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize(MEAN, STD)])
    dl = torch.utils.data.DataLoader(
        PIECropDataset(df, crops, tf), batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
        collate_fn=lambda b: torch.stack([x[0] for x in b]))
    model.eval()
    ps = []
    for x in dl:
        p = F.softmax(model(x.cuda(non_blocking=True)), 1)
        ps.append(p.cpu())
    p = torch.cat(ps).numpy()
    out = df.copy()
    out["p_cross"] = p[:, 1]
    eps = 1e-12
    out["entropy"] = -(np.clip(p, eps, 1) * np.log(np.clip(p, eps, 1))).sum(1)
    out["margin"] = np.abs(p[:, 1] - p[:, 0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--out", type=Path, default=ROOT / "runs_plain")
    ap.add_argument("--arch", default="resnet18", choices=list(BUILDERS))
    ap.add_argument("--split", choices=["kfold", "official"], default="kfold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--min-bbox-h", type=float, default=40.0)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.backends.cudnn.benchmark = True

    tr, va, te, tag = splits(a)
    run = a.out / f"{a.arch}_{tag}"
    run.mkdir(parents=True, exist_ok=True)
    dest = run / "predictions.parquet"
    if dest.exists():
        print(f"{dest} exists; skipping")
        return
    print(f"{a.arch} {tag} | train {len(tr):,} val {len(va):,} test {len(te):,}")

    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(),
        T.Normalize(MEAN, STD)])
    y = tr.intent_binary.to_numpy()
    w = np.where(y == 1, 1 / max((y == 1).sum(), 1), 1 / max((y == 0).sum(), 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    dl = torch.utils.data.DataLoader(
        PIECropDataset(tr, a.crops, train_tf), batch_size=a.batch,
        sampler=sampler, num_workers=a.workers, pin_memory=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                              torch.tensor([x[1] for x in b])))

    model = make_model(a.arch).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    from sklearn.metrics import roc_auc_score
    best, log = -1.0, []
    for ep in range(a.epochs):
        model.train()
        tot = n = 0
        for x, yy in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x.cuda()), yy.cuda())
            loss.backward()
            opt.step()
            tot += loss.item() * len(yy)
            n += len(yy)
        vp = predict(model, va, a.crops, a.batch, a.workers)
        # select on pedestrian-level AUROC: crops of one pedestrian are near
        # duplicates, and accuracy is misleading under a balanced sampler
        g = vp.groupby("ped_id").agg(y=("intent_binary", "first"),
                                     p=("p_cross", "mean"))
        auc = roc_auc_score(g.y, g.p)
        log.append({"epoch": ep, "train_loss": tot / n, "val_auroc": auc})
        print(f"  epoch {ep} loss {tot/n:.4f} val_ped_auroc {auc:.4f}", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), run / "best.pt")

    model.load_state_dict(torch.load(run / "best.pt"))
    out = predict(model, te, a.crops, a.batch, a.workers)
    out["arch"] = a.arch
    out["model_kind"] = "plain"
    out.to_parquet(dest, index=False)
    pd.DataFrame(log).to_csv(run / "metrics.csv", index=False)
    g = out.groupby("ped_id").agg(y=("intent_binary", "first"),
                                  p=("p_cross", "mean"))
    print(f"best val ped-AUROC {best:.4f} | TEST ped-AUROC "
          f"{roc_auc_score(g.y, g.p):.4f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
