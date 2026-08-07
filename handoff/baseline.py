"""Plain ResNet classifier on Sidewalk label crops -- the fast path to Role B.

Role B does not need prototypes. It needs a classifier whose uncertainty means
something, and a plain backbone gives that in a fraction of the time ProtoPNet's
warm/joint/push schedule takes. Running this first gets the paper's central
finding -- whether reviewer disagreement is recoverable from the image -- before
the prototype machinery is known to work, which is the schedule risk.

`train.py` writes `test_preds.parquet` with the same columns, so `roleb.py`
reads either interchangeably and the two can be compared directly.

**Three uncertainties, not one.** With dropout left on at inference and N
stochastic passes, the predictive entropy decomposes:

    total     = H[ E_t p_t ]            all of it
    aleatoric = E_t H[ p_t ]            irreducible given this image
    epistemic = total - aleatoric       mutual information; what more data fixes

That decomposition is worth the extra passes here because the two disagreement
targets are supposed to behave differently under it. `unsure_adj` means
reviewers could not tell from the image -- a claim about missing evidence.
`split_adj` means they could tell and disagreed anyway -- a claim about where
the standard sits, which no amount of image evidence resolves. If the
dissociation is real, the components should not track both targets equally.

`--passes 1` disables dropout sampling and reports the deterministic softmax
entropy alone, which is the vanilla baseline.

Usage:
    python handoff/baseline.py --task obstacle --dry-run
    python handoff/baseline.py --task obstacle --passes 30
    python handoff/baseline.py --task obstacle --seed 1     # ensemble member
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "handoff"))

from dataset import (CropDataset, TwoTuple, class_weights,  # noqa: E402
                     held_out_city, load_manifest)

# Carried through to test_preds.parquet so roleb.py can partial on them. These
# are the covariates a "resolution detector" would exploit; see DESIGN.md §5.
COVARIATES = ["severity", "n_val", "image_age_days", "label_age_days",
              "zoom", "pano_width"]


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build(arch: str, dropout: float, dev) -> nn.Module:
    import torchvision.models as tvm

    net = getattr(tvm, arch)(weights="IMAGENET1K_V1")
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(net.fc.in_features, 2))
    return net.to(dev)


def mc_forward(net, loader, dev, passes: int):
    """N stochastic passes with dropout active; returns (T, N, 2) probabilities.

    Only the dropout layers are put back into train mode. Calling net.train()
    wholesale would also unfreeze BatchNorm's running statistics, so the
    "uncertainty" would partly be batch-composition noise -- an artefact of how
    the loader happened to group crops, not a property of the image.
    """
    net.eval()
    if passes > 1:
        for m in net.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    P, Y = [], []
    with torch.no_grad():
        for t in range(passes):
            pt, yt = [], []
            for x, y in loader:
                pt.append(torch.softmax(net(x.to(dev)), 1).cpu().numpy())
                if t == 0:
                    yt.append(y.numpy())
            P.append(np.concatenate(pt))
            if t == 0:
                Y = np.concatenate(yt)
    return np.stack(P), Y


def decompose(P: np.ndarray) -> dict:
    """P is (T, N, C). Returns mean probabilities and the entropy split."""
    def H(p):
        return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1) / np.log(2)

    mean = P.mean(0)
    total = H(mean)
    aleatoric = H(P).mean(0)
    return {"p": mean, "total": total, "aleatoric": aleatoric,
            "epistemic": total - aleatoric}


def scores(P: np.ndarray, Y: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    pred = P.argmax(1)
    out = {"acc": float((pred == Y).mean()),
           "balanced_acc": float(balanced_accuracy_score(Y, pred)),
           "majority": float(max(np.bincount(Y, minlength=2)) / len(Y))}
    try:
        out["auc"] = float(roc_auc_score(Y, P[:, 1]))
    except ValueError:
        out["auc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="obstacle")
    ap.add_argument("--data", type=Path, default=ROOT / "handoff/data")
    ap.add_argument("--out", type=Path, default=ROOT / "handoff/runs")
    ap.add_argument("--held-out-city", default="")
    ap.add_argument("--all-train", action="store_true",
                    help="train on contested labels too -- makes Role B "
                         "circular; for the ablation only")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--passes", type=int, default=30,
                    help="MC-Dropout passes; 1 = deterministic softmax")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_all(a.seed)
    crops = a.data / "hf" / a.task
    tag = (f"{a.task}_baseline_{'all' if a.all_train else 'consensus'}"
           f"{'_' + a.held_out_city if a.held_out_city else '_pano'}"
           f"_{a.arch}_s{a.seed}")
    run = a.out / tag
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print(f"{run} already complete; delete DONE to retrain")
        return

    m = load_manifest(a.data / f"manifest_{a.task}.parquet",
                      consensus_only=not a.all_train)
    if a.held_out_city:
        m = held_out_city(m, a.held_out_city)
    tr, va, te = (m[m.split == s] for s in ("train", "val", "test"))

    print(f"task={a.task} | consensus_only={not a.all_train} | "
          f"split={'city:' + a.held_out_city if a.held_out_city else 'pano'}")
    for nm, s in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {nm:5s} {len(s):6,} crops  {s.pano_id.nunique():5,} panos  "
              f"incorrect-rate {s.y.mean():.3f}  "
              f"contested {int(s.contested.sum()):4,}  "
              f"unclear {int(s.unclear.sum()):4,}")

    n_eval = int(te.contested.sum() + te.unclear.sum())
    if n_eval < 100:
        msg = (f"only {n_eval} contested-or-unclear crops in test; Role B "
               f"needs a few hundred (crosswalk never clears this)")
        if not a.dry_run:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")

    def dl(sub, shuffle, augment):
        return torch.utils.data.DataLoader(
            TwoTuple(CropDataset(sub, crops, a.img_size, augment)),
            batch_size=a.batch, shuffle=shuffle, num_workers=a.workers,
            pin_memory=True)

    va_l, te_l = dl(va, False, False), dl(te, False, False)
    cw = class_weights(tr)
    if a.balance and float(cw.max() / cw.min()) > 1.5:
        w = cw.numpy()[tr.y.values]
        tr_l = torch.utils.data.DataLoader(
            TwoTuple(CropDataset(tr, crops, a.img_size, True)),
            batch_size=a.batch, num_workers=a.workers, pin_memory=True,
            sampler=torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(w, dtype=torch.double), len(w), replacement=True))
        print(f"class imbalance {float(cw.max()/cw.min()):.1f}:1 -> balanced sampler")
    else:
        tr_l = dl(tr, True, True)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    net = build(a.arch, a.dropout, dev)
    print(f"device={dev}  arch={a.arch}  dropout={a.dropout}  passes={a.passes}")

    if a.dry_run:
        x, y = next(iter(tr_l))
        with torch.no_grad():
            out = net(x.to(dev))
        print(f"dry run: batch {tuple(x.shape)} -> logits {tuple(out.shape)}")
        P, Y = mc_forward(net, va_l, dev, min(a.passes, 3))
        u = decompose(P)
        print(f"  MC forward {P.shape} -> total {u['total'].shape}, "
              f"epistemic mean {u['epistemic'].mean():.4f}")
        print("dry run OK")
        return

    head = list(net.fc.parameters())
    body = [p for n, p in net.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": a.backbone_lr},
         {"params": head, "lr": a.lr}], weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.CrossEntropyLoss()

    best_va, best = -1.0, run / "best.pth"
    for ep in range(a.epochs):
        net.train()
        tot = n = 0
        for x, y in tr_l:
            opt.zero_grad()
            loss = lossf(net(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        sched.step()
        # selection uses the deterministic pass; sampling is for the test set
        Pv, Yv = mc_forward(net, va_l, dev, 1)
        acc = float((Pv[0].argmax(1) == Yv).mean())
        print(f"epoch {ep:3d} loss {tot/max(n,1):.4f}  val acc {acc:.4f}")
        if acc > best_va:
            best_va = acc
            torch.save({"state_dict": net.state_dict(), "epoch": ep,
                        "val_acc": acc, "args": vars(a)}, best)

    net.load_state_dict(torch.load(best, map_location=dev)["state_dict"])
    P, Y = mc_forward(net, te_l, dev, a.passes)
    u = decompose(P)
    s = scores(u["p"], Y)
    print(f"\nTEST {json.dumps(s)}")

    keep = ["uid", "city", "pano_id", "y", "split_adj", "unsure_adj",
            "contested", "unclear"] + COVARIATES
    out = te[[c for c in keep if c in te.columns]].copy()
    out["p_incorrect"] = u["p"][:, 1]
    out["uncertainty"] = u["total"]          # what roleb.py reads by default
    out["u_total"] = u["total"]
    out["u_aleatoric"] = u["aleatoric"]
    out["u_epistemic"] = u["epistemic"]
    out.to_parquet(run / "test_preds.parquet", index=False)
    (run / "scores.json").write_text(json.dumps(
        {"test": s, "val_acc": best_va, "n_test": len(te),
         "passes": a.passes}, indent=2))
    (run / "DONE").write_text("")
    print(f"wrote {run/'test_preds.parquet'}")


if __name__ == "__main__":
    main()
