"""Train a competing-prototype classifier on Project Sidewalk label crops.

The model answers "is this crowdsourced label correct?" -- the reviewer's
question -- and the point of the run is not the accuracy. It is whether the
model's *uncertainty* on a crop lines up with the disagreement real reviewers
showed on that same label, separately for the two kinds of disagreement
(handoff/DESIGN.md, Role B). Accuracy only has to be good enough that the
uncertainty means something.

Three things are arranged so the answer can come out negative:

  * `--task crosswalk` is the control. Only 7.7% of its labels are contested
    against 20.8% for nocurbramp, so tornness there should be markedly lower.
    If it is not, the measure is tracking image quality, not ambiguity.
  * Training excludes contested and unclear labels by default
    (`--all-train` disables it). Training on a label whose reviewers were split
    teaches the model the noise it is meant to be uncertain about, and would
    make Role B circular.
  * `--held-out-city` replaces the pano-grouped split with a whole city. These
    are not pooled runs and are reported apart.

Checkpoints are selected on validation. Test is scored exactly once, at the end.

Usage:
    python handoff/train.py --task crosswalk --dry-run
    python handoff/train.py --task nocurbramp
    python handoff/train.py --task nocurbramp --held-out-city chicago
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "handoff"))

import model as ppnet_model            # noqa: E402
import push as ppnet_push              # noqa: E402
import train_and_test as tnt           # noqa: E402
from log import create_logger          # noqa: E402

from dataset import (MEAN, STD, CropDataset, TwoTuple, UnNormalized,  # noqa: E402
                     class_weights, held_out_city, load_manifest)

CLASSES = ["correct", "incorrect"]


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def preprocess():
    m = torch.tensor(MEAN).view(1, -1, 1, 1)
    s = torch.tensor(STD).view(1, -1, 1, 1)

    def f(x):
        return (x - m.to(x.device)) / s.to(x.device)
    return f


def set_mode(ppnet, mode: str, train_features: bool):
    """Which parameters may move, in each phase.

    The backbone stays frozen unless asked. On the pedestrian work an
    unfrozen backbone collapsed the feature space onto a couple of corners,
    and prototypes stopped meaning anything.
    """
    for p in ppnet.add_on_layers.parameters():
        p.requires_grad = mode != "last"
    ppnet.prototype_vectors.requires_grad = mode != "last"
    for p in ppnet.last_layer.parameters():
        p.requires_grad = True
    for p in ppnet.features.parameters():
        p.requires_grad = train_features and mode == "joint"


def scores(P: np.ndarray, Y: np.ndarray) -> dict:
    """Accuracy alone is misleading when one class holds most of the split."""
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


def evaluate(ppnet, loader, dev):
    ppnet.eval()
    P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = ppnet(x.to(dev))
            P.append(torch.softmax(logits, 1).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    return float((P.argmax(1) == Y).mean()), P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="nocurbramp")
    ap.add_argument("--data", type=Path, default=ROOT / "handoff/data")
    ap.add_argument("--out", type=Path, default=ROOT / "handoff/runs")
    ap.add_argument("--held-out-city", default="")
    ap.add_argument("--all-train", action="store_true",
                    help="train on contested labels too -- makes Role B "
                         "circular; for the ablation only")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--protos-per-class", type=int, default=10)
    ap.add_argument("--proto-dim", type=int, default=128)
    ap.add_argument("--proto-activation", default="log")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warm-epochs", type=int, default=5)
    ap.add_argument("--push-start", type=int, default=10)
    ap.add_argument("--push-every", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--features-lr", type=float, default=0.0)
    ap.add_argument("--addon-lr", type=float, default=3e-3)
    ap.add_argument("--proto-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clst", type=float, default=0.2)
    ap.add_argument("--sep", type=float, default=-0.08)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=10)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data, model and phase masks on CPU")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_all(a.seed)
    crops = a.data / "hf" / a.task
    tag = (f"{a.task}_{'all' if a.all_train else 'consensus'}"
           f"{'_' + a.held_out_city if a.held_out_city else '_pano'}"
           f"_{a.arch}_s{a.seed}")
    run = a.out / tag
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print(f"{run} already complete; delete DONE to retrain")
        return
    log, logclose = create_logger(log_filename=str(run / "train.log"))

    m = load_manifest(a.data / f"manifest_{a.task}.parquet",
                      consensus_only=not a.all_train)
    if a.held_out_city:
        m = held_out_city(m, a.held_out_city)
    tr = m[m.split == "train"]
    va = m[m.split == "val"]
    te = m[m.split == "test"]

    log(f"task={a.task} {CLASSES} | consensus_only={not a.all_train} "
        f"| split={'city:' + a.held_out_city if a.held_out_city else 'pano'}")
    for nm, s in [("train", tr), ("val", va), ("test", te)]:
        log(f"  {nm:5s} {len(s):6,} crops  {s.pano_id.nunique():5,} panos  "
            f"incorrect-rate {s.y.mean():.3f}  "
            f"contested {int(s.contested.sum()):4,}  "
            f"unclear {int(s.unclear.sum()):4,}")
    log(f"  majority-class baseline on test: "
        f"{max(np.bincount(te.y, minlength=2)) / len(te):.4f}")
    log("checkpoint selected on VAL; test scored once at the end")

    # Role B needs contested labels in the evaluation set, or there is nothing
    # to correlate uncertainty against. Fail loudly rather than produce a run
    # whose headline analysis cannot be computed.
    n_eval = int(te.contested.sum() + te.unclear.sum())
    if n_eval < 100:
        msg = (f"only {n_eval} contested-or-unclear crops in test; Role B "
               f"needs a few hundred. Use a larger task or another held-out "
               f"city. (crosswalk is a Role A control and never clears this.)")
        if not a.dry_run:
            raise SystemExit(msg)
        log(f"WARNING: {msg}")

    ds_tr = CropDataset(tr, crops, a.img_size, augment=True)
    ds_va = CropDataset(va, crops, a.img_size, augment=False)
    ds_te = CropDataset(te, crops, a.img_size, augment=False)

    def dl(d, sh):
        return torch.utils.data.DataLoader(
            TwoTuple(d), batch_size=a.batch, shuffle=sh,
            num_workers=a.workers, pin_memory=True)

    va_l, te_l = dl(ds_va, False), dl(ds_te, False)

    cw = class_weights(tr)
    imbalance = float(cw.max() / cw.min())
    if a.balance and imbalance > 1.5:
        w = cw.numpy()[tr.y.values]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        tr_l = torch.utils.data.DataLoader(
            TwoTuple(ds_tr), batch_size=a.batch, sampler=sampler,
            num_workers=a.workers, pin_memory=True)
        log(f"class imbalance {imbalance:.1f}:1 -> balanced sampler on train")
    else:
        tr_l = dl(ds_tr, True)

    push_l = torch.utils.data.DataLoader(
        UnNormalized(CropDataset(tr, crops, a.img_size, augment=False)),
        batch_size=a.batch, shuffle=False, num_workers=a.workers)

    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=a.arch, pretrained=True, img_size=a.img_size,
        prototype_shape=(2 * a.protos_per_class, a.proto_dim, 1, 1),
        num_classes=2, prototype_activation_function=a.proto_activation,
        add_on_layers_type="regular").to(dev)
    log(f"device={dev}  prototypes={2 * a.protos_per_class}")

    if a.dry_run:
        x, y = next(iter(tr_l))
        logits, dists = ppnet(x.to(dev))
        log(f"dry run: batch {tuple(x.shape)} -> logits {tuple(logits.shape)}, "
            f"min-dists {tuple(dists.shape)}")
        for mode in ("warm", "joint", "last"):
            set_mode(ppnet, mode, a.features_lr > 0)
            n = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
            log(f"  {mode:5s} trainable params {n:,}")
        log("dry run OK")
        logclose()
        return

    warm_opt = torch.optim.Adam([
        {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
         "weight_decay": a.weight_decay},
        {"params": ppnet.prototype_vectors, "lr": a.proto_lr}])
    joint_opt = torch.optim.Adam([
        {"params": ppnet.features.parameters(), "lr": a.features_lr,
         "weight_decay": a.weight_decay},
        {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
         "weight_decay": a.weight_decay},
        {"params": ppnet.prototype_vectors, "lr": a.proto_lr}])
    last_opt = torch.optim.Adam(
        [{"params": ppnet.last_layer.parameters(), "lr": 1e-4}])
    sched = torch.optim.lr_scheduler.StepLR(joint_opt, a.lr_step, a.lr_gamma)
    coefs = {"crs_ent": 1, "clst": a.clst, "sep": a.sep, "l1": a.l1}

    best_va, best = -1.0, run / "best.pth"
    for ep in range(a.epochs):
        mode = "warm" if ep < a.warm_epochs else "joint"
        set_mode(ppnet, mode, a.features_lr > 0)
        opt = warm_opt if mode == "warm" else joint_opt
        tnt.train(model=ppnet, dataloader=tr_l, optimizer=opt,
                  class_specific=True, coefs=coefs, log=log)
        if mode == "joint":
            sched.step()

        do_push = ep >= a.push_start and (ep - a.push_start) % a.push_every == 0
        if do_push:
            ppnet_push.push_prototypes(
                push_l, prototype_network_parallel=ppnet,
                class_specific=True, preprocess_input_function=preprocess(),
                root_dir_for_saving_prototypes=str(run / "protos"),
                epoch_number=ep, save_prototype_class_identity=True, log=log)
            set_mode(ppnet, "last", False)
            for _ in range(5):
                tnt.train(model=ppnet, dataloader=tr_l, optimizer=last_opt,
                          class_specific=True, coefs=coefs, log=log)

        acc, _, _ = evaluate(ppnet, va_l, dev)
        log(f"epoch {ep:3d} [{mode}{' +push' if do_push else ''}] val acc {acc:.4f}")
        if acc > best_va:
            best_va = acc
            torch.save({"state_dict": ppnet.state_dict(), "epoch": ep,
                        "val_acc": acc, "args": vars(a)}, best)

    ppnet.load_state_dict(torch.load(best, map_location=dev)["state_dict"])
    _, P, Y = evaluate(ppnet, te_l, dev)
    s = scores(P, Y)
    log(f"\nTEST {json.dumps(s)}")

    # Everything Role B needs, per crop, in test order.
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1) / np.log(2)
    out = te[["uid", "city", "pano_id", "y", "split_adj", "unsure_adj",
              "contested", "unclear", "severity", "n_val"]].copy()
    out["p_incorrect"] = P[:, 1]
    out["uncertainty"] = ent
    out.to_parquet(run / "test_preds.parquet", index=False)
    (run / "scores.json").write_text(json.dumps(
        {"test": s, "val_acc": best_va, "n_test": len(te)}, indent=2))
    (run / "DONE").write_text("")
    log(f"wrote {run/'test_preds.parquet'}")
    logclose()


if __name__ == "__main__":
    main()
