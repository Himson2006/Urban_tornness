"""Train a competing-prototype classifier on paired pre/post building crops.

The question the run is set up to answer is not "how accurate is this" -- xBD
damage classification is a solved benchmark and beating it is not the point.
The question is whether a prototype model's *tornness* between two adjacent
damage classes is a measurable, meaningful quantity here, when it was not on
pedestrian crops.

Three things are deliberately arranged so the answer can come out negative:

  * The paired input is a flag. `--no-paired` runs the identical architecture on
    the post-disaster crop alone. If tornness on six channels is no better
    structured than tornness on three, the pairing bought nothing and the paper
    says so.
  * `--task extremes` runs no-damage vs destroyed. Co-activation should be much
    lower there than on the contested minor/major boundary. If it is not, the
    measure is tracking something other than ambiguity.
  * Folds are grouped by scene, and `--group disaster` holds out whole events.
    The pedestrian work's headline number came apart once scene leakage was
    closed, and that is a mistake worth not repeating.

Checkpoints are selected on validation. Test is scored exactly once, at the end.

Usage:
    python xbd/train.py --fold 0
    python xbd/train.py --fold 0 --no-paired          # ablation
    python xbd/train.py --fold 0 --task extremes      # control
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "xbd"))

import model as ppnet_model            # noqa: E402
import push as ppnet_push              # noqa: E402
import train_and_test as tnt           # noqa: E402
from log import create_logger           # noqa: E402

from dataset import (MEAN, STD, TASKS, PairedCropDataset, assign_folds,  # noqa: E402
                     class_weights, load_meta)


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
        n = 2 if ds.paired else 1
        self.m = torch.tensor(MEAN * n)[:, None, None]
        self.s = torch.tensor(STD * n)[:, None, None]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y, _ = self.ds[i]
        return (x * self.s + self.m).clamp(0, 1), y


def preprocess_stack(n_blocks: int):
    m = torch.tensor(MEAN * n_blocks).view(1, -1, 1, 1)
    s = torch.tensor(STD * n_blocks).view(1, -1, 1, 1)

    def f(x):
        return (x - m.to(x.device)) / s.to(x.device)
    return f


def inflate_conv1(features, n_blocks: int):
    """Widen the stem conv from 3 to 3*n channels, reusing pretrained weights.

    Each block's copy is the original kernel divided by n, so an input whose
    blocks are identical -- a building that did not change -- produces exactly
    the activations the pretrained single-image stem would have produced.
    Training starts somewhere sane instead of at random, and "no change" is the
    natural origin of the feature space.
    """
    if n_blocks == 1:
        return features
    old = features.conv1
    new = torch.nn.Conv2d(3 * n_blocks, old.out_channels,
                          kernel_size=old.kernel_size, stride=old.stride,
                          padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight.repeat(1, n_blocks, 1, 1) / n_blocks)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    features.conv1 = new
    return features


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def evaluate(ppnet, loader, dev) -> tuple[float, np.ndarray, np.ndarray]:
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
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--buildings", type=Path,
                    default=ROOT / "xbd/data/buildings.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/runs")
    ap.add_argument("--task", default="middle", choices=list(TASKS))
    ap.add_argument("--paired", action="store_true", default=True)
    ap.add_argument("--no-paired", dest="paired", action="store_false")
    ap.add_argument("--group", default="scene", choices=["scene", "disaster"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--min-side", type=float, default=24.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-px", type=int, default=96,
                    help="native crop is read at this size, then resized")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--protos-per-class", type=int, default=10)
    ap.add_argument("--proto-dim", type=int, default=128)
    ap.add_argument("--proto-activation", default="log")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warm-epochs", type=int, default=5)
    ap.add_argument("--push-start", type=int, default=10)
    ap.add_argument("--push-every", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    # conv1 has to move when the input is 6-channel; the rest of the backbone
    # stays put unless asked, which is what kept the pedestrian model from
    # collapsing its feature space onto two corners
    ap.add_argument("--features-lr", type=float, default=0.0)
    ap.add_argument("--stem-lr", type=float, default=1e-4)
    ap.add_argument("--addon-lr", type=float, default=3e-3)
    ap.add_argument("--proto-lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clst", type=float, default=0.2)
    ap.add_argument("--sep", type=float, default=-0.08)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=10)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_all(a.seed)
    classes = TASKS[a.task]
    n_cls = len(classes)
    n_blocks = 2 if a.paired else 1
    tag = (f"{a.task}_{'pair' if a.paired else 'post'}_{a.group}"
           f"_f{a.fold}_{a.arch}")
    run = a.out / tag
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print(f"{run} already complete; delete DONE to retrain")
        return
    log, logclose = create_logger(log_filename=str(run / "train.log"))

    m = load_meta(a.crops, a.task, a.min_side, a.buildings)
    m["fold"] = assign_folds(m, a.folds, a.group, a.seed)
    te = m[m.fold == a.fold]
    rest = m[m.fold != a.fold]
    # validation is a held-out fold too, so no group ever spans train and val
    va_fold = (a.fold + 1) % a.folds
    va, tr = rest[rest.fold == va_fold], rest[rest.fold != va_fold]

    log(f"task={a.task} {classes} | paired={a.paired} | group={a.group} "
        f"| fold {a.fold}/{a.folds}")
    for nm, s in [("train", tr), ("val", va), ("test", te)]:
        log(f"  {nm:5s} {len(s):7,} crops  {s[a.group].nunique():5,} {a.group}s"
            f"  pos-rate {s.label.mean():.3f}")
    log(f"  majority-class baseline on test: "
        f"{max(np.bincount(te.label, minlength=n_cls)) / len(te):.4f}")
    log("checkpoint selected on VAL; test scored once at the end")

    def mk(sub, augment):
        return PairedCropDataset(sub, a.crops, a.crop_px, a.paired, augment)

    ds_tr, ds_va, ds_te = mk(tr, True), mk(va, False), mk(te, False)
    dl = lambda d, sh: torch.utils.data.DataLoader(   # noqa: E731
        TwoTuple(d), batch_size=a.batch, shuffle=sh, num_workers=a.workers,
        pin_memory=True, drop_last=False)
    tr_l, va_l, te_l = dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False)
    push_l = torch.utils.data.DataLoader(
        UnNormalized(mk(tr, False)), batch_size=a.batch, shuffle=False,
        num_workers=a.workers, pin_memory=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_proto = n_cls * a.protos_per_class
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=a.arch, pretrained=True, img_size=a.img_size,
        prototype_shape=(n_proto, a.proto_dim, 1, 1), num_classes=n_cls,
        prototype_activation_function=a.proto_activation,
        add_on_layers_type="regular")
    ppnet.features = inflate_conv1(ppnet.features, n_blocks)
    ppnet = ppnet.to(dev)
    ppnet_par = torch.nn.DataParallel(ppnet) if dev == "cuda" else ppnet
    log(f"{n_proto} prototypes ({a.protos_per_class}/class), dim {a.proto_dim}"
        f" | input {3 * n_blocks} channels")

    stem = list(ppnet.features.conv1.parameters())
    stem_ids = {id(p) for p in stem}
    rest_feat = [p for p in ppnet.features.parameters() if id(p) not in stem_ids]
    if a.features_lr <= 0:
        for p in rest_feat:
            p.requires_grad = False
        log("backbone frozen except the stem conv "
            f"(stem_lr={a.stem_lr}) -- the stem must learn to read a pair")
    groups = [
        {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
         "weight_decay": a.weight_decay},
        {"params": [ppnet.prototype_vectors], "lr": a.proto_lr},
    ]
    if n_blocks > 1 or a.features_lr > 0:
        groups.append({"params": stem, "lr": a.stem_lr,
                       "weight_decay": a.weight_decay})
    if a.features_lr > 0:
        groups.append({"params": rest_feat, "lr": a.features_lr,
                       "weight_decay": a.weight_decay})

    joint_opt = torch.optim.Adam(groups)
    joint_sched = torch.optim.lr_scheduler.StepLR(
        joint_opt, step_size=a.lr_step, gamma=a.lr_gamma)
    warm_opt = torch.optim.Adam([
        {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
         "weight_decay": a.weight_decay},
        {"params": [ppnet.prototype_vectors], "lr": a.proto_lr},
    ] + ([{"params": stem, "lr": a.stem_lr}] if n_blocks > 1 else []))
    last_opt = torch.optim.Adam(
        [{"params": ppnet.last_layer.parameters(), "lr": 1e-4}])

    coefs = {"crs_ent": 1, "clst": a.clst, "sep": a.sep, "l1": a.l1}
    log(f"coefs={coefs} | class weights "
        f"{class_weights(tr, n_cls).numpy().round(3).tolist()}")
    push_epochs = [e for e in range(a.epochs)
                   if e >= a.push_start and e % a.push_every == 0]

    best, rows = 0.0, []
    for ep in range(a.epochs):
        if ep < a.warm_epochs:
            tnt.warm_only(model=ppnet_par, log=log)
            tnt.train(model=ppnet_par, dataloader=tr_l, optimizer=warm_opt,
                      class_specific=True, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_par, log=log)
            tnt.train(model=ppnet_par, dataloader=tr_l, optimizer=joint_opt,
                      class_specific=True, coefs=coefs, log=log)
            joint_sched.step()

        if ep in push_epochs:
            ppnet_push.push_prototypes(
                push_l, prototype_network_parallel=ppnet_par,
                class_specific=True, preprocess_input_function=preprocess_stack(n_blocks),
                prototype_layer_stride=1, root_dir_for_saving_prototypes=None,
                epoch_number=ep, log=log)
            tnt.last_only(model=ppnet_par, log=log)
            for _ in range(5):
                tnt.train(model=ppnet_par, dataloader=tr_l,
                          optimizer=last_opt, class_specific=True,
                          coefs=coefs, log=log)

        acc, _, _ = evaluate(ppnet, va_l, dev)
        rows.append({"epoch": ep, "val_acc": acc,
                     "lr": joint_opt.param_groups[0]["lr"]})
        pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
        log(f"epoch {ep:3d}  val_acc {acc:.4f}"
            f"{'  <- best' if acc > best else ''}")
        if acc > best:
            best = acc
            tmp = run / "best.pth.tmp"
            torch.save({"model": ppnet.state_dict(), "epoch": ep,
                        "val_acc": acc, "args": vars(a)}, tmp)
            os.replace(tmp, run / "best.pth")

    st = torch.load(run / "best.pth", map_location=dev, weights_only=False)
    ppnet.load_state_dict(st["model"])
    acc, P, Y = evaluate(ppnet, te_l, dev)
    maj = float(max(np.bincount(Y, minlength=n_cls)) / len(Y))
    log(f"\nFINAL  best epoch {st['epoch']}  val {st['val_acc']:.4f}  "
        f"test {acc:.4f}  (majority {maj:.4f})")
    np.savez(run / "test_probs.npz", probs=P, y=Y,
             uid=te.uid.values.astype(str))
    (run / "final_test.json").write_text(json.dumps(
        {"tag": tag, "task": a.task, "paired": a.paired, "group": a.group,
         "fold": a.fold, "classes": classes, "best_epoch": st["epoch"],
         "val_acc": st["val_acc"], "test_acc": acc, "majority": maj,
         "n_train": len(tr), "n_val": len(va), "n_test": len(te)}, indent=2))
    (run / "DONE").touch()
    logclose()


if __name__ == "__main__":
    main()
