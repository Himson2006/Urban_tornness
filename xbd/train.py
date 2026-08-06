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


def set_mode(ppnet, mode: str, stem, rest_feat, train_rest: bool):
    """Which parameters may move, in each phase.

    ProtoPNet's own `warm_only`/`joint` cannot be used here. `warm_only` freezes
    every feature parameter, which would include the inflated stem -- the one
    weight that *has* to move, since it is the only part of the network that has
    never seen a 6-channel input. `joint` unfreezes the whole backbone, undoing
    the freeze that keeps the feature space from collapsing.
    """
    for p in ppnet.add_on_layers.parameters():
        p.requires_grad = mode != "last"
    ppnet.prototype_vectors.requires_grad = mode != "last"
    for p in ppnet.last_layer.parameters():
        p.requires_grad = True
    for p in stem:
        p.requires_grad = mode != "last"
    for p in rest_feat:
        p.requires_grad = train_rest and mode == "joint"


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def scores(P: np.ndarray, Y: np.ndarray, n_cls: int) -> dict:
    """Accuracy alone is misleading when a class holds 74% of the split.

    Michael is 74% minor and Harvey 74% major, so a model that has learned
    nothing can post a high accuracy, and one trained under balanced sampling
    can post a low one while discriminating perfectly well. Balanced accuracy
    and AUC say whether the classes are separable at all, independent of where
    the threshold happens to sit.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    pred = P.argmax(1)
    out = {"acc": float((pred == Y).mean()),
           "balanced_acc": float(balanced_accuracy_score(Y, pred)),
           "majority": float(max(np.bincount(Y, minlength=n_cls)) / len(Y))}
    try:
        out["auc"] = float(roc_auc_score(Y, P[:, 1]) if n_cls == 2
                           else roc_auc_score(Y, P, multi_class="ovr"))
    except ValueError:      # a split with one class present
        out["auc"] = float("nan")
    return out


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
    # This defaulted to 96 while --img-size said 224, and --img-size only feeds
    # construct_PPNet's bookkeeping: the tensor the network actually saw was
    # 96x96, giving a 3x3 prototype grid instead of 7x7. Nine reachable
    # positions makes "on the building" collapse to "in the centre cell". 0
    # means follow --img-size, which is what anyone reading these flags expects.
    ap.add_argument("--crop-px", type=int, default=0,
                    help="input size; 0 follows --img-size")
    ap.add_argument("--disaster", default="",
                    help="train and evaluate within one event. Pooled across "
                         "events, class composition differs enough that a model "
                         "scores well by learning which disaster it is looking "
                         "at while discriminating nothing within any of them.")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data, model and phase masks on CPU")
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--align", action="store_true", default=True)
    ap.add_argument("--no-align", dest="align", action="store_false",
                    help="skip radiometric alignment -- measures how much of "
                         "the result was the difference in satellite passes")
    ap.add_argument("--radiometry", type=Path,
                    default=ROOT / "xbd/data/scene_radiometry.parquet")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    seed_all(a.seed)
    if a.crop_px <= 0:
        a.crop_px = a.img_size
    classes = TASKS[a.task]
    n_cls = len(classes)
    n_blocks = 2 if a.paired else 1
    dis_tag = f"_{a.disaster.replace('-', '')}" if a.disaster else ""
    tag = (f"{a.task}_{'pair' if a.paired else 'post'}"
           f"{'' if a.align else '_raw'}{dis_tag}_{a.group}_f{a.fold}_{a.arch}")
    run = a.out / tag
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists():
        print(f"{run} already complete; delete DONE to retrain")
        return
    log, logclose = create_logger(log_filename=str(run / "train.log"))

    m = load_meta(a.crops, a.task, a.min_side, a.buildings, a.radiometry)
    if a.disaster:
        m = m[m.disaster == a.disaster].reset_index(drop=True)
        if len(m) < 500:
            raise SystemExit(f"only {len(m)} crops for {a.disaster!r}; "
                             f"too few to split five ways")
    m["fold"] = assign_folds(m, a.folds, a.group, a.seed)
    te = m[m.fold == a.fold]
    rest = m[m.fold != a.fold]
    # validation is a held-out fold too, so no group ever spans train and val
    va_fold = (a.fold + 1) % a.folds
    va, tr = rest[rest.fold == va_fold], rest[rest.fold != va_fold]

    if a.align and a.paired:
        cov = float(m.aligned.mean()) if "aligned" in m else 0.0
        if cov < 0.99:
            raise SystemExit(
                f"radiometric alignment covers only {cov:.1%} of crops. "
                f"Unmeasured scenes fall back to the identity, so training now "
                f"would mix aligned and unaligned pairs and the pre/post "
                f"difference would partly encode which scenes got measured.\n"
                f"  fix: python xbd/radiometry.py\n"
                f"  or:  --no-align, to train on raw captures deliberately")
        log(f"radiometric alignment: {cov:.1%} of crops")

    log(f"task={a.task} {classes} | paired={a.paired} | group={a.group} "
        f"| fold {a.fold}/{a.folds}")
    for nm, s in [("train", tr), ("val", va), ("test", te)]:
        log(f"  {nm:5s} {len(s):7,} crops  {s[a.group].nunique():5,} {a.group}s"
            f"  pos-rate {s.label.mean():.3f}")
    log(f"  majority-class baseline on test: "
        f"{max(np.bincount(te.label, minlength=n_cls)) / len(te):.4f}")
    log("checkpoint selected on VAL; test scored once at the end")

    def mk(sub, augment):
        return PairedCropDataset(sub, a.crops, a.crop_px, a.paired, augment,
                                 align=a.align)

    ds_tr, ds_va, ds_te = mk(tr, True), mk(va, False), mk(te, False)
    dl = lambda d, sh: torch.utils.data.DataLoader(   # noqa: E731
        TwoTuple(d), batch_size=a.batch, shuffle=sh, num_workers=a.workers,
        pin_memory=True, drop_last=False)
    va_l, te_l = dl(ds_va, False), dl(ds_te, False)

    # minor vs major is near-balanced, but no-damage vs destroyed is roughly
    # 16:1 in these scenes -- a model can hit 94% there by never predicting
    # destroyed, and its prototypes would mean nothing. Sample instead of
    # reweighting the loss, so ProtoPNet's clustering terms stay untouched.
    cw = class_weights(tr, n_cls)
    imbalance = float(cw.max() / cw.min())
    if a.balance and imbalance > 1.5:
        w = cw.numpy()[tr.label.values]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        tr_l = torch.utils.data.DataLoader(
            TwoTuple(ds_tr), batch_size=a.batch, sampler=sampler,
            num_workers=a.workers, pin_memory=True)
        log(f"class imbalance {imbalance:.1f}:1 -> balanced sampler on train")
    else:
        tr_l = dl(ds_tr, True)
        log(f"class imbalance {imbalance:.1f}:1 -> plain shuffling")
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
    if dev != "cuda" and not a.dry_run:
        raise SystemExit(
            "ProtoPNet's training loop calls .cuda() internally, so training "
            "needs a GPU. Use --dry-run to validate data, model and a forward "
            "pass on CPU before shipping to the server.")
    ppnet_par = torch.nn.DataParallel(ppnet)
    # Probe the real spatial grid rather than trusting --img-size. The gap
    # between the two is what made the first wave train at 96x96 with a 3x3
    # grid while every flag said 224.
    with torch.no_grad():
        probe = torch.zeros(1, 3 * n_blocks, a.crop_px, a.crop_px, device=dev)
        _, pd_ = ppnet.push_forward(probe)
        gh, gw = pd_.shape[-2:]
    log(f"{n_proto} prototypes ({a.protos_per_class}/class), dim {a.proto_dim}"
        f" | input {3 * n_blocks}x{a.crop_px}x{a.crop_px}"
        f" -> prototype grid {gh}x{gw} ({gh * gw} positions)")
    if gh * gw < 16:
        log(f"  WARNING: {gh}x{gw} is too coarse to localise within a building")

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
    log(f"coefs={coefs} | align={a.align}")
    push_epochs = [e for e in range(a.epochs)
                   if e >= a.push_start and e % a.push_every == 0]

    if a.dry_run:
        set_mode(ppnet, "warm", stem, rest_feat, a.features_lr > 0)
        x, y = next(iter(tr_l))
        logits, msim = ppnet(x)
        trainable = [n for n, p in ppnet.named_parameters() if p.requires_grad]
        log(f"DRY RUN: batch {tuple(x.shape)} -> logits {tuple(logits.shape)}, "
            f"similarities {tuple(msim.shape)}")
        log(f"  warm-phase trainable groups: "
            f"{sorted({n.split('.')[0] for n in trainable})}")
        log(f"  stem trainable in warm phase: "
            f"{all(p.requires_grad for p in stem)}  (must be True when paired)")
        set_mode(ppnet, "joint", stem, rest_feat, a.features_lr > 0)
        log(f"  backbone still frozen in joint phase: "
            f"{not any(p.requires_grad for p in rest_feat)}")
        px, py = next(iter(push_l))
        log(f"  push loader range [{px.min():.3f}, {px.max():.3f}] "
            f"(must be within [0,1])")
        log("  data, model and modes all check out; train on a GPU")
        logclose()
        return

    best, rows = 0.0, []
    train_rest = a.features_lr > 0
    for ep in range(a.epochs):
        if ep < a.warm_epochs:
            set_mode(ppnet, "warm", stem, rest_feat, train_rest)
            tnt.train(model=ppnet_par, dataloader=tr_l, optimizer=warm_opt,
                      class_specific=True, coefs=coefs, log=log)
        else:
            set_mode(ppnet, "joint", stem, rest_feat, train_rest)
            tnt.train(model=ppnet_par, dataloader=tr_l, optimizer=joint_opt,
                      class_specific=True, coefs=coefs, log=log)
            joint_sched.step()

        if ep in push_epochs:
            ppnet_push.push_prototypes(
                push_l, prototype_network_parallel=ppnet_par,
                class_specific=True, preprocess_input_function=preprocess_stack(n_blocks),
                prototype_layer_stride=1, root_dir_for_saving_prototypes=None,
                epoch_number=ep, log=log)
            set_mode(ppnet, "last", stem, rest_feat, train_rest)
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
    sc = scores(P, Y, n_cls)
    maj = sc["majority"]
    log(f"\nFINAL  best epoch {st['epoch']}  val {st['val_acc']:.4f}")
    log(f"  test acc {acc:.4f}  (majority {maj:.4f}, "
        f"lift {acc - maj:+.4f})")
    log(f"  balanced acc {sc['balanced_acc']:.4f}  (chance 0.5000)   "
        f"AUC {sc['auc']:.4f}  (chance 0.5000)")
    log("  AUC is the one to read when a class holds most of the split")

    # A pooled accuracy can hide a disaster with no signal in it. The raw
    # pre/post difference separates damage classes cleanly in Florence and
    # Michael and not at all in Harvey -- flood damage is inside the building,
    # not on the roof -- and Harvey is roughly half the contested set. Pooling
    # over events is how that disappears.
    by_dis = {}
    te_out = te.copy()
    te_out["_i"] = np.arange(len(te_out))
    log(f"\n  {'disaster':24s} {'n':>7} {'acc':>7} {'majority':>9} {'AUC':>7}")
    for dis, g in te_out.groupby("disaster"):
        if len(g) < 50:
            continue
        gi = g._i.values
        gs = scores(P[gi], Y[gi], n_cls)
        by_dis[str(dis)] = {"n": len(g), **gs}
        log(f"  {str(dis)[:24]:24s} {len(g):7,} {gs['acc']:7.4f} "
            f"{gs['majority']:9.4f} {gs['auc']:7.4f}")
    # the number the pooled figure hides: a model can beat the pooled majority
    # while losing to every within-event majority, by learning which event it
    # is looking at
    if by_dis:
        n_tot = sum(v["n"] for v in by_dis.values())
        wmaj = sum(v["majority"] * v["n"] for v in by_dis.values()) / n_tot
        wacc = sum(v["acc"] * v["n"] for v in by_dis.values()) / n_tot
        log(f"\n  within-event weighted:  acc {wacc:.4f}  vs majority "
            f"{wmaj:.4f}  ({wacc - wmaj:+.4f})")
        log(f"  pooled:                 acc {acc:.4f}  vs majority "
            f"{maj:.4f}  ({acc - maj:+.4f})")
        if (acc - maj) > 0 and (wacc - wmaj) <= 0:
            log("  -> the pooled lift is Simpson's paradox: skill on the")
            log("     mixture, none within any event.")
    np.savez(run / "test_probs.npz", probs=P, y=Y,
             uid=te.uid.values.astype(str))
    (run / "final_test.json").write_text(json.dumps(
        {"tag": tag, "task": a.task, "paired": a.paired, "group": a.group,
         "fold": a.fold, "classes": classes, "best_epoch": st["epoch"],
         "val_acc": st["val_acc"], "test_acc": acc, "majority": maj,
         "balanced_acc": sc["balanced_acc"], "auc": sc["auc"],
         "disaster": a.disaster, "crop_px": a.crop_px,
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "align": a.align, "by_disaster": by_dis}, indent=2))
    (run / "DONE").touch()
    logclose()


if __name__ == "__main__":
    main()
