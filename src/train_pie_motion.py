"""Train a multi-frame ProtoPNet on PIE, and score it the same way.

Identical to train_pie.py except the input is T stacked crops and the backbone
stem is inflated to match. Everything else -- folds, val selection, sampler,
push schedule, loss coefficients -- is held fixed so the comparison against the
single-frame model isolates the temporal input.

Push image saving is disabled: push.py writes prototype PNGs with plt.imsave,
which cannot handle a 3T-channel tensor. The self-activation .npy files are
still written, so prototype_localization.py works unchanged.

Usage:
    python src/train_pie_motion.py --fold 0 --n-frames 3 --gap 5 --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "src"))

import model as ppnet_model          # noqa: E402
import push as ppnet_push            # noqa: E402
import train_and_test as tnt         # noqa: E402
from helpers import makedir          # noqa: E402
from log import create_logger        # noqa: E402

from pie_dataset import kfold_assign, load_store                    # noqa: E402
from pie_motion import (PIESeqDataset, TwoTuple, UnNormalized,      # noqa: E402
                        build_lookup, inflate_conv1, preprocess_stack)
from train_pie import append_metrics, save_ckpt, rng_state, load_rng  # noqa: E402

IMG_SIZE = 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--out", type=Path, default=ROOT / "runs_motion")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--gap", type=int, default=5,
                    help="frames between stacked crops (~0.17 s at 30 fps)")
    ap.add_argument("--crop-scale", type=float, default=2.0)
    ap.add_argument("--protos-per-class", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warm-epochs", type=int, default=5)
    ap.add_argument("--push-start", type=int, default=5)
    ap.add_argument("--push-every", type=int, default=5)
    ap.add_argument("--last-iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--min-bbox-h", type=float, default=40.0)
    # the stem is new, so it must train even when the rest of the backbone is not
    ap.add_argument("--features-lr", type=float, default=0.0)
    ap.add_argument("--stem-lr", type=float, default=1e-4)
    ap.add_argument("--addon-lr", type=float, default=1e-3)
    ap.add_argument("--proto-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--clst", type=float, default=0.2)
    ap.add_argument("--sep", type=float, default=-0.08)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    torch.backends.cudnn.benchmark = True
    if a.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    store = load_store(crops=a.crops, manifest=a.manifest)
    lookup = build_lookup(store)          # every crop, so history is reachable

    df = store[store.in_exp_window & (store.bbox_h >= a.min_bbox_h)]
    peds = pd.read_parquet(a.manifest / "peds.parquet")
    folds = kfold_assign(peds, n_folds=a.n_folds, group="video", seed=0)
    df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
    vf = (a.fold + 1) % a.n_folds
    tr = df[(df.fold != a.fold) & (df.fold != vf)]
    va, te = df[df.fold == vf], df[df.fold == a.fold]
    if a.frame_stride > 1:
        tr = tr[tr.frame % a.frame_stride == 0]

    mk_ds = lambda d, train: PIESeqDataset(
        d, a.crops, lookup, a.n_frames, a.gap, IMG_SIZE, a.crop_scale, train)
    train_ds, val_ds, test_ds = mk_ds(tr, True), mk_ds(va, False), mk_ds(te, False)

    y = tr.intent_binary.to_numpy()
    w = np.where(y == 1, 1 / max((y == 1).sum(), 1), 1 / max((y == 0).sum(), 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
    dl = lambda ds, **kw: torch.utils.data.DataLoader(
        ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True,
        persistent_workers=a.workers > 0, **kw)

    train_loader = dl(TwoTuple(train_ds), sampler=sampler)
    push_loader = dl(UnNormalized(mk_ds(tr, False)), shuffle=False)
    val_loader = dl(TwoTuple(val_ds), shuffle=False)
    test_loader = dl(TwoTuple(test_ds), shuffle=False)

    run = a.out / f"{a.arch}_f{a.n_frames}g{a.gap}_fold{a.fold}"
    makedir(str(run))
    makedir(str(run / "img"))
    if (run / "DONE").exists() and not a.no_resume:
        print(f"{run} already complete")
        return
    log, logclose = create_logger(log_filename=str(run / "train.log"))
    log(f"MOTION n_frames={a.n_frames} gap={a.gap} "
        f"({a.n_frames * 3} input channels) | fold{a.fold}")
    log(f"train {len(tr):,} / {tr.ped_id.nunique()} peds | "
        f"val {len(va):,} / {va.ped_id.nunique()} | "
        f"test {len(te):,} / {te.ped_id.nunique()}")

    n_proto = 2 * a.protos_per_class
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=a.arch, pretrained=True, img_size=IMG_SIZE,
        prototype_shape=(n_proto, 128, 1, 1), num_classes=2,
        prototype_activation_function="log", add_on_layers_type="regular")
    ppnet.features = inflate_conv1(ppnet.features, a.n_frames)
    ppnet = ppnet.cuda()
    par = torch.nn.DataParallel(ppnet)
    log(f"stem inflated to {3*a.n_frames} channels; prototypes {n_proto}")

    stem = list(ppnet.features.conv1.parameters())
    rest = [p for n, p in ppnet.features.named_parameters()
            if not n.startswith("conv1")]
    if a.features_lr <= 0:
        for p in rest:
            p.requires_grad = False
        log("backbone frozen except the new stem")
    groups = [{"params": stem, "lr": a.stem_lr, "weight_decay": a.weight_decay},
              {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
               "weight_decay": a.weight_decay},
              {"params": ppnet.prototype_vectors, "lr": a.proto_lr}]
    if a.features_lr > 0:
        groups.append({"params": rest, "lr": a.features_lr,
                       "weight_decay": a.weight_decay})
    joint_opt = torch.optim.Adam(groups)
    joint_sched = torch.optim.lr_scheduler.StepLR(joint_opt, 10, 0.5)
    warm_opt = torch.optim.Adam([
        {"params": stem, "lr": a.stem_lr, "weight_decay": a.weight_decay},
        {"params": ppnet.add_on_layers.parameters(), "lr": a.addon_lr,
         "weight_decay": a.weight_decay},
        {"params": ppnet.prototype_vectors, "lr": a.proto_lr}])
    last_opt = torch.optim.Adam(
        [{"params": ppnet.last_layer.parameters(), "lr": 1e-4}])

    coefs = {"crs_ent": 1, "clst": a.clst, "sep": a.sep, "l1": a.l1}
    best, start = 0.0, 0
    ck = run / "ckpt.pth"
    if ck.exists() and not a.no_resume:
        st = torch.load(ck, map_location="cuda", weights_only=False)
        ppnet.load_state_dict(st["model"])
        joint_opt.load_state_dict(st["joint_opt"])
        warm_opt.load_state_dict(st["warm_opt"])
        last_opt.load_state_dict(st["last_opt"])
        joint_sched.load_state_dict(st["joint_sched"])
        best, start = st["best"], st["epoch"] + 1
        try:
            load_rng(st["rng"])
        except Exception as e:
            log(f"RNG restore skipped ({e})")
        log(f"RESUMED at epoch {start} (best val {best:.4f})")

    metrics = run / "metrics.csv"
    push_epochs = [e for e in range(a.epochs)
                   if e >= a.push_start and e % a.push_every == 0]
    pre = preprocess_stack(a.n_frames)

    for epoch in range(start, a.epochs):
        log(f"epoch {epoch}")
        if epoch < a.warm_epochs:
            phase = "warm"
            tnt.warm_only(model=par, log=log)
            for p in stem:
                p.requires_grad = True       # warm_only froze it; the stem is new
            trn = tnt.train(model=par, dataloader=train_loader,
                            optimizer=warm_opt, class_specific=True,
                            coefs=coefs, log=log)
        else:
            phase = "joint"
            tnt.joint(model=par, log=log)
            if a.features_lr <= 0:
                for p in rest:
                    p.requires_grad = False
            trn = tnt.train(model=par, dataloader=train_loader,
                            optimizer=joint_opt, class_specific=True,
                            coefs=coefs, log=log)
            joint_sched.step()
        acc = tnt.test(model=par, dataloader=val_loader,
                       class_specific=True, log=log)
        append_metrics(metrics, {"epoch": epoch, "phase": phase,
                                 "train_acc": trn, "val_acc": acc,
                                 "pushed": 0, "best_val": max(best, acc)})

        if epoch in push_epochs:
            ppnet_push.push_prototypes(
                push_loader, prototype_network_parallel=par,
                class_specific=True, preprocess_input_function=pre,
                root_dir_for_saving_prototypes=str(run / "img"),
                epoch_number=epoch,
                prototype_img_filename_prefix=None,      # PNGs cannot hold 3T ch
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                save_prototype_class_identity=True, log=log)
            tnt.last_only(model=par, log=log)
            for li in range(a.last_iters):
                trn = tnt.train(model=par, dataloader=train_loader,
                                optimizer=last_opt, class_specific=True,
                                coefs=coefs, log=log)
                acc = tnt.test(model=par, dataloader=val_loader,
                               class_specific=True, log=log)
                append_metrics(metrics, {"epoch": epoch, "phase": f"last_{li}",
                                         "train_acc": trn, "val_acc": acc,
                                         "pushed": 1, "best_val": max(best, acc)})
                if acc > best:
                    best = acc
                    torch.save(ppnet, run / "best.pth")
                    log(f"  saved best.pth (val {acc:.4f})")

        save_ckpt(ck, epoch=epoch, model=ppnet.state_dict(),
                  warm_opt=warm_opt.state_dict(), joint_opt=joint_opt.state_dict(),
                  last_opt=last_opt.state_dict(),
                  joint_sched=joint_sched.state_dict(), best=best, rng=rng_state())
        log(f"  checkpoint saved (epoch {epoch}, best val {best:.4f})")

    torch.save(ppnet, run / "final.pth")
    if (run / "best.pth").exists():
        sel = torch.load(run / "best.pth", map_location="cuda",
                         weights_only=False).cuda()
        ta = tnt.test(model=torch.nn.DataParallel(sel), dataloader=test_loader,
                      class_specific=True, log=log)
        log(f"FINAL val-selected: val {best:.4f} | TEST {ta:.4f}")
        (run / "final_test.txt").write_text(
            f"val_acc={best:.6f}\ntest_acc={ta:.6f}\n")
    (run / "DONE").touch()
    log(f"done. best val {best:.4f}")
    logclose()


if __name__ == "__main__":
    main()
