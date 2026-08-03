"""Train a ProtoPNet intent classifier on PIE pedestrian crops.

Wraps the upstream ProtoPNet training loop without modifying its source. Three
things differ from the CUB recipe and each matters for the tornness experiments:

1. num_classes=2 with few prototypes per class. CUB uses 2000 prototypes for 200
   classes; here the whole point is that co-activation across two class-specific
   prototype sets is readable, so we default to 20 per class.
2. Class imbalance. PIE intent is ~75/25. Upstream `_train_or_test` calls plain
   `F.cross_entropy`, so we balance with a WeightedRandomSampler rather than
   patching their loop -- the sampler leaves ProtoPNet's source untouched.
3. Split. `--fold k` uses the video-grouped k-fold from pie_dataset (no scene
   spans train/test); `--split official` reproduces the PIE paper's split.

Usage:
    python src/train_pie.py --fold 0 --arch resnet34 --epochs 40
    python src/train_pie.py --split official --arch resnet34 --epochs 40
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "src"))

import model as ppnet_model          # noqa: E402
import push as ppnet_push            # noqa: E402
import train_and_test as tnt         # noqa: E402
from helpers import makedir          # noqa: E402
from log import create_logger         # noqa: E402
from preprocess import mean, std, preprocess_input_function  # noqa: E402

from pie_dataset import (PIECropDataset, kfold_assign, load_store)  # noqa: E402

IMG_SIZE = 224


class TwoTuple(torch.utils.data.Dataset):
    """ProtoPNet's loops unpack `(image, label)`; our dataset yields metadata too."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label, _ = self.ds[i]
        return img, label


def build_loaders(args):
    df = load_store(crops=args.crops, manifest=args.manifest)

    # One crop per (ped, frame) is far more data than we need per epoch and is
    # heavily autocorrelated: adjacent frames of the same pedestrian are nearly
    # identical. Subsample within the experiment window.
    df = df[df.in_exp_window]
    if args.frame_stride > 1:
        df = df[df.frame % args.frame_stride == 0]
    if args.min_bbox_h > 0:
        df = df[df.bbox_h >= args.min_bbox_h]

    if args.split == "official":
        tr = df[df.split == "train"]
        te = df[df.split == "test"]
        tag = "official"
    else:
        peds = pd.read_parquet(args.manifest / "peds.parquet")
        folds = kfold_assign(peds, n_folds=args.n_folds, group="video", seed=0)
        df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
        tr = df[df.fold != args.fold]
        te = df[df.fold == args.fold]
        tag = f"fold{args.fold}"

    norm = T.Normalize(mean=mean, std=std)
    # Modest augmentation only: prototypes get projected onto real patches, so
    # aggressive warping would make the exemplars unreadable as evidence.
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(), norm,
    ])
    eval_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])
    push_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])  # [0,1]

    train_ds = TwoTuple(PIECropDataset(tr, args.crops, train_tf))
    push_ds = TwoTuple(PIECropDataset(tr, args.crops, push_tf))
    test_ds = TwoTuple(PIECropDataset(te, args.crops, eval_tf))

    # balance 75/25 intent by resampling
    y = tr.intent_binary.to_numpy()
    w = np.where(y == 1, 1.0 / max((y == 1).sum(), 1), 1.0 / max((y == 0).sum(), 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(w, dtype=torch.double), num_samples=len(w), replacement=True)

    # Training on 224px crops is usually input-bound, not GPU-bound: the GPU
    # finishes a batch faster than the workers can decode the next one.
    # persistent_workers avoids respawning them every epoch, prefetch_factor
    # keeps a queue ahead of the GPU.
    def mk(ds, bs, **kw):
        extra = {}
        if args.workers > 0:
            extra = {"persistent_workers": True, "prefetch_factor": 4}
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, num_workers=args.workers, pin_memory=True,
            **extra, **kw)
    return (mk(train_ds, args.batch, sampler=sampler),
            mk(push_ds, args.push_batch, shuffle=False),
            mk(test_ds, args.batch, shuffle=False),
            tag, len(tr), len(te), tr.ped_id.nunique(), te.ped_id.nunique())


def save_ckpt(path: Path, **state) -> None:
    """Atomic checkpoint write: temp file then rename.

    A kill signal partway through a multi-hundred-MB torch.save leaves a
    truncated file that fails to load. Renaming is atomic on POSIX, so the
    checkpoint at `path` is always either the previous good one or the new one.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def load_rng(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--split", choices=["kfold", "official"], default="kfold")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--protos-per-class", type=int, default=20)
    ap.add_argument("--proto-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warm-epochs", type=int, default=5)
    ap.add_argument("--push-start", type=int, default=10)
    ap.add_argument("--push-every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--push-batch", type=int, default=75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frame-stride", type=int, default=5,
                    help="keep 1 in N frames; adjacent frames are near-duplicates")
    ap.add_argument("--min-bbox-h", type=float, default=40.0)
    ap.add_argument("--gpu", default="0",
                    help="GPU id, or a comma list ('0,1') to DataParallel one "
                         "run across several GPUs -- scale --batch with it")
    ap.add_argument("--tf32", action="store_true",
                    help="enable TF32 matmuls (Ampere+); faster, slightly lower "
                         "mantissa precision")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint and start from scratch")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Crop sizes are fixed (224x224), so cuDNN can pick its best kernels once
    # and reuse them. Free speedup; only hurts with varying input shapes.
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        # Ampere+ (A100/A6000/RTX30xx and newer). Large matmul speedup at
        # slightly reduced mantissa precision.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    (train_loader, push_loader, test_loader,
     tag, n_tr, n_te, p_tr, p_te) = build_loaders(args)

    run_dir = args.out / f"{args.arch}_{tag}"
    makedir(str(run_dir))
    makedir(str(run_dir / "img"))

    # Already finished? Nothing to do -- makes re-running the whole script safe.
    if (run_dir / "DONE").exists() and not args.no_resume:
        print(f"{run_dir} already complete (DONE marker); skipping. "
              f"Use --no-resume to retrain.")
        return

    log, logclose = create_logger(log_filename=str(run_dir / "train.log"))
    log(f"split={args.split} {tag} | train {n_tr:,} crops / {p_tr} peds "
        f"| test {n_te:,} crops / {p_te} peds")

    n_proto = 2 * args.protos_per_class
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=args.arch, pretrained=True, img_size=IMG_SIZE,
        prototype_shape=(n_proto, args.proto_dim, 1, 1), num_classes=2,
        prototype_activation_function="log", add_on_layers_type="regular")
    ppnet = ppnet.cuda()
    ppnet_par = torch.nn.DataParallel(ppnet)
    log(f"prototypes: {n_proto} ({args.protos_per_class}/class), dim {args.proto_dim}")

    joint_opt = torch.optim.Adam([
        {"params": ppnet.features.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
        {"params": ppnet.add_on_layers.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
        {"params": ppnet.prototype_vectors, "lr": 3e-3},
    ])
    joint_sched = torch.optim.lr_scheduler.StepLR(joint_opt, step_size=5, gamma=0.1)
    warm_opt = torch.optim.Adam([
        {"params": ppnet.add_on_layers.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
        {"params": ppnet.prototype_vectors, "lr": 3e-3},
    ])
    last_opt = torch.optim.Adam(
        [{"params": ppnet.last_layer.parameters(), "lr": 1e-4}])

    coefs = {"crs_ent": 1, "clst": 0.8, "sep": -0.08, "l1": 1e-4}
    push_epochs = [e for e in range(args.epochs)
                   if e >= args.push_start and e % args.push_every == 0]
    best = 0.0
    start_epoch = 0

    ckpt_path = run_dir / "ckpt.pth"
    if ckpt_path.exists() and not args.no_resume:
        st = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        ppnet.load_state_dict(st["model"])
        warm_opt.load_state_dict(st["warm_opt"])
        joint_opt.load_state_dict(st["joint_opt"])
        last_opt.load_state_dict(st["last_opt"])
        joint_sched.load_state_dict(st["joint_sched"])
        best = st["best"]
        start_epoch = st["epoch"] + 1
        try:
            load_rng(st["rng"])
        except Exception as e:      # different GPU count, say -- not fatal
            log(f"could not restore RNG state ({e}); continuing")
        log(f"RESUMED from {ckpt_path} at epoch {start_epoch} (best {best:.4f})")
        if start_epoch >= args.epochs:
            log("checkpoint is already at the final epoch; nothing to do")
            (run_dir / "DONE").touch()
            logclose()
            return
    else:
        log("starting from scratch")

    for epoch in range(start_epoch, args.epochs):
        log(f"epoch {epoch}")
        if epoch < args.warm_epochs:
            tnt.warm_only(model=ppnet_par, log=log)
            tnt.train(model=ppnet_par, dataloader=train_loader,
                      optimizer=warm_opt, class_specific=True, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_par, log=log)
            tnt.train(model=ppnet_par, dataloader=train_loader,
                      optimizer=joint_opt, class_specific=True, coefs=coefs, log=log)
            joint_sched.step()

        acc = tnt.test(model=ppnet_par, dataloader=test_loader,
                       class_specific=True, log=log)

        if epoch in push_epochs:
            # projects every prototype onto its nearest real training patch --
            # after this the exemplars are actual pedestrians we can show
            ppnet_push.push_prototypes(
                push_loader, prototype_network_parallel=ppnet_par,
                class_specific=True,
                preprocess_input_function=preprocess_input_function,
                root_dir_for_saving_prototypes=str(run_dir / "img"),
                epoch_number=epoch,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                save_prototype_class_identity=True, log=log)
            acc = tnt.test(model=ppnet_par, dataloader=test_loader,
                           class_specific=True, log=log)

            tnt.last_only(model=ppnet_par, log=log)
            for _ in range(20):
                tnt.train(model=ppnet_par, dataloader=train_loader,
                          optimizer=last_opt, class_specific=True,
                          coefs=coefs, log=log)
                acc = tnt.test(model=ppnet_par, dataloader=test_loader,
                               class_specific=True, log=log)
            if acc > best:
                best = acc
                # tornness.py loads the whole pickled module, so keep saving it
                torch.save(ppnet, run_dir / "best.pth")
                log(f"  saved best.pth (acc {acc:.4f})")

        # Checkpoint every epoch, after any push cycle has completed. Worst case
        # an interrupt costs one epoch, not the whole run.
        save_ckpt(ckpt_path, epoch=epoch, model=ppnet.state_dict(),
                  warm_opt=warm_opt.state_dict(), joint_opt=joint_opt.state_dict(),
                  last_opt=last_opt.state_dict(),
                  joint_sched=joint_sched.state_dict(), best=best, rng=rng_state())
        log(f"  checkpoint saved (epoch {epoch}, best {best:.4f})")

    torch.save(ppnet, run_dir / "final.pth")
    (run_dir / "DONE").touch()
    log(f"done. best test acc {best:.4f}. run dir {run_dir}")
    logclose()


if __name__ == "__main__":
    main()
