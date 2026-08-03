"""Train a ProtoPNet on CIFAR-10 for the shape experiment.

Trains on the CIFAR-10 *train* split; the *test* split is where CIFAR-10H's
~51 human votes per image live, so that is what tornness gets extracted on.
A 5,000-image slice of train is held out for checkpoint selection, so test is
never used to choose anything.

Two deliberate differences from the PIE model:

  10 classes, so co-activation generalises from "the two classes' max
  similarities" to "the top-2 classes' max similarities" -- and the model then
  names *which* two readings compete, which can be checked against the humans'
  own top-2 pair. That test does not exist in binary PIE.

  32x32 upsampled to 224. A prototype's receptive field still covers a large
  fraction of the image, so this arm carries the co-activation geometry claim,
  not a strong visual exemplar story.

Usage:
    python src/cifar_train.py --gpu 0 --epochs 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "src"))

import model as ppnet_model          # noqa: E402
import push as ppnet_push            # noqa: E402
import train_and_test as tnt         # noqa: E402
from helpers import makedir          # noqa: E402
from log import create_logger        # noqa: E402
from preprocess import mean, std, preprocess_input_function  # noqa: E402
from train_pie import append_metrics, save_ckpt, rng_state, load_rng  # noqa: E402

IMG_SIZE = 224
N_CLASSES = 10


def build_loaders(args):
    norm = T.Normalize(mean=mean, std=std)
    train_tf = T.Compose([
        T.Resize(IMG_SIZE), T.RandomCrop(IMG_SIZE, padding=16),
        T.RandomHorizontalFlip(), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(IMG_SIZE), T.ToTensor(), norm])
    push_tf = T.Compose([T.Resize(IMG_SIZE), T.ToTensor()])       # [0,1]

    full = CIFAR10(args.data, train=True, download=True, transform=train_tf)
    full_push = CIFAR10(args.data, train=True, download=False, transform=push_tf)
    full_eval = CIFAR10(args.data, train=True, download=False, transform=eval_tf)
    test = CIFAR10(args.data, train=False, download=True, transform=eval_tf)

    # fixed val slice, so selection never touches the CIFAR-10H test images
    g = np.random.default_rng(0)
    perm = g.permutation(len(full))
    val_idx, tr_idx = perm[:args.n_val], perm[args.n_val:]
    sub = torch.utils.data.Subset
    mk = lambda ds, bs, sh: torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=sh, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0)

    return {
        "train": mk(sub(full, tr_idx), args.batch, True),
        "push": mk(sub(full_push, tr_idx), args.push_batch, False),
        "val": mk(sub(full_eval, val_idx), args.batch, False),
        "test": mk(test, args.batch, False),
        "n": (len(tr_idx), len(val_idx), len(test)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/cifar10")
    ap.add_argument("--out", type=Path, default=ROOT / "runs_cifar")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--protos-per-class", type=int, default=10)
    ap.add_argument("--proto-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warm-epochs", type=int, default=3)
    ap.add_argument("--push-start", type=int, default=10)
    ap.add_argument("--push-every", type=int, default=10)
    ap.add_argument("--last-iters", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--push-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-val", type=int, default=5000)
    # CIFAR-10 has 50k images vs PIE's 880 pedestrians, so light fine-tuning is
    # affordable here where it was catastrophic there. Check diag anyway.
    ap.add_argument("--features-lr", type=float, default=1e-5)
    ap.add_argument("--addon-lr", type=float, default=1e-3)
    ap.add_argument("--proto-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lr-step", type=int, default=10)
    ap.add_argument("--lr-gamma", type=float, default=0.5)
    ap.add_argument("--clst", type=float, default=0.2)
    ap.add_argument("--sep", type=float, default=-0.08)
    ap.add_argument("--l1", type=float, default=1e-4)
    ap.add_argument("--proto-activation", default="log", choices=["log", "linear"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    L = build_loaders(args)
    run_dir = args.out / f"{args.arch}_cifar10"
    makedir(str(run_dir))
    makedir(str(run_dir / "img"))
    if (run_dir / "DONE").exists() and not args.no_resume:
        print(f"{run_dir} already complete; --no-resume to retrain")
        return

    log, logclose = create_logger(log_filename=str(run_dir / "train.log"))
    log(f"train {L['n'][0]:,} | val {L['n'][1]:,} | test {L['n'][2]:,} "
        f"(test = CIFAR-10H images)")

    n_proto = N_CLASSES * args.protos_per_class
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=args.arch, pretrained=True, img_size=IMG_SIZE,
        prototype_shape=(n_proto, args.proto_dim, 1, 1), num_classes=N_CLASSES,
        prototype_activation_function=args.proto_activation,
        add_on_layers_type="regular").cuda()
    par = torch.nn.DataParallel(ppnet)
    log(f"prototypes: {n_proto} ({args.protos_per_class}/class)")

    groups = [
        {"params": ppnet.add_on_layers.parameters(), "lr": args.addon_lr,
         "weight_decay": args.weight_decay},
        {"params": ppnet.prototype_vectors, "lr": args.proto_lr},
    ]
    if args.features_lr > 0:
        groups.insert(0, {"params": ppnet.features.parameters(),
                          "lr": args.features_lr,
                          "weight_decay": args.weight_decay})
    else:
        for p in ppnet.features.parameters():
            p.requires_grad = False
        log("backbone FROZEN")
    joint_opt = torch.optim.Adam(groups)
    joint_sched = torch.optim.lr_scheduler.StepLR(
        joint_opt, step_size=args.lr_step, gamma=args.lr_gamma)
    warm_opt = torch.optim.Adam([
        {"params": ppnet.add_on_layers.parameters(), "lr": args.addon_lr,
         "weight_decay": args.weight_decay},
        {"params": ppnet.prototype_vectors, "lr": args.proto_lr}])
    last_opt = torch.optim.Adam(
        [{"params": ppnet.last_layer.parameters(), "lr": 1e-4}])

    coefs = {"crs_ent": 1, "clst": args.clst, "sep": args.sep, "l1": args.l1}
    log(f"coefs={coefs} proto_act={args.proto_activation} "
        f"features_lr={args.features_lr}")

    best, start = 0.0, 0
    ck = run_dir / "ckpt.pth"
    if ck.exists() and not args.no_resume:
        st = torch.load(ck, map_location="cuda", weights_only=False)
        ppnet.load_state_dict(st["model"])
        warm_opt.load_state_dict(st["warm_opt"])
        joint_opt.load_state_dict(st["joint_opt"])
        last_opt.load_state_dict(st["last_opt"])
        joint_sched.load_state_dict(st["joint_sched"])
        best, start = st["best"], st["epoch"] + 1
        try:
            load_rng(st["rng"])
        except Exception as e:
            log(f"RNG restore skipped ({e})")
        log(f"RESUMED at epoch {start} (best val {best:.4f})")

    metrics = run_dir / "metrics.csv"
    push_epochs = [e for e in range(args.epochs)
                   if e >= args.push_start and e % args.push_every == 0]

    for epoch in range(start, args.epochs):
        log(f"epoch {epoch}")
        if epoch < args.warm_epochs:
            phase = "warm"
            tnt.warm_only(model=par, log=log)
            tr = tnt.train(model=par, dataloader=L["train"], optimizer=warm_opt,
                           class_specific=True, coefs=coefs, log=log)
        else:
            phase = "joint"
            tnt.joint(model=par, log=log)
            tr = tnt.train(model=par, dataloader=L["train"], optimizer=joint_opt,
                           class_specific=True, coefs=coefs, log=log)
            joint_sched.step()
        acc = tnt.test(model=par, dataloader=L["val"], class_specific=True, log=log)
        append_metrics(metrics, {"epoch": epoch, "phase": phase, "train_acc": tr,
                                 "val_acc": acc, "pushed": 0,
                                 "best_val": max(best, acc)})

        if epoch in push_epochs:
            ppnet_push.push_prototypes(
                L["push"], prototype_network_parallel=par, class_specific=True,
                preprocess_input_function=preprocess_input_function,
                root_dir_for_saving_prototypes=str(run_dir / "img"),
                epoch_number=epoch,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                save_prototype_class_identity=True, log=log)
            tnt.last_only(model=par, log=log)
            for li in range(args.last_iters):
                tr = tnt.train(model=par, dataloader=L["train"],
                               optimizer=last_opt, class_specific=True,
                               coefs=coefs, log=log)
                acc = tnt.test(model=par, dataloader=L["val"],
                               class_specific=True, log=log)
                append_metrics(metrics, {"epoch": epoch, "phase": f"last_{li}",
                                         "train_acc": tr, "val_acc": acc,
                                         "pushed": 1, "best_val": max(best, acc)})
                if acc > best:
                    best = acc
                    torch.save(ppnet, run_dir / "best.pth")
                    log(f"  saved best.pth (val {acc:.4f})")

        save_ckpt(ck, epoch=epoch, model=ppnet.state_dict(),
                  warm_opt=warm_opt.state_dict(), joint_opt=joint_opt.state_dict(),
                  last_opt=last_opt.state_dict(),
                  joint_sched=joint_sched.state_dict(), best=best, rng=rng_state())
        log(f"  checkpoint saved (epoch {epoch}, best val {best:.4f})")

    torch.save(ppnet, run_dir / "final.pth")
    if (run_dir / "best.pth").exists():
        sel = torch.load(run_dir / "best.pth", map_location="cuda",
                         weights_only=False).cuda()
        ta = tnt.test(model=torch.nn.DataParallel(sel), dataloader=L["test"],
                      class_specific=True, log=log)
        log(f"FINAL val-selected: val {best:.4f} | TEST {ta:.4f}")
        (run_dir / "final_test.txt").write_text(
            f"val_acc={best:.6f}\ntest_acc={ta:.6f}\n")
    (run_dir / "DONE").touch()
    log(f"done. best val {best:.4f}")
    logclose()


if __name__ == "__main__":
    main()
