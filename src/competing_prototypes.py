"""Decision-grounded competing-prototype explanations for pedestrian intent.

The hand-off figure. For an ambiguous pedestrian, retrieve the real training
exemplar most responsible for each competing reading -- one "will cross", one
"won't cross" -- so a remote operator sees the evidence pulling each way instead
of a scalar that only says the model hesitated.

Grounded in the decision, not in raw similarity. Per-class max similarities are
near-uniform, so ranking prototypes by similarity surfaces high-activation
"magnet" prototypes unrelated to the decision. We use the prototype contributing
most to each class's logit:

    k*(c) = argmax_k  a_{c,k} * W_{c,k}

where a is the prototype activation vector and W the (fixed-sign) last layer.

Usage:
    python src/competing_prototypes.py --fold 0 --n 12
    python src/competing_prototypes.py --fold 0 --ped-id 3_5_120
"""

from __future__ import annotations

import argparse
import os
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

from preprocess import mean, std                                # noqa: E402
from pie_dataset import PIECropDataset, kfold_assign, load_store  # noqa: E402

IMG_SIZE = 224
CLASS_NAME = {0: "will NOT cross", 1: "WILL cross"}


def find_proto_dir(run_dir: Path) -> Path:
    """Latest epoch-N directory of pushed prototype images."""
    eps = sorted(run_dir.glob("img/epoch-*"),
                 key=lambda p: int(p.name.split("-")[1]))
    if not eps:
        raise SystemExit(f"no pushed prototypes under {run_dir}/img/")
    return eps[-1]


@torch.no_grad()
def explain(ppnet, batch, proto_class):
    """Return per-image logits, probs, activations and the competing pair."""
    logits, min_dist = ppnet(batch)
    acts = ppnet.distance_2_similarity(min_dist)             # (B, P)
    probs = F.softmax(logits, 1)
    W = ppnet.last_layer.weight                              # (C, P)
    # contribution of every prototype to every class logit
    contrib = acts.unsqueeze(1) * W.unsqueeze(0)             # (B, C, P)
    out = []
    for i in range(batch.size(0)):
        picks = {}
        for c in range(W.size(0)):
            m = (proto_class == c)
            idx = torch.arange(len(proto_class), device=acts.device)[m]
            best = idx[contrib[i, c, m].argmax()].item()
            picks[c] = {"proto": best,
                        "activation": acts[i, best].item(),
                        "contribution": contrib[i, c, best].item()}
        out.append({"probs": probs[i].cpu().numpy(), "picks": picks})
    return out


def render(fig_path, crop_img, picks, probs, meta, proto_dir, proto_meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1.15], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(crop_img)
    ax.set_title(f"Pedestrian {meta['ped_id']}\n"
                 f"frame {meta['frame']}  ({meta['action']}, {meta['look']})",
                 fontsize=9)
    ax.axis("off")

    # competing exemplars: class 1 first so "will cross" reads left-to-right
    for col, c in enumerate((1, 0), start=1):
        ax = fig.add_subplot(gs[0, col])
        p = proto_dir / f"prototype-img{picks[c]['proto']}.png"
        if p.exists():
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(.5, .5, "prototype image\nnot found", ha="center",
                    va="center", fontsize=8)
        pm = proto_meta.get(picks[c]["proto"], {})
        sub = (f"proto #{picks[c]['proto']}  act {picks[c]['activation']:.2f}\n"
               f"contrib {picks[c]['contribution']:+.2f}")
        if pm:
            sub += (f"\nfrom {pm.get('ped_id','?')} "
                    f"({pm.get('action','?')}, {pm.get('look','?')})")
        ax.set_title(f"{CLASS_NAME[c]}\n{sub}", fontsize=8.5,
                     color="tab:red" if c == 1 else "tab:blue")
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    ax.barh([1, 0], [probs[1], probs[0]], color=["tab:red", "tab:blue"])
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["will cross", "won't cross"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("model probability", fontsize=9)
    eps = 1e-12
    H = -(probs * np.log(np.clip(probs, eps, 1))).sum() / np.log(len(probs))
    ax.set_title(f"U = {H:.2f}\nhuman intent = {meta['intention_prob']:.2f}"
                 f"  (disagreement {meta['human_disagreement']:.2f})",
                 fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None,
                    help="run dir; default runs/resnet34_fold<fold>")
    ap.add_argument("--crops", type=Path, default=ROOT / "data/pie_crops")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/pie_manifest")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--ped-id", default=None)
    ap.add_argument("--mode", choices=["contested", "torn", "silent"],
                    default="contested",
                    help="contested: humans split (any model output); "
                         "torn: humans split AND model torn -- the hand-off "
                         "fires, this is the method figure; "
                         "silent: humans split BUT model near-certain -- the "
                         "failure figure")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    run = a.run or (ROOT / f"runs/resnet34_fold{a.fold}")
    proto_dir = find_proto_dir(run)
    a.out.mkdir(parents=True, exist_ok=True)

    df = load_store(crops=a.crops, manifest=a.manifest)
    peds = pd.read_parquet(a.manifest / "peds.parquet")
    folds = kfold_assign(peds, n_folds=5, group="video", seed=0)
    df = df.merge(folds[["ped_id", "fold"]], on="ped_id", how="inner")
    df = df[(df.fold == a.fold) & df.in_exp_window].reset_index(drop=True)

    ppnet = torch.load(run / "best.pth", map_location="cuda",
                       weights_only=False).cuda().eval()
    proto_class = ppnet.prototype_class_identity.argmax(1).cuda()

    # Which prototype came from which training pedestrian: push.py records the
    # source image index per prototype in bb<epoch>.npy, column 0.
    proto_meta = {}
    bb = sorted(proto_dir.glob("bb*.npy"))
    bb = [f for f in bb if "receptive_field" not in f.name]
    if bb:
        arr = np.load(bb[0])
        proto_meta = {j: {"src_index": int(arr[j, 0])} for j in range(len(arr))}

    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize(mean=mean, std=std)])
    raw_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))])

    # Pick the cases the paper is about: the model is torn AND humans were split.
    per = df.groupby("ped_id").agg(hd=("human_disagreement", "first"),
                                   ip=("intention_prob", "first"))
    if a.ped_id:
        chosen = [a.ped_id]
    else:
        # model-side tornness, averaged over the window humans actually saw
        pc = df.groupby("ped_id").p_cross_hint.mean() \
            if "p_cross_hint" in df else None
        split = per[per.ip.between(0.35, 0.65)]
        if a.mode == "contested":
            chosen = split.sort_values("hd", ascending=False).head(a.n).index.tolist()
        else:
            # Select on the SAME frame the figure renders -- the one closest to
            # the critical point. Selecting on the window mean picks pedestrians
            # whose average is 0.5 because the model flips between 0 and 1
            # across frames, not because it is uncertain on any single frame.
            tor = pd.read_parquet(run / f"tornness_fold{a.fold}.parquet")
            tor = tor[tor.in_exp_window].copy()
            key = tor.frames_to_critical.abs()
            tor = tor.loc[tor.assign(_k=key).groupby("ped_id")._k.idxmin()]
            tor = tor.set_index("ped_id").p_cross
            j = split.join(tor.rename("p"), how="inner").dropna()
            j["torn"] = (j.p - 0.5).abs()
            if a.mode == "torn":
                j = j.sort_values("torn")                 # most torn first
            else:
                j = j.sort_values("torn", ascending=False)  # most certain first
            chosen = j.head(a.n).index.tolist()
            print(f"  mode={a.mode}: model |p-0.5| range "
                  f"{j.torn.head(a.n).min():.3f}..{j.torn.head(a.n).max():.3f}")
    print(f"{len(chosen)} pedestrians; prototypes from {proto_dir}")

    from PIL import Image
    rows = []
    for pid in chosen:
        sub = df[df.ped_id == pid]
        # the frame closest to the critical point: the decision moment
        r = sub.iloc[(sub.frames_to_critical.abs()).argmin()]
        img = Image.open(a.crops / r.crop_path).convert("RGB")
        x = tf(img).unsqueeze(0).cuda()
        e = explain(ppnet, x, proto_class)[0]
        meta = {"ped_id": pid, "frame": int(r.frame), "action": r.action,
                "look": r.look, "intention_prob": float(r.intention_prob),
                "human_disagreement": float(r.human_disagreement)}
        dest = a.out / f"handoff_{a.mode}_{pid}.png"
        render(dest, raw_tf(img), e["picks"], e["probs"], meta,
               proto_dir, proto_meta)
        rows.append({**meta, "p_cross": float(e["probs"][1]),
                     "proto_cross": e["picks"][1]["proto"],
                     "proto_notcross": e["picks"][0]["proto"],
                     "fig": dest.name})
        print(f"  {pid}: p_cross={e['probs'][1]:.2f} "
              f"human={meta['intention_prob']:.2f} -> {dest.name}")

    pd.DataFrame(rows).to_csv(a.out / f"handoff_cases_{a.mode}.csv", index=False)
    print(f"\nwrote {len(rows)} figures + "
          f"{a.out/f'handoff_cases_{a.mode}.csv'}")
    print("Pick the clearest one or two as the paper's hero figures.")


if __name__ == "__main__":
    main()
