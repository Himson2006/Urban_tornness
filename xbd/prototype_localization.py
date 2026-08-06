"""Do the prototypes land on the building, or on the driveway next to it?

This is the check that should have been run on the pedestrian work months before
it was. There, prototypes fell inside the pedestrian's box 12% of the time
against a 25% chance rate -- *below* chance -- which meant the model's
explanations pointed at road surface while its accuracy looked fine. Every
tornness statistic computed on top of that was describing the curb.

Two numbers matter, and they are different questions:

  * **all prototypes**: across every test building, where does each prototype's
    activation peak land? Diffuse but unbiased.
  * **the shown prototype**: for each building, the prototype the explanation
    would actually display -- argmax over a_k * W_ck, the one that moves the
    predicted class's logit, not the one that merely lights up brightest. This
    is the number a reader of the explanation experiences.

Both are compared against a per-building chance rate rather than a constant.
The crop is `crop_scale` times the building's side, clamped at image edges, so
the building's share of the crop varies building to building and a single
baseline would be wrong.

Localisation is drawn as **bounding boxes, not heatmaps**: green for the
building's own extent, amber for the prototype's box. Whether the amber box sits
on the building is a yes/no question, and a heatmap invites you to squint at it
until the answer is the one you wanted.

Usage:
    python xbd/prototype_localization.py --run xbd/runs/middle_pair_scene_f0_resnet34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ProtoPNet"))
sys.path.insert(0, str(ROOT / "xbd"))

import model as ppnet_model            # noqa: E402
from dataset import TASKS, PairedCropDataset, assign_folds, load_meta  # noqa: E402

GREEN = (80, 220, 80)
AMBER = (40, 170, 250)      # BGR
GREY = (170, 170, 170)


def act_bbox(act: np.ndarray, size: int, pct: float = 95.0):
    """Bounding box of the top-`pct` region of an upsampled activation map.

    ProtoPNet's own visualisation takes the high-activation region rather than
    the single peak cell, which is the honest extent: one cell of a 7x7 grid
    understates the receptive field that actually produced the score.
    """
    import cv2

    up = cv2.resize(act.astype(np.float32), (size, size),
                    interpolation=cv2.INTER_CUBIC)
    thr = np.percentile(up, pct)
    ys, xs = np.where(up >= thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def building_box(row, size: int) -> tuple[int, int, int, int]:
    """The building's own extent inside the crop, in resized pixels.

    The crop is centred on the polygon centroid with a half-side of
    crop_scale/2 building sides, so the building spans px_side/crop_w of the
    width. Edge-clamped crops are not square, which is why width and height are
    computed separately.
    """
    fx = min(row.px_side / max(row.crop_w, 1), 1.0)
    fy = min(row.px_side / max(row.crop_h, 1), 1.0)
    x1 = int(round(size * (0.5 - fx / 2)))
    x2 = int(round(size * (0.5 + fx / 2)))
    y1 = int(round(size * (0.5 - fy / 2)))
    y2 = int(round(size * (0.5 + fy / 2)))
    return x1, y1, x2, y2


def cell_centres(grid: int, size: int) -> np.ndarray:
    """Where each activation cell's box lands, through the real code path.

    The chance rate is *not* the building's share of the crop area. Activations
    live on a coarse grid -- 7x7 for a 224-pixel input -- so a peak can only
    occupy one of 49 positions, and cubic upsampling shifts the resulting box
    off the cell's analytic centre. Comparing against the area fraction would
    have understated chance by five points here and made a null look like a
    finding. Measuring it through `act_bbox` itself removes the guesswork.
    """
    out = []
    for r in range(grid):
        for c in range(grid):
            z = np.zeros((grid, grid), np.float32)
            z[r, c] = 1.0
            b = act_bbox(z, size)
            out.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return np.array(out)


def chance_rate(bld, centres: np.ndarray) -> float:
    """Fraction of reachable peak positions that fall on the building."""
    x, y = centres[:, 0], centres[:, 1]
    return float(np.mean((x >= bld[0]) & (x <= bld[2]) &
                         (y >= bld[1]) & (y <= bld[3])))


def centre_inside(box, bld) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return bld[0] <= cx <= bld[2] and bld[1] <= cy <= bld[3]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


@torch.no_grad()
def collect(ppnet, ds, dev, batch, workers):
    """Similarity maps, per-prototype peaks and the decision-grounded choice."""
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    ident = ppnet.prototype_class_identity.cpu().numpy()
    W = ppnet.last_layer.weight.detach().cpu().numpy()
    ppnet.eval()
    maps, best_proto, preds = [], [], []
    for x, _, _ in dl:
        x = x.to(dev)
        _, dist = ppnet.push_forward(x)
        sim = ppnet.distance_2_similarity(dist)          # (B, P, H, W)
        amax = sim.amax(dim=(2, 3))                      # (B, P)
        logits, _ = ppnet(x)
        c = logits.argmax(1).cpu().numpy()
        a = amax.cpu().numpy()
        # the prototype the explanation would show: largest contribution to the
        # predicted class's logit, not the largest raw similarity
        for i, ci in enumerate(c):
            m = ident[:, ci] > 0
            idx = np.where(m)[0]
            best_proto.append(int(idx[np.argmax(a[i, m] * W[ci, m])]))
        preds.extend(c.tolist())
        maps.append(sim.cpu().numpy().astype(np.float32))
    return np.concatenate(maps), np.array(best_proto), np.array(preds), ident


def draw(crop, bld, box, label, colour=AMBER):
    import cv2
    im = crop.copy()
    cv2.rectangle(im, bld[:2], bld[2:], GREEN, 1)
    if box is not None:
        cv2.rectangle(im, box[:2], box[2:], colour, 2)
    cv2.putText(im, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (255, 255, 255), 1)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--buildings", type=Path,
                    default=ROOT / "xbd/data/buildings.parquet")
    ap.add_argument("--radiometry", type=Path,
                    default=ROOT / "xbd/data/scene_radiometry.parquet")
    ap.add_argument("--limit", type=int, default=3000,
                    help="test buildings to score; 0 = all")
    ap.add_argument("--panel", type=int, default=12,
                    help="buildings drawn in the figure")
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import cv2

    cfg = json.loads((a.run / "final_test.json").read_text())
    st = torch.load(a.run / "best.pth", map_location="cpu", weights_only=False)
    A = st["args"]
    classes = TASKS[cfg["task"]]
    n_cls = len(classes)

    m = load_meta(a.crops, cfg["task"], A["min_side"], a.buildings,
                  Path(A.get("radiometry", a.radiometry)))
    m["fold"] = assign_folds(m, A["folds"], cfg["group"], A["seed"])
    te = m[m.fold == cfg["fold"]].reset_index(drop=True)
    if a.limit and len(te) > a.limit:
        te = te.sample(a.limit, random_state=0).reset_index(drop=True)
    ds = PairedCropDataset(te, a.crops, A["crop_px"], cfg["paired"], False,
                           align=A.get("align", True))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ppnet = ppnet_model.construct_PPNet(
        base_architecture=A["arch"], pretrained=False, img_size=A["img_size"],
        prototype_shape=(n_cls * A["protos_per_class"], A["proto_dim"], 1, 1),
        num_classes=n_cls, prototype_activation_function=A["proto_activation"],
        add_on_layers_type="regular")
    from train import inflate_conv1
    ppnet.features = inflate_conv1(ppnet.features, 2 if cfg["paired"] else 1)
    ppnet.load_state_dict(st["model"])
    ppnet = ppnet.to(dev)

    print(f"{a.run.name}: {cfg['task']} {classes}, test acc {cfg['test_acc']:.4f}"
          f" | {len(te):,} buildings, {n_cls * A['protos_per_class']} prototypes")

    sim, shown, preds, ident = collect(ppnet, ds, dev, a.batch, a.workers)
    S = A["crop_px"]
    blds = [building_box(r, S) for _, r in te.iterrows()]

    # per-building chance, measured over the positions a peak can actually take
    grid = sim.shape[-1]
    centres = cell_centres(grid, S)
    chance = np.array([chance_rate(b, centres) for b in blds])
    print(f"  activation grid {grid}x{grid} -> {grid * grid} reachable "
          f"positions; chance measured over those, not crop area")

    rows = []
    for i in range(len(te)):
        bl = blds[i]
        for j in range(sim.shape[1]):
            bx = act_bbox(sim[i, j], S)
            if bx is None:
                continue
            rows.append({"i": i, "proto": j, "shown": j == shown[i],
                         "inside": centre_inside(bx, bl), "iou": iou(bx, bl),
                         "chance": chance[i]})
    d = pd.DataFrame(rows)

    print(f"\n=== do prototypes land on the building? ===")
    print(f"  chance rate over reachable positions: {chance.mean():.1%}")
    if grid <= 8:
        print(f"  NOTE: a {grid}x{grid} grid is coarse. This detects gross")
        print(f"  mislocalisation -- the pedestrian failure was 12% against 25%")
        print(f"  -- but cannot resolve roof-versus-wall.")
    for nm, sub in [("all prototypes", d), ("the shown prototype", d[d.shown])]:
        obs = sub.inside.mean()
        lift = obs - sub.chance.mean()
        # binomial test against the mean per-building chance
        from scipy import stats
        p = stats.binomtest(int(sub.inside.sum()), len(sub),
                            sub.chance.mean()).pvalue
        print(f"  {nm:22s} inside {obs:6.1%}  chance {sub.chance.mean():5.1%}  "
              f"lift {lift:+.1%}  IoU {sub.iou.mean():.3f}  (p={p:.2g}, n={len(sub):,})")
    verdict = d[d.shown].inside.mean() - d[d.shown].chance.mean()
    print(f"\n  -> shown prototypes are {'ABOVE' if verdict > 0.05 else 'AT OR BELOW'}"
          f" chance on the building")
    if verdict <= 0.05:
        print("     the explanations point somewhere other than the building.")
        print("     This is the pedestrian failure; do not report tornness")
        print("     statistics built on these prototypes.")

    # per-class, since one class may localise and the other not
    te["shown"] = shown
    te["inside"] = [d[(d.i == i) & d.shown].inside.iloc[0]
                    if ((d.i == i) & d.shown).any() else np.nan
                    for i in range(len(te))]
    print(f"\n  {'class':16s} {'n':>7} {'inside':>8} {'chance':>8}")
    for c, g in te.groupby("damage"):
        if len(g) < 30:
            continue
        ch = chance[g.index.values].mean()
        print(f"  {c:16s} {len(g):7,} {g.inside.mean():7.1%} {ch:7.1%}")

    # ---- the figure: bounding boxes, not heatmaps -------------------------
    sel = te.sample(min(a.panel, len(te)), random_state=0).index.values
    blocks = []
    for i in sel:
        r = te.loc[i]
        bl = tuple(int(v * a.cell / S) for v in blds[i])
        bx = act_bbox(sim[i, shown[i]], S)
        bx = tuple(int(v * a.cell / S) for v in bx) if bx else None
        pre = cv2.resize(cv2.imread(str(a.crops / r.pre)), (a.cell, a.cell))
        post = cv2.resize(cv2.imread(str(a.crops / r.post)), (a.cell, a.cell))
        ok = centre_inside(bx, bl) if bx else False
        tag = f"{r.damage.split('-')[0]} p{shown[i]}"
        blocks.append(np.hstack([
            draw(pre, bl, None, "pre"),
            draw(post, bl, bx, f"{tag} {'ON' if ok else 'OFF'}",
                 AMBER if ok else (60, 60, 240))]))
        blocks.append(np.full((3, a.cell * 2, 3), 50, np.uint8))
    sheet = np.vstack(blocks)
    hdr = np.full((20, a.cell * 2, 3), 30, np.uint8)
    cv2.putText(hdr, "green = building   amber/red = prototype", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    out = a.run / "localization.png"
    cv2.imwrite(str(out), np.vstack([hdr, sheet]))
    d.to_parquet(a.run / "localization.parquet", index=False)
    print(f"\nwrote {out}\nwrote {a.run/'localization.parquet'}")


if __name__ == "__main__":
    main()
