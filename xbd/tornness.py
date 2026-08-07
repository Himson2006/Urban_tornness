"""Measure tornness, then try hard to explain it away.

Co-activation is the quantity of interest: for a building the model places near
the minor/major boundary, does it resemble *both* classes' prototypes strongly
(conflicting evidence) or *neither* (missing evidence)? Those are different
situations for a damage assessor -- one wants a second opinion, the other wants
a better image -- and telling them apart is the whole claim.

The measure is easy to compute and easy to fool, so the order of operations here
matters more than the statistics:

  1. **Saturation first.** On pedestrian crops `max_sim` took exactly two
     distinct values: the log activation's ceiling at distance zero, and one
     other. Co-activation had no variance and every correlation computed from it
     was meaningless. Nothing below is worth reading until this section says the
     measure varies.

  2. **Degradation second.** Small, oblique, badly-lit buildings are harder, and
     a measure that only recovers "this crop is poor" is a resolution detector
     wearing a nicer name. Every association is therefore reported twice: raw,
     and partialled on building size, off-nadir angle, ground sample distance
     and sun elevation.

  3. **The control task third.** Co-activation on no-damage vs destroyed should
     be markedly lower than on minor vs major. If it is not, the measure is not
     tracking ambiguity.

Only then: where is tornness concentrated, spatially and by event.

Usage:
    python xbd/tornness.py --run xbd/runs/middle_pair_scene_f0_resnet34
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
from dataset import (TASKS, PairedCropDataset, assign_folds,  # noqa: E402
                     load_meta)

DEGRADE = ["px_side", "off_nadir", "gsd", "sun_elev"]
LOG_CEILING = float(np.log(1 / 1e-4))       # 9.2103, activation at distance 0


def partial_spearman(x, y, covars: pd.DataFrame):
    """Spearman of x and y after linearly removing covars from both ranks."""
    from scipy import stats

    ok = np.isfinite(x) & np.isfinite(y)
    C = covars.copy()
    for c in C.columns:
        ok &= np.isfinite(C[c].values)
    if ok.sum() < 30:
        return np.nan, np.nan, int(ok.sum())
    rx = stats.rankdata(x[ok])
    ry = stats.rankdata(y[ok])
    Z = np.column_stack([stats.rankdata(C[c].values[ok]) for c in C.columns]
                        + [np.ones(ok.sum())])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p), int(ok.sum())


def morans_i(values: np.ndarray, lon: np.ndarray, lat: np.ndarray,
             k: int = 8) -> tuple[float, float]:
    """Moran's I over a k-nearest-neighbour graph, with a permutation p."""
    from sklearn.neighbors import NearestNeighbors

    ok = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
    v, xy = values[ok], np.column_stack([lon[ok], lat[ok]])
    n = len(v)
    if n < 50:
        return np.nan, np.nan
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    z = v - v.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return np.nan, np.nan

    # binary knn weights, so the weight total is n*k and I reduces to
    # sum_ij z_i z_j / (k * sum_i z_i^2)
    def stat(zz):
        return float((zz[:, None] * zz[idx]).sum() * n / (idx.size * denom))

    obs = stat(z)
    rng = np.random.default_rng(0)
    null = np.array([stat(rng.permutation(z)) for _ in range(999)])
    p = (1 + (np.abs(null) >= abs(obs)).sum()) / 1000
    return obs, float(p)


@torch.no_grad()
def similarities(ppnet, ds, dev, batch=128, workers=8):
    """Per-sample max similarity to each class's prototypes, plus probabilities.

    Prototype-to-class assignment comes from the model's own identity matrix,
    so this stays correct if the number of prototypes per class changes.
    """
    ident = ppnet.prototype_class_identity.cpu().numpy()   # (n_proto, n_class)
    W = ppnet.last_layer.weight.detach().cpu().numpy()     # (n_class, n_proto)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=workers, pin_memory=True)
    ppnet.eval()
    S, P, U = [], [], []
    for x, _, uid in dl:
        logits, msim = ppnet(x.to(dev))       # msim: (B, n_proto)
        S.append(msim.cpu().numpy())
        P.append(torch.softmax(logits, 1).cpu().numpy())
        U.extend(uid)
    return np.concatenate(S), np.concatenate(P), np.array(U), ident, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--buildings", type=Path,
                    default=ROOT / "xbd/data/buildings.parquet")
    ap.add_argument("--radiometry", type=Path,
                    default=ROOT / "xbd/data/scene_radiometry.parquet")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    cfg = json.loads((a.run / "final_test.json").read_text())
    st = torch.load(a.run / "best.pth", map_location="cpu", weights_only=False)
    A = st["args"]
    classes = TASKS[cfg["task"]]
    n_cls = len(classes)
    print(f"{a.run.name}: {cfg['task']} {classes} | paired={cfg['paired']} "
          f"| test acc {cfg['test_acc']:.4f} (majority {cfg['majority']:.4f})")

    # the dataset must be built exactly as it was for training, alignment
    # included, or the model is scored on inputs it never saw
    m = load_meta(a.crops, cfg["task"], A["min_side"], a.buildings,
                  Path(A.get("radiometry", a.radiometry)))
    if A.get("disaster"):
        m = m[m.disaster == A["disaster"]].reset_index(drop=True)
        print(f"  restricted to {A['disaster']} ({len(m):,} crops) -- the model "
              f"was trained only on this event")
    m["fold"] = assign_folds(m, A["folds"], cfg["group"], A["seed"])
    te = m[m.fold == cfg["fold"]].reset_index(drop=True)
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

    S, P, U, ident, W = similarities(ppnet, ds, dev, a.batch, a.workers)
    d = te.set_index("uid").loc[U].reset_index()

    # per-class best similarity, and the decision-grounded prototype: the one
    # that actually moves the class's logit, not the one that merely lights up
    per_cls = np.stack([S[:, ident[:, c] > 0].max(1) for c in range(n_cls)], 1)
    contrib = np.stack(
        [(S[:, ident[:, c] > 0] * W[c, ident[:, c] > 0]).max(1)
         for c in range(n_cls)], 1)

    d["p1"] = P[:, 1] if n_cls == 2 else P.max(1)
    d["margin"] = np.abs(P[:, 0] - P[:, 1]) if n_cls == 2 else \
        np.sort(P, 1)[:, -1] - np.sort(P, 1)[:, -2]
    d["sim_best"] = per_cls.max(1)
    d["coact"] = np.sort(per_cls, 1)[:, -2]      # second-best class similarity
    d["coact_rel"] = d.coact / np.maximum(d.sim_best, 1e-6)
    d["contrib_gap"] = np.sort(contrib, 1)[:, -1] - np.sort(contrib, 1)[:, -2]

    print("\n=== 1. is the measure saturated? ===")
    for nm in ["sim_best", "coact"]:
        v = d[nm].values
        nuniq = len(np.unique(np.round(v, 4)))
        at_ceil = float(np.mean(np.isclose(v, LOG_CEILING, atol=1e-3)))
        print(f"  {nm:10s} n_unique(4dp) {nuniq:6,}  sd {v.std():.4f}  "
              f"range [{v.min():.4f}, {v.max():.4f}]  at log-ceiling {at_ceil:.1%}")
    ok = d.coact.std() > 1e-3 and len(np.unique(np.round(d.coact, 4))) > 50
    print(f"  -> co-activation {'VARIES; continue' if ok else 'is DEGENERATE'}")
    if not ok:
        print("     nothing below is interpretable. This is exactly how the")
        print("     pedestrian experiment failed; stop and fix the model.")

    print("\n=== 2. is it just image degradation? ===")
    # Within one event the capture parameters barely move -- off-nadir, GSD and
    # sun elevation are properties of the satellite pass, and an event is a
    # handful of passes. A constant covariate contributes nothing to a partial
    # correlation and makes spearmanr return nan, so drop them and say so.
    have, const = [], []
    for c in DEGRADE:
        if c not in d or not d[c].notna().any():
            continue
        (const if d[c].nunique(dropna=True) < 3 else have).append(c)
    print(f"  covariates available: {have}")
    if const:
        print(f"  constant within this subset, dropped: {const}")
        print(f"  (within one event the capture geometry is fixed, so these")
        print(f"   cannot confound anything here -- which is itself a reason")
        print(f"   to prefer within-event analysis)")
    from scipy import stats
    for target, name in [(d.margin.values, "margin (low = torn)")]:
        r, p = stats.spearmanr(d.coact.values, target, nan_policy="omit")
        pr, pp, n = partial_spearman(d.coact.values, target, d[have]) \
            if have else (np.nan, np.nan, 0)
        print(f"  coact vs {name}: raw rho {r:+.3f} (p={p:.3g}); "
              f"partialled rho {pr:+.3f} (p={pp:.3g}, n={n:,})")
    for c in have:
        r, p = stats.spearmanr(d.coact.values, d[c].values, nan_policy="omit")
        print(f"    coact vs {c:10s} rho {r:+.3f} (p={p:.3g})")

    print("\n=== 3. torn vs confident, on the measures ===")
    q = d.margin.quantile([0.25, 0.75])
    torn = d[d.margin <= q.iloc[0]]
    conf = d[d.margin >= q.iloc[1]]
    for nm in ["coact", "coact_rel", "sim_best", "contrib_gap", "px_side"]:
        if nm not in d:
            continue
        t, c = torn[nm].dropna(), conf[nm].dropna()
        if len(t) < 20 or len(c) < 20:
            continue
        u = stats.mannwhitneyu(t, c)
        print(f"  {nm:12s} torn {t.median():8.4f}  confident {c.median():8.4f}"
              f"  (Mann-Whitney p={u.pvalue:.3g})")

    print("\n=== 4. where is tornness concentrated? ===")
    if {"lon", "lat"} <= set(d.columns) and d.lon.notna().any():
        i, p = morans_i(d.coact.values, d.lon.values, d.lat.values)
        print(f"  Moran's I on co-activation: {i:+.4f} (perm p={p:.3g})")
        i2, p2 = morans_i((-d.margin).values, d.lon.values, d.lat.values)
        print(f"  Moran's I on tornness (-margin): {i2:+.4f} (perm p={p2:.3g})")
        print("  positive and significant means ambiguity clusters in space --")
        print("  neighbourhoods, not scattered buildings")
    if "disaster" in d and d.disaster.notna().any():
        print(f"\n  {'disaster':26s} {'n':>7} {'median coact':>13} {'torn rate':>10}")
        thr = d.margin.quantile(0.25)
        for dis, g in d.groupby("disaster"):
            if len(g) < 100:
                continue
            print(f"  {str(dis)[:26]:26s} {len(g):7,} {g.coact.median():13.4f} "
                  f"{(g.margin <= thr).mean():9.1%}")

    out = a.run / "tornness.parquet"
    keep = ["uid", "scene", "disaster", "damage", "label", "p1", "margin",
            "sim_best", "coact", "coact_rel", "contrib_gap"] + have + \
           [c for c in ["lon", "lat"] if c in d]
    d[[c for c in keep if c in d]].to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
