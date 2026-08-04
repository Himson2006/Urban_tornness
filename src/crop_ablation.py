"""Controlled crop-scale ablation: the paper's main table.

Everything is matched between conditions -- folds, epochs, push schedule
(start 5, every 5), optimiser, sampler -- so the only difference is how much
context surrounds the pedestrian:

    runs_ctrl   2.0 x bbox   the standard framing
    runs_tight  1.3 x bbox   pedestrian dominates the frame

Reports, pooled over the five held-out folds:

  AUROC              does tightening cost accuracy?
  silent failure     of pedestrians annotators split on (intention_prob in
                     [0.4, 0.6]), what fraction does the model call with
                     p > 0.9?  With a bootstrap CI, because n is only ~140.
  selectivity        that rate divided by the model's overall p>0.9 rate. A
                     model that is simply under-confident everywhere scores the
                     same on both; only a model that hesitates *specifically* on
                     contested cases gets a low ratio.
  localisation       fraction of prototypes peaking inside the pedestrian box,
                     against the chance baseline (1/crop_scale)^2. Raw rates are
                     not comparable across crop scales; the chance-corrected
                     difference is.

Usage:
    python src/crop_ablation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def ped_level(files) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    w = d[d.in_exp_window]
    g = w.groupby("ped_id")
    return pd.DataFrame({
        "p_cross": g.p_cross.mean(),
        "intent": g.intent_binary.first(),
        "ip": g.intention_prob.first(),
        "hd": g.human_disagreement.first(),
        "stand": g.action.apply(lambda s: (s == "standing").mean()),
        "entropy": g.entropy.mean(),
    })


def boot_rate(mask: np.ndarray, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    if not len(mask):
        return np.nan, np.nan
    r = [mask[rng.integers(0, len(mask), len(mask))].mean() for _ in range(n)]
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def localisation(run_glob: str, crop_scale: float):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from prototype_localization import classify, find_proto_dir, locate
    ins, ys = [], []
    for run in sorted(ROOT.glob(run_glob)):
        try:
            pdir = find_proto_dir(run)
        except SystemExit:
            continue
        for f in sorted(pdir.glob("prototype-self-act*.npy")):
            x, y = locate(np.load(f))
            inside, _ = classify(x, y, crop_scale)
            ins.append(inside)
            ys.append(y)
    if not ins:
        return None
    return {"n_proto": len(ins), "inside": float(np.mean(ins)),
            "chance": (1.0 / crop_scale) ** 2, "mean_y": float(np.mean(ys))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["runs_ctrl:2.0", "runs_tight:1.3"],
                    help="dir:crop_scale pairs")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score
    rows = []
    for spec in a.conditions:
        d, sc = spec.split(":")
        sc = float(sc)
        files = sorted(ROOT.glob(f"{d}/{a.arch}_fold*/tornness_fold*.parquet"))
        if not files:
            print(f"  skip {d}: no tornness parquets "
                  f"(run src/tornness.py --crop-scale {sc})")
            continue
        p = ped_level(files)
        split = p.ip.between(0.4, 0.6)
        conf = ((p.p_cross - 0.5).abs() * 2 > 0.8).to_numpy()
        sf = conf[split.to_numpy()]
        lo, hi = boot_rate(sf)
        loc = localisation(f"{d}/{a.arch}_fold*", sc)
        rows.append({
            "condition": d, "crop_scale": sc, "n_ped": len(p),
            "auroc": roc_auc_score(p.intent, p.p_cross),
            "conf_all": conf.mean(),
            "silent": sf.mean(), "silent_lo": lo, "silent_hi": hi,
            "n_split": int(split.sum()),
            "selectivity": sf.mean() / max(conf.mean(), 1e-9),
            "inside": loc["inside"] if loc else np.nan,
            "chance": loc["chance"] if loc else np.nan,
            "inside_vs_chance": (loc["inside"] - loc["chance"]) if loc else np.nan,
            "mean_y": loc["mean_y"] if loc else np.nan,
        })

    if not rows:
        raise SystemExit("no conditions had tornness parquets")
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "crop_ablation.csv", index=False)

    print("\n=== controlled crop-scale ablation (matched folds, epochs, push) ===\n")
    print(f"{'condition':12s} {'scale':>5} {'AUROC':>7} {'p>.9 all':>9} "
          f"{'silent':>8} {'95% CI':>16} {'select':>7}")
    for _, r in res.iterrows():
        print(f"{r.condition:12s} {r.crop_scale:5.1f} {r.auroc:7.4f} "
              f"{r.conf_all:8.1%} {r.silent:8.1%} "
              f"[{r.silent_lo:5.1%},{r.silent_hi:5.1%}] {r.selectivity:7.2f}")

    print(f"\n{'condition':12s} {'protos in box':>14} {'chance':>7} "
          f"{'vs chance':>10} {'mean height':>12}")
    for _, r in res.iterrows():
        print(f"{r.condition:12s} {r.inside:13.1%} {r.chance:7.0%} "
              f"{r.inside_vs_chance:+10.1%} {r.mean_y:12.3f}")

    if len(res) == 2:
        a_, b_ = res.iloc[0], res.iloc[1]
        d_auroc = b_.auroc - a_.auroc
        overlap = not (b_.silent_hi < a_.silent_lo or a_.silent_hi < b_.silent_lo)
        print(f"\n  AUROC change {d_auroc:+.4f}  "
              f"({'no accuracy cost' if abs(d_auroc) < 0.02 else 'accuracy changed'})")
        print(f"  silent failure {a_.silent:.1%} -> {b_.silent:.1%}  "
              f"({'CIs OVERLAP -- not distinguishable' if overlap else 'CIs separate'})")
        print(f"  prototypes vs chance {a_.inside_vs_chance:+.1%} -> "
              f"{b_.inside_vs_chance:+.1%}")
    print(f"\nwrote {a.out/'crop_ablation.csv'}")


if __name__ == "__main__":
    main()
