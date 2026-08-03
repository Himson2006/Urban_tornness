"""Experiment 1 -- does tornness *shape* predict human disagreement?

The claim under test: among predictions where the model is torn, dual-match
tornness (high co-activation: the crop matches both classes' prototypes) tracks
pedestrians humans also split on, while weak-match tornness (low co-activation
AND low global max similarity: the crop matches nothing) does not. Scalar
uncertainty -- entropy, margin, MC-dropout -- cannot make that distinction even
in principle, so beating them is the whole point.

METHOD NOTE, and it is the difference between a result and an artefact:
`intention_prob` is a single value per pedestrian track. Correlating a per-crop
feature against it at crop level is pseudo-replication -- it repeats the same y
40x per pedestrian and shrinks p-values by a factor of sqrt(40) for free. So
everything here aggregates to ONE ROW PER PEDESTRIAN first, and n is the number
of held-out pedestrians (~1,842 across folds), not the number of crops.

Confounds: contested pedestrians in any driving dataset skew small, blurry and
occluded. Every correlation is therefore reported both raw and partial, with
crop size / blur / occlusion / truncation regressed out on ranks.

Usage:
    python src/exp1_disagreement.py                     # all folds pooled
    python src/exp1_disagreement.py --torn-quantile 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# uncertainty measures under comparison
SHAPE = ["coact_min", "coact_prod", "typing_score", "global_max_sim"]
SCALAR = ["entropy", "margin", "mc_dropout_std"]
COVARS = ["bbox_h", "blur_var_laplacian", "occluded_flag", "truncated"]


def load(runs: Path, arch: str) -> pd.DataFrame:
    files = sorted(runs.glob(f"{arch}_fold*/tornness_fold*.parquet"))
    if not files:
        raise SystemExit(
            f"no tornness parquets under {runs}/{arch}_fold*/\n"
            f"run: ./scripts/train_all.sh  (it extracts them after training)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(files)} folds, {len(df):,} crops, "
          f"{df.ped_id.nunique():,} pedestrians")
    return df


def to_pedestrian_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pedestrian: the unit at which the human label exists."""
    d = df[df.in_exp_window] if "in_exp_window" in df else df
    agg = {c: "mean" for c in SHAPE + SCALAR if c in d.columns}
    agg.update({c: "mean" for c in COVARS if c in d.columns})
    agg["human_disagreement"] = "first"
    agg["intention_prob"] = "first"
    agg["intent_binary"] = "first"
    agg["p_cross"] = "mean"
    out = d.groupby("ped_id").agg(agg)
    # peak co-activation over the window: a pedestrian is "dual-match" if they
    # ever strongly matched both readings, not only on average
    for c in ("coact_min", "global_max_sim"):
        if c in d.columns:
            out[c + "_max"] = d.groupby("ped_id")[c].max()
    return out.reset_index()


def partial_spearman(x, y, Z):
    """Spearman of x,y after linearly removing Z -- all on ranks."""
    r = lambda a: stats.rankdata(a)
    X, Y = r(x), r(y)
    Zr = np.column_stack([r(Z[:, j]) for j in range(Z.shape[1])])
    Zr = np.column_stack([np.ones(len(X)), Zr])
    bx = np.linalg.lstsq(Zr, X, rcond=None)[0]
    by = np.linalg.lstsq(Zr, Y, rcond=None)[0]
    return stats.pearsonr(X - Zr @ bx, Y - Zr @ by)


def boot_ci(x, y, n=2000, seed=0):
    """Percentile bootstrap over pedestrians (the independent unit)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        rs.append(stats.spearmanr(x[b], y[b]).statistic)
    return (np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)) if rs else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--torn-quantile", type=float, default=0.25,
                    help="fraction of most-torn pedestrians analysed")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    df = load(a.runs, a.arch)
    ped = to_pedestrian_level(df)
    print(f"pedestrian-level rows: {len(ped):,}")

    # torn = smallest margin. This is where scalar uncertainty gives up and the
    # shape question becomes meaningful.
    thr = ped.margin.quantile(a.torn_quantile)
    torn = ped[ped.margin <= thr].reset_index(drop=True)
    print(f"torn subset: {len(torn):,} pedestrians (margin <= {thr:.4f})")
    print(f"  their human_disagreement: mean {torn.human_disagreement.mean():.3f} "
          f"vs {ped.human_disagreement.mean():.3f} overall")

    feats = [c for c in SHAPE + SCALAR + ["coact_min_max", "global_max_sim_max"]
             if c in torn.columns]
    Z = torn[[c for c in COVARS if c in torn.columns]].to_numpy(float)
    Z = np.nan_to_num(Z)
    y = torn.human_disagreement.to_numpy(float)

    rows = []
    for f in feats:
        x = torn[f].to_numpy(float)
        if np.all(np.isnan(x)) or np.nanstd(x) == 0:
            continue
        rho, p = stats.spearmanr(x, y)
        lo, hi = boot_ci(x, y, a.bootstrap)
        prho, pp = partial_spearman(x, y, Z)
        rows.append({
            "feature": f,
            "kind": "shape" if f.startswith(tuple(s[:5] for s in SHAPE)) or
                    f in ("coact_min_max", "global_max_sim_max", "typing_score")
                    else "scalar",
            "spearman": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
            "partial_rho": prho, "partial_p": pp,
        })

    res = pd.DataFrame(rows).sort_values("spearman", key=abs, ascending=False)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "exp1_disagreement.csv", index=False)

    print("\n=== Experiment 1: correlation with human disagreement "
          "(torn pedestrians only) ===\n")
    show = res.copy()
    for c in ("spearman", "ci_lo", "ci_hi", "partial_rho"):
        show[c] = show[c].round(4)
    for c in ("p", "partial_p"):
        show[c] = show[c].apply(lambda v: f"{v:.2e}")
    print(show.to_string(index=False))

    # Is the "torn" subset actually torn? If the model is saturated, margin's
    # lower quartile still sits near 1.0 and this experiment has not run at all.
    print(f"\n=== is the torn subset actually torn? ===")
    q = ped.margin.quantile([0.01, 0.05, 0.10, 0.25, 0.50]).round(4)
    print("  margin quantiles: " + ", ".join(f"p{int(k*100)}={v}" for k, v in q.items()))
    frac = (ped.margin < 0.5).mean()
    print(f"  pedestrians with margin < 0.50: {frac:.4f}  "
          f"(< 0.20: {(ped.margin < 0.2).mean():.4f})")
    saturated = thr > 0.9
    if saturated:
        print("  !! SATURATED: even the most-torn quartile is near-certain.")
        print("     The model outputs p~0/1 everywhere, so there are no torn")
        print("     predictions to type. Fix calibration/overfitting before")
        print("     drawing any conclusion from the table above.")

    shp = res[res.kind == "shape"]
    scl = res[res.kind == "scalar"]
    b_shp, b_scl = shp.spearman.abs().max(), scl.spearman.abs().max()
    sig = res[res.p < 0.05]
    print(f"\nbest shape |rho| = {b_shp:.4f}   |   best scalar |rho| = {b_scl:.4f}")
    print(f"features significant at p<0.05: {len(sig)} of {len(res)}")
    if saturated:
        print("=> INCONCLUSIVE: no genuinely torn predictions exist. Not a test "
              "of the hypothesis.")
    elif len(sig) == 0:
        print("=> NULL RESULT: nothing predicts human disagreement, shape or scalar.")
    elif b_shp > b_scl + 0.05 and (shp.p < 0.05).any():
        print("=> Shape beats scalar uncertainty. The core claim has support.")
    else:
        print("=> Shape does NOT meaningfully beat scalar uncertainty "
              "(difference within noise).")
    print(f"\nwrote {a.out/'exp1_disagreement.csv'}")


if __name__ == "__main__":
    main()
