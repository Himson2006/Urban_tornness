"""Experiment 1 on CIFAR-10H -- the actual test of the shape claim.

PIE could only ask "does co-activation predict HOW MUCH humans disagreed", and
the answer was no once degradation was controlled. CIFAR-10H has the individual
votes, so we can ask the question the thesis is really about:

    among torn predictions, does co-activation predict the SHAPE of human
    disagreement -- two strong competing readings (bimodal / dual-match) rather
    than mass smeared over many (diffuse / weak-match)?

Outcomes, all continuous so every contested image counts:
    top2_mass    human mass in their top two classes   -- bimodality
    split_ratio  human top2/top1                       -- evenness of the split
    n_eff        exp(human entropy)                    -- diffuseness (inverse)

Predictors: shape features (coact_top2, coact_ratio, top2_share, global_max)
against scalar baselines (entropy, margin, mc_std) that cannot express shape.

Plus the test binary PIE could never run: does the model's competing PAIR match
the humans' competing pair, and is pair agreement higher when co-activation is
high?

Usage:
    python src/exp1_cifar.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SHAPE = ["coact_top2", "coact_ratio", "top2_share", "global_max", "n_eff_sim"]
SCALAR = ["entropy", "margin", "mc_std"]
OUTCOMES = ["top2_mass", "split_ratio", "n_eff", "disagree"]


def boot_ci(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        rs.append(stats.spearmanr(x[b], y[b]).statistic)
    return (np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)) if rs \
        else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=ROOT / "runs_cifar/resnet34_cifar10/tornness_cifar10h.parquet")
    ap.add_argument("--torn-quantile", type=float, default=0.25)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    print(f"{len(df):,} test images")

    thr = df.margin.quantile(a.torn_quantile)
    torn = df[df.margin <= thr].reset_index(drop=True)
    print(f"torn subset: {len(torn):,} images (model margin <= {thr:.4f})")
    print(f"  their human disagreement {torn.disagree.mean():.3f} "
          f"vs {df.disagree.mean():.3f} overall")
    if thr > 0.95:
        print("  !! model is saturated -- these are not really torn")

    rows = []
    for out in OUTCOMES:
        y = torn[out].to_numpy(float)
        for f in SHAPE + SCALAR:
            if f not in torn or torn[f].std() == 0 or torn[f].isna().all():
                continue
            x = torn[f].to_numpy(float)
            rho, p = stats.spearmanr(x, y)
            lo, hi = boot_ci(x, y, a.bootstrap)
            rows.append({"outcome": out, "feature": f,
                         "kind": "shape" if f in SHAPE else "scalar",
                         "spearman": rho, "p": p, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "exp1_cifar.csv", index=False)

    for out in OUTCOMES:
        sub = res[res.outcome == out].sort_values("spearman", key=abs,
                                                  ascending=False)
        print(f"\n=== outcome: {out} ===")
        s = sub.copy()
        for c in ("spearman", "ci_lo", "ci_hi"):
            s[c] = s[c].round(4)
        s["p"] = s.p.apply(lambda v: f"{v:.2e}")
        print(s[["feature", "kind", "spearman", "p", "ci_lo", "ci_hi"]]
              .to_string(index=False))
        b_s = sub[sub.kind == "shape"].spearman.abs().max()
        b_c = sub[sub.kind == "scalar"].spearman.abs().max()
        verdict = ("shape wins" if b_s > b_c + 0.05 else
                   "scalar wins" if b_c > b_s + 0.05 else "tie")
        print(f"  best shape {b_s:.4f} vs best scalar {b_c:.4f}  ->  {verdict}")

    print("\n=== does the model name the same two readings as the humans? ===")
    con = df[df.top1 < 0.90]
    print(f"  pair match, all images        : {df.pair_match.mean():.4f}")
    print(f"  pair match, contested images  : {con.pair_match.mean():.4f}")
    if len(torn):
        hi = torn[torn.coact_ratio >= torn.coact_ratio.quantile(0.75)]
        lo = torn[torn.coact_ratio <= torn.coact_ratio.quantile(0.25)]
        print(f"  torn & HIGH co-activation     : {hi.pair_match.mean():.4f} "
              f"(n={len(hi)})")
        print(f"  torn & LOW  co-activation     : {lo.pair_match.mean():.4f} "
              f"(n={len(lo)})")
        if len(hi) and len(lo):
            t = stats.fisher_exact([[hi.pair_match.sum(), (~hi.pair_match).sum()],
                                    [lo.pair_match.sum(), (~lo.pair_match).sum()]])
            print(f"  Fisher exact p = {t.pvalue:.2e}")
            print("  (higher pair match under high co-activation is the strongest")
            print("   form of 'the exemplars name the two competing readings')")
    print(f"\nwrote {a.out/'exp1_cifar.csv'}")


if __name__ == "__main__":
    main()
