"""What actually drives human disagreement in PIE?

Experiment 1 returned a null: co-activation predicts `human_disagreement` at
rho~0.10 raw, but ~0.03 after regressing out crop size, blur, occlusion and
truncation. So the raw signal was degradation, not prototype geometry.

That failure is informative if we take it seriously. The paper's thesis names two
kinds of tornness:

  weak-match     no evidence -- distance, occlusion, blur. Resolvable by looking
                 longer. Humans facing the same crop are individually unsure.
  dual-match     abundant, conflicting evidence. Humans split into camps.

If PIE's contested pedestrians are overwhelmingly the *first* kind, then PIE
simply lacks the population the hypothesis is about, and co-activation failing to
predict disagreement there is expected rather than disconfirming.

This script tests that directly:
  1. how strongly do degradation covariates alone predict disagreement?
  2. do the behavioural cue-conflict tags (standing AND looking -- the running
     example's "at the curb, checking traffic, not moving") predict it?
  3. within visually CLEAN pedestrians only -- large, sharp, unoccluded, where
     an information deficit cannot be the explanation -- does co-activation
     predict disagreement?

(3) is the real test. It is the sub-population where dual-match is the only
available account of disagreement.

Usage:
    python src/exp1b_what_drives_disagreement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DEG = ["bbox_h", "blur_var_laplacian", "occluded_flag", "truncated"]
SHAPE = ["coact_min", "coact_prod", "typing_score", "global_max_sim"]
SCALAR = ["entropy", "margin", "mc_dropout_std"]


def spear(x, y):
    r = stats.spearmanr(x, y)
    return r.statistic, r.pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--clean-quantile", type=float, default=0.5,
                    help="keep pedestrians in the cleanest half on every "
                         "degradation axis")
    a = ap.parse_args()

    files = sorted(a.runs.glob(f"{a.arch}_fold*/tornness_fold*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    win = df[df.in_exp_window]

    agg = {c: "mean" for c in SHAPE + SCALAR + DEG if c in win.columns}
    agg["human_disagreement"] = "first"
    agg["p_cross"] = "mean"
    ped = win.groupby("ped_id").agg(agg)
    # behavioural cue conflict, straight from PIE's per-frame tags
    ped["frac_standing"] = win.groupby("ped_id").action.apply(
        lambda s: (s == "standing").mean())
    ped["frac_looking"] = win.groupby("ped_id").look.apply(
        lambda s: (s == "looking").mean())
    ped["cue_conflict"] = ped.frac_standing * ped.frac_looking
    ped["p_half"] = (ped.p_cross - 0.5).abs()
    y = ped.human_disagreement.to_numpy()
    print(f"{len(ped):,} pedestrians\n")

    print("=== 1. does image degradation predict human disagreement? ===")
    for c in DEG + ["cue_conflict", "frac_standing", "frac_looking"]:
        if c not in ped:
            continue
        r, p = spear(ped[c], y)
        print(f"  {c:22s} rho={r:+.3f}  p={p:.2e} {'*' if p < 0.05 else ''}")

    print("\n=== 2. clean sub-population (degradation cannot explain it) ===")
    clean = ped[
        (ped.bbox_h >= ped.bbox_h.quantile(a.clean_quantile))
        & (ped.blur_var_laplacian >= ped.blur_var_laplacian.quantile(a.clean_quantile))
        & (ped.occluded_flag <= ped.occluded_flag.quantile(1 - a.clean_quantile))
    ]
    print(f"  n={len(clean):,} of {len(ped):,} "
          f"(bbox_h >= {ped.bbox_h.quantile(a.clean_quantile):.0f}px, "
          f"sharp, unoccluded)")
    print(f"  their disagreement: {clean.human_disagreement.mean():.3f} "
          f"vs {ped.human_disagreement.mean():.3f} overall")

    for label, sub in [("clean, ALL", clean),
                       ("clean, torn 50%",
                        clean[clean.p_half <= clean.p_half.quantile(0.5)]),
                       ("clean, torn 25%",
                        clean[clean.p_half <= clean.p_half.quantile(0.25)])]:
        if len(sub) < 30:
            print(f"  {label}: n={len(sub)} too small")
            continue
        yy = sub.human_disagreement.to_numpy()
        out = []
        for c in SHAPE + SCALAR + ["cue_conflict"]:
            if c not in sub or sub[c].std() == 0:
                continue
            r, p = spear(sub[c], yy)
            out.append(f"{c}={r:+.3f}{'*' if p < 0.05 else ' '}")
        print(f"  {label:18s} n={len(sub):>5,}  " + "  ".join(out))

    print("\n  (* = p<0.05)")
    print("\n=== 3. read ===")
    rdeg = max(abs(spear(ped[c], y)[0]) for c in DEG if c in ped)
    print(f"  strongest degradation correlate |rho| = {rdeg:.3f}")
    if rdeg > 0.15:
        print("  PIE's contested pedestrians are substantially a WEAK-MATCH")
        print("  population -- humans disagree because evidence is poor, not")
        print("  because it conflicts. That is the type the hypothesis predicts")
        print("  co-activation should NOT track.")


if __name__ == "__main__":
    main()
