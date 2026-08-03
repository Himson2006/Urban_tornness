"""Robustness checks for the CIFAR-10H pair-identification result.

Headline: within contested images, the top quartile of `coact_top2` gets the
humans' competing pair right 47.2% of the time vs 16.1% for the bottom quartile
(Fisher p=8.7e-17), while entropy (p=0.12) and margin (p=0.17) do not stratify
pair accuracy at all.

Three ways that could still be less than it looks, tested here:

  1. Threshold-picking. Does it survive other definitions of "contested" and
     other quartile cuts, or does it live at top1<0.90 specifically?
  2. Confidence in disguise. Co-activation correlates with being unsure. Does it
     predict pair match *beyond* entropy and margin, or is it a proxy? Tested by
     nested logistic regression + AUROC, which does not require picking cuts.
  3. One easy pair. CIFAR-10H's contested cases are dominated by cat/dog. If the
     effect is that pair alone, the claim is about one confusion, not geometry.

Usage:
    python src/exp1_cifar_robust.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def gate(df, f, q=0.25):
    v = df[f]
    hi, lo = df[v >= v.quantile(1 - q)], df[v <= v.quantile(q)]
    if not len(hi) or not len(lo):
        return None
    ft = stats.fisher_exact([[hi.pair_match.sum(), (~hi.pair_match).sum()],
                             [lo.pair_match.sum(), (~lo.pair_match).sum()]])
    return hi.pair_match.mean(), lo.pair_match.mean(), ft.pvalue, len(hi), len(lo)


def boot_spread(df, f, q=0.25, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    out = []
    for _ in range(n):
        s = df.iloc[rng.choice(idx, len(idx), replace=True)]
        g = gate(s, f, q)
        if g:
            out.append(g[0] - g[1])
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path,
                    default=ROOT / "runs_cifar/resnet34_cifar10/tornness_cifar10h.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.features)
    df["pair_match"] = df.pair_match.astype(bool)

    print("=== 1. sensitivity to the contested threshold and quartile cut ===\n")
    rows = []
    for thr in (0.80, 0.85, 0.90, 0.95, 0.99):
        con = df[df.top1 < thr]
        for q in (0.20, 0.25, 0.33):
            g = gate(con, "coact_top2", q)
            if not g:
                continue
            rows.append({"contested_thr": thr, "quantile": q, "n": len(con),
                         "hi": g[0], "lo": g[1], "spread": g[0] - g[1],
                         "p": g[2]})
    sens = pd.DataFrame(rows)
    s = sens.copy()
    for c in ("hi", "lo", "spread"):
        s[c] = s[c].round(4)
    s["p"] = s.p.apply(lambda v: f"{v:.1e}")
    print(s.to_string(index=False))
    print(f"\n  spread range {sens.spread.min():.3f}..{sens.spread.max():.3f}; "
          f"significant at p<0.01 in {(sens.p < 0.01).sum()}/{len(sens)} settings")

    con = df[df.top1 < 0.90]
    lo95, hi95 = boot_spread(con, "coact_top2")
    print(f"  bootstrap 95% CI on the spread (thr=0.90, q=0.25): "
          f"[{lo95:.3f}, {hi95:.3f}]")

    print("\n=== 2. does co-activation add beyond entropy and margin? ===\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = con.pair_match.to_numpy().astype(int)
    sets = {
        "entropy+margin (scalar only)": ["entropy", "margin"],
        "coact_top2 only": ["coact_top2"],
        "coact_top2 + entropy + margin": ["coact_top2", "entropy", "margin"],
        "all shape": ["coact_top2", "coact_ratio", "top2_share", "global_max"],
    }
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, cols in sets.items():
        X = StandardScaler().fit_transform(con[cols].to_numpy(float))
        aucs = []
        for tr, te in skf.split(X, y):
            m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        print(f"  {name:32s} AUROC {np.mean(aucs):.4f} "
              f"(+/-{np.std(aucs):.4f})")
    print("\n  5-fold CV, predicting whether the model's top-2 pair IS the")
    print("  humans' pair. No thresholds involved.")

    print("\n=== 3. is it just cat/dog? ===\n")
    con = con.copy()
    con["hpair"] = [tuple(sorted((CLASSES[i], CLASSES[j])))
                    for i, j in zip(con.human_label, con.runner_up)]
    top = con.hpair.value_counts().head(6)
    print(f"  {'human pair':26s} {'n':>5} {'hi':>7} {'lo':>7} {'spread':>7}")
    for pair, n in top.items():
        sub = con[con.hpair == pair]
        g = gate(sub, "coact_top2")
        if g and min(g[3], g[4]) >= 5:
            print(f"  {pair[0]+'/'+pair[1]:26s} {n:>5} {g[0]:>7.3f} "
                  f"{g[1]:>7.3f} {g[0]-g[1]:>7.3f}")
        else:
            print(f"  {pair[0]+'/'+pair[1]:26s} {n:>5}   (too few)")
    excl = con[con.hpair != ("cat", "dog")]
    g = gate(excl, "coact_top2")
    if g:
        print(f"\n  EXCLUDING cat/dog: n={len(excl):,} hi={g[0]:.4f} "
              f"lo={g[1]:.4f} spread={g[0]-g[1]:.4f} p={g[2]:.2e}")
        print("  (if the spread holds here, the effect is not one confusion)")

    sens.to_csv(a.out / "exp1_cifar_robustness.csv", index=False)
    print(f"\nwrote {a.out/'exp1_cifar_robustness.csv'}")


if __name__ == "__main__":
    main()
