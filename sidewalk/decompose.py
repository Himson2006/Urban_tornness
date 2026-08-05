"""Separate what the label is from who reviewed it.

Reviewer "unsure" rates run from 0% to 60%, so a raw unsure count on a label
conflates two things: whether the image was genuinely unclear, and whether an
unsure-prone person happened to review it. The same confound applies to
disagreement -- some reviewers are simply more willing to overrule a labeller.

We fit the additive model

    y_ij  ~  mu + a_i (label) + b_j (reviewer)

by alternating means, which is the two-way additive fit and converges in a few
passes. Label effects are then shrunk toward zero in proportion to how few
reviews they rest on (empirical Bayes), so a label seen by three people is not
trusted as much as one seen by twelve.

Two outputs matter:

  variance split   how much of the disagreement lives in reviewers vs labels.
                   If reviewers dominate, counting votes measures the crowd,
                   not the sidewalk.
  adjusted scores  per-label difficulty with reviewer tendency removed. Every
                   downstream claim should use these, not raw counts.

Usage:
    python sidewalk/decompose.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def two_way_fit(y: np.ndarray, li: np.ndarray, ri: np.ndarray,
                n_label: int, n_rev: int, iters: int = 30):
    """Alternating means for y ~ mu + a[label] + b[reviewer]."""
    mu = y.mean()
    a = np.zeros(n_label)
    b = np.zeros(n_rev)
    lab_n = np.bincount(li, minlength=n_label).astype(float)
    rev_n = np.bincount(ri, minlength=n_rev).astype(float)
    for _ in range(iters):
        resid = y - mu - b[ri]
        a = np.bincount(li, resid, minlength=n_label) / np.maximum(lab_n, 1)
        resid = y - mu - a[li]
        b = np.bincount(ri, resid, minlength=n_rev) / np.maximum(rev_n, 1)
        b -= b.mean()          # identifiability: reviewer effects sum to zero
    return mu, a, b, lab_n, rev_n


def shrink(a: np.ndarray, n: np.ndarray, resid_var: float) -> np.ndarray:
    """Empirical-Bayes shrinkage: trust a label in proportion to its reviews."""
    tau2 = max(a.var() - resid_var / np.maximum(n, 1).mean(), 1e-6)
    w = tau2 / (tau2 + resid_var / np.maximum(n, 1))
    return a * w


def analyse(v: pd.DataFrame, outcome: str, mask: np.ndarray | None = None):
    d = v if mask is None else v[mask]
    y = (d.vote == outcome).to_numpy(float)
    lcat = d.label_id.astype("category")
    rcat = d.validator_id.astype("category")
    li, ri = lcat.cat.codes.to_numpy(), rcat.cat.codes.to_numpy()
    mu, a, b, lab_n, rev_n = two_way_fit(y, li, ri, len(lcat.cat.categories),
                                         len(rcat.cat.categories))
    resid = y - mu - a[li] - b[ri]
    rv = resid.var()

    # crude but interpretable: variance of each effect across observations
    var_lab, var_rev = a[li].var(), b[ri].var()
    tot = var_lab + var_rev + rv
    print(f"\n=== {outcome} ({len(d):,} reviews, "
          f"{len(lcat.cat.categories):,} labels, "
          f"{len(rcat.cat.categories):,} reviewers) ===")
    print(f"  base rate                 {mu:6.3f}")
    print(f"  variance from the LABEL    {var_lab/tot:6.1%}")
    print(f"  variance from the REVIEWER {var_rev/tot:6.1%}")
    print(f"  unexplained                {rv/tot:6.1%}")
    print(f"  reviewer effect spread: {b.min():+.3f} to {b.max():+.3f} "
          f"(sd {b.std():.3f})")

    a_s = shrink(a, lab_n, rv)
    eff = pd.DataFrame({
        "label_id": lcat.cat.categories,
        f"{outcome.lower()}_raw": mu + a,
        f"{outcome.lower()}_adj": mu + a_s,
        f"{outcome.lower()}_n": lab_n,
    })
    rev_eff = pd.Series(b, index=rcat.cat.categories, name="rev_effect")
    return eff, rev_eff


def order_check(v: pd.DataFrame) -> None:
    """Is vote_index chronological? Test it before using it as time."""
    print("\n=== is vote_index an ordering? ===")
    act = v.validator_id.value_counts()
    v = v.assign(rev_activity=v.validator_id.map(act))
    from scipy import stats
    r = stats.spearmanr(v.vote_index, v.rev_activity)
    print(f"  vote_index vs reviewer activity: rho={r.statistic:+.3f} "
          f"(p={r.pvalue:.1e})")
    print("  a strong negative value would suggest prolific reviewers arrive")
    print("  first, i.e. the array is ordered by time rather than shuffled")
    early = v[v.vote_index < 3]
    late = v[v.vote_index >= 3]
    if len(late) > 500:
        print(f"  unsure rate  first 3 reviews {(early.vote=='Unsure').mean():.3f}"
              f"  |  later {(late.vote=='Unsure').mean():.3f}")
        print(f"  disagree rate first 3        {(early.vote=='Disagree').mean():.3f}"
              f"  |  later {(late.vote=='Disagree').mean():.3f}")
    print("  NOTE: only interpretable as resolution if the order is real AND")
    print("  the same labels appear in both groups -- compare within-label next")


def composition_check(v: pd.DataFrame, rev_eff: pd.Series, outcome: str) -> None:
    """Do later reviews simply come from more critical reviewers?

    If the reviewer pool shifts with arrival order, an apparent rise in
    disagreement over a label's lifetime could be about who showed up rather
    than about the label. This quantifies that directly.
    """
    e = v.validator_id.map(rev_eff)
    early, late = v.vote_index < 3, v.vote_index >= 3
    print(f"\n=== reviewer composition by arrival, {outcome} ===")
    print(f"  mean reviewer effect, first 3 reviews : {e[early].mean():+.4f}")
    print(f"  mean reviewer effect, later reviews   : {e[late].mean():+.4f}")
    print(f"  difference                            : "
          f"{e[late].mean()-e[early].mean():+.4f}")
    print("  (positive => later reviewers are inherently more likely to give")
    print(f"   '{outcome}', which would inflate any within-label trend)")


def within_label(v: pd.DataFrame, rev_eff: dict, min_reviews: int = 6) -> None:
    """The resolvability test, done within each label rather than across them.

    Run twice: on the raw votes, and after subtracting each reviewer's
    estimated tendency. If the trend survives adjustment it is about the label.
    """
    g = v.groupby("label_id").vote_index.max()
    keep = g[g >= min_reviews - 1].index
    s = v[v.label_id.isin(keep)]
    if len(s) < 500:
        print(f"\n(only {s.label_id.nunique()} labels with >={min_reviews} "
              f"reviews; skipping within-label test)")
        return
    from scipy import stats
    print(f"\n=== within-label: first 3 reviews vs later "
          f"({s.label_id.nunique():,} labels) ===")
    print(f"  {'outcome':10s} {'adj':>4} {'early':>7} {'later':>7} "
          f"{'delta':>8} {'p':>10}")
    for outcome in ("Unsure", "Disagree"):
        sub = s if outcome == "Unsure" else s[s.vote != "Unsure"]
        y = (sub.vote == outcome).astype(float)
        b = sub.validator_id.map(rev_eff[outcome]).fillna(0.0)
        for tag, val in (("no", y), ("yes", y - b)):
            t = pd.DataFrame({"label_id": sub.label_id.values,
                              "late": (sub.vote_index >= 3).values,
                              "y": val.values})
            g = t.groupby(["label_id", "late"]).y.mean().unstack()
            g = g.dropna()
            if len(g) < 200:
                continue
            w = stats.wilcoxon(g[False], g[True]) if (g[False] != g[True]).any() else None
            print(f"  {outcome:10s} {tag:>4} {g[False].mean():7.3f} "
                  f"{g[True].mean():7.3f} {g[True].mean()-g[False].mean():+8.3f} "
                  f"{w.pvalue if w else float('nan'):10.1e}")
    print("  'adj=yes' subtracts each reviewer's estimated tendency first.")
    print("  A trend that survives adjustment is about the label, not the crowd.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    a = ap.parse_args()

    v = pd.read_parquet(a.data / "validations.parquet")
    print(f"{len(v):,} individual reviews")

    uns, uns_rev = analyse(v, "Unsure")
    took = (v.vote != "Unsure").to_numpy()
    dis, dis_rev = analyse(v, "Disagree", took)
    rev_eff = {"Unsure": uns_rev, "Disagree": dis_rev}

    out = uns.merge(dis, on="label_id", how="outer")
    dest = a.data / "label_effects.parquet"
    out.to_parquet(dest, index=False)

    order_check(v)
    for oc, re_ in rev_eff.items():
        composition_check(v if oc == "Unsure" else v[v.vote != "Unsure"], re_, oc)
    within_label(v, rev_eff)

    lab = pd.read_parquet(a.data / "labels.parquet")
    m = lab.merge(out, on="label_id", how="inner")
    if len(m):
        from scipy import stats
        r = stats.spearmanr(m.unsure, m.unsure_adj)
        print(f"\nraw unsure vs reviewer-adjusted: rho={r.statistic:+.3f}")
        print("  values well below 1 mean the ranking of 'unclear' labels")
        print("  changes once reviewer tendency is removed")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
