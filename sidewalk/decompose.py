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
    return pd.DataFrame({
        "label_id": lcat.cat.categories,
        f"{outcome.lower()}_raw": mu + a,
        f"{outcome.lower()}_adj": mu + a_s,
        f"{outcome.lower()}_n": lab_n,
    })


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


def within_label(v: pd.DataFrame, min_reviews: int = 6) -> None:
    """The resolvability test, done within each label rather than across them."""
    g = v.groupby("label_id").vote_index.max()
    keep = g[g >= min_reviews - 1].index
    s = v[v.label_id.isin(keep)]
    if len(s) < 500:
        print(f"\n(only {s.label_id.nunique()} labels with >={min_reviews} "
              f"reviews; skipping within-label test)")
        return
    early = s[s.vote_index < 3].groupby("label_id").vote.apply(
        lambda x: (x == "Unsure").mean())
    late = s[s.vote_index >= 3].groupby("label_id").vote.apply(
        lambda x: (x == "Unsure").mean())
    e_d = s[s.vote_index < 3].groupby("label_id").vote.apply(
        lambda x: (x == "Disagree").mean())
    l_d = s[s.vote_index >= 3].groupby("label_id").vote.apply(
        lambda x: (x == "Disagree").mean())
    from scipy import stats
    j = pd.DataFrame({"e": early, "l": late}).dropna()
    jd = pd.DataFrame({"e": e_d, "l": l_d}).dropna()
    print(f"\n=== within-label: first 3 reviews vs later ({len(j):,} labels) ===")
    for name, t in (("unsure", j), ("disagree", jd)):
        w = stats.wilcoxon(t.e, t.l) if (t.e != t.l).any() else None
        print(f"  {name:9s} early {t.e.mean():.3f} -> later {t.l.mean():.3f}"
              f"  (delta {t.l.mean()-t.e.mean():+.3f}"
              f"{f', p={w.pvalue:.1e}' if w else ''})")
    print("  same labels on both sides, so this is not a selection effect --")
    print("  it is the actual test of whether reviewing resolves anything")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    a = ap.parse_args()

    v = pd.read_parquet(a.data / "validations.parquet")
    print(f"{len(v):,} individual reviews")

    uns = analyse(v, "Unsure")
    took = (v.vote != "Unsure").to_numpy()
    dis = analyse(v, "Disagree", took)

    out = uns.merge(dis, on="label_id", how="outer")
    dest = a.data / "label_effects.parquet"
    out.to_parquet(dest, index=False)

    order_check(v)
    within_label(v)

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
