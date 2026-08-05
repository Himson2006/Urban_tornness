"""Build the analysis table and report the paper's core numbers.

Every reviewed label gets its disagreement decomposed into two quantities that
mean different things:

  split   how evenly the agree/disagree votes divide, ignoring "not sure".
          High = reviewers took opposite positions confidently. This is a
          disagreement about WHERE THE STANDARD SITS -- does that crack count
          as a surface problem? More reviewing will not settle it.

  unsure  the fraction of reviewers who answered "not sure". High = nobody
          took a position. This is a disagreement about WHAT IS VISIBLE, and
          it is the kind better imagery could fix.

The claim under test is that these are different phenomena with different
remedies, and that the field's standard quality filter deletes both.

Usage:
    python sidewalk/manifest.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Project Sidewalk's published quality rule (see LabelAId, CHI 2024).
QC_AGREE_MIN, QC_DISAGREE_MAX = 2, 2


def load(raw: Path) -> pd.DataFrame:
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSVs in {raw}; run sidewalk/fetch_cities.py first")
    parts = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["city"] = f.stem
        parts.append(d)
        print(f"  {f.stem:12s} {len(d):>8,} labels")
    return pd.concat(parts, ignore_index=True)


def decompose(df: pd.DataFrame, min_val: int = 3) -> pd.DataFrame:
    d = df.copy()
    for c in ("agree_count", "disagree_count", "unsure_count"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
    d["n_val"] = d.agree_count + d.disagree_count + d.unsure_count
    d = d[d.n_val >= min_val].copy()

    taken = (d.agree_count + d.disagree_count).clip(lower=1)
    # 1.0 when agree and disagree are equal, 0.0 when unanimous among those
    # who took a position
    d["split"] = 1 - (d.agree_count - d.disagree_count).abs() / taken
    d["unsure"] = d.unsure_count / d.n_val
    d["p_agree"] = d.agree_count / d.n_val

    d["is_split"] = (d.split > 0.6) & (d.unsure < 0.2)
    d["is_unsure"] = d.unsure >= 0.34
    d["qc_kept"] = (d.agree_count > QC_AGREE_MIN) & (d.disagree_count <= QC_DISAGREE_MAX)

    # time_created is epoch MILLISECONDS, not a date string; parsing it as a
    # string silently produced a constant and made the age column useless.
    if "time_created" in d:
        t = pd.to_datetime(pd.to_numeric(d.time_created, errors="coerce"),
                           unit="ms", utc=True, errors="coerce")
        d["label_age_days"] = (pd.Timestamp.now(tz="UTC") - t).dt.days
    # image_capture_date is "YYYY-MM"
    if "image_capture_date" in d:
        ic = pd.to_datetime(d.image_capture_date, format="%Y-%m",
                            utc=True, errors="coerce")
        d["image_age_days"] = (pd.Timestamp.now(tz="UTC") - ic).dt.days

    # The `split` statistic is biased upward by the number of reviewers: with
    # more votes you are more likely to see at least one dissenter even when the
    # underlying disagreement rate is fixed. Subtract the expectation under a
    # matched null so trends across review counts mean something.
    d["split_excess"] = d.split - _null_split(d)
    return d


def _null_split(d: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Expected `split` if every label shared one disagreement rate."""
    rng = np.random.default_rng(seed)
    taken = (d.agree_count + d.disagree_count).to_numpy()
    rate = d.disagree_count.sum() / max(taken.sum(), 1)
    out = np.zeros(len(d))
    for n in np.unique(taken):
        if n < 1:
            continue
        dis = rng.binomial(n, rate, 4000)
        out[taken == n] = np.mean(1 - np.abs((n - dis) - dis) / n)
    return out


def explode_validations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (label, validator).

    `validations` holds a JSON array of individual reviewer records with
    user_id and Agree/Disagree/Unsure, in sequence. That makes two things
    possible the aggregate counts do not:

      * separating a validator who is often unsure from a label that is
        genuinely unclear -- the same confound, at the level of people;
      * comparing the first reviews of a label against later ones, which is
        the within-label test of whether reviewing actually resolves anything.

    Position in the array is recorded as `vote_index`; treat it as ordering
    only after checking it against label timestamps.
    """
    import json
    rows = []
    for lid, city, raw in zip(df.label_id, df.city, df.validations):
        if not isinstance(raw, str) or len(raw) < 5:
            continue
        try:
            arr = json.loads(raw)
        except Exception:
            continue
        for i, v in enumerate(arr):
            rows.append({"label_id": lid, "city": city, "vote_index": i,
                         "validator_id": v.get("user_id"),
                         "vote": v.get("validation")})
    return pd.DataFrame(rows)


def report(d: pd.DataFrame) -> None:
    n = len(d)
    print(f"\n=== {n:,} labels with >= 3 reviews, {d.city.nunique()} cities ===\n")

    print("the two kinds of disagreement")
    print(f"  reviewers SPLIT into camps   : {d.is_split.sum():>7,}  ({d.is_split.mean():5.1%})")
    print(f"  reviewers individually UNSURE: {d.is_unsure.sum():>7,}  ({d.is_unsure.mean():5.1%})")
    print(f"  both                         : {(d.is_split & d.is_unsure).sum():>7,}")
    from scipy import stats
    r = stats.spearmanr(d.split, d.unsure)
    print(f"  correlation between them     : rho={r.statistic:+.3f}  "
          f"(low => separate phenomena)\n")

    print(f"the standard quality filter (agree>{QC_AGREE_MIN} & "
          f"disagree<={QC_DISAGREE_MAX})")
    print(f"  keeps overall                : {d.qc_kept.mean():5.1%}")
    for name, m in (("SPLIT labels", d.is_split), ("UNSURE labels", d.is_unsure)):
        sub = d[m]
        if len(sub):
            print(f"  keeps {name:22s}: {sub.qc_kept.mean():5.1%}"
                  f"   -> discards {1-sub.qc_kept.mean():.0%}")

    print("\ndoes more reviewing resolve it?")
    print(f"  {'reviews':>9} {'unsure':>8} {'split':>8} {'labels':>9} {'med age(d)':>11}")
    for lo, hi in [(3, 3), (4, 4), (5, 6), (7, 9), (10, 999)]:
        g = d[(d.n_val >= lo) & (d.n_val <= hi)]
        if len(g) < 50:
            continue
        age = g.label_age_days.median() if "label_age_days" in g else np.nan
        print(f"  {f'{lo}-{hi}':>9} {g.unsure.mean():8.3f} {g.split.mean():8.3f} "
              f"{len(g):9,} {age:11.0f}")
    print("  (watch the age column -- if heavily reviewed labels are simply")
    print("   older, the trend may be selection rather than resolution)")

    print("\nby label type")
    print(f"  {'type':22s} {'n':>8} {'split':>8} {'unsure':>8} {'QC keeps':>9}")
    for t, g in d.groupby("label_type"):
        if len(g) < 200:
            continue
        print(f"  {str(t)[:22]:22s} {len(g):8,} {g.is_split.mean():8.1%} "
              f"{g.is_unsure.mean():8.1%} {g.qc_kept.mean():9.1%}")

    if "severity" in d:
        s = d[pd.to_numeric(d.severity, errors="coerce").notna()].copy()
        s["severity"] = s.severity.astype(float)
        print("\nby severity (1 = minor, 5 = severe)")
        print(f"  {'severity':>9} {'n':>8} {'split':>8} {'unsure':>8}")
        for sv, g in s.groupby("severity"):
            if len(g) < 100:
                continue
            print(f"  {sv:9.0f} {len(g):8,} {g.is_split.mean():8.1%} "
                  f"{g.is_unsure.mean():8.1%}")
        print("  (a standards disagreement should peak at middle severity,")
        print("   where the threshold actually sits)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "sidewalk/data/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--min-val", type=int, default=3)
    a = ap.parse_args()

    print("loading")
    df = load(a.raw)
    d = decompose(df, a.min_val)
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "labels.parquet"
    d.to_parquet(dest, index=False)
    report(d)
    print(f"\nwrote {dest} ({len(d):,} rows)")

    v = explode_validations(d)
    if len(v):
        vdest = a.out / "validations.parquet"
        v.to_parquet(vdest, index=False)
        print(f"wrote {vdest} ({len(v):,} individual reviews, "
              f"{v.validator_id.nunique():,} reviewers)")
        share = v.groupby("validator_id").vote.apply(
            lambda s: (s == "Unsure").mean())
        busy = v.validator_id.value_counts()
        share = share[busy[busy >= 20].index]
        print(f"  among reviewers with >=20 reviews (n={len(share):,}), "
              f"'unsure' rate ranges {share.min():.1%}-{share.max():.1%} "
              f"(median {share.median():.1%})")
        print("  -> large spread means 'unsure' is partly a property of the")
        print("     reviewer, and must be controlled before calling it evidence")


if __name__ == "__main__":
    main()
