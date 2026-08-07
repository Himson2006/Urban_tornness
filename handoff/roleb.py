"""Role B: is reviewer disagreement recoverable from the image?

Reads `test_preds.parquet` from any run (baseline.py or train.py) and answers
the paper's central question separately for the two kinds of disagreement:

    split_adj    reviewers took opposing positions -- about where the standard
                 sits. Predicted NOT to be image-recoverable.
    unsure_adj   reviewers declined to judge -- about what is visible.
                 Predicted TO BE image-recoverable.

The headline is the DISSOCIATION -- the gap between the two -- not either
coefficient alone. That is deliberate. The imaged subset is range-restricted on
both targets (see below), which drags both correlations toward zero; a bare null
on `split_adj` is therefore ambiguous between real unpredictability and mere
restriction, while the *contrast* survives because restriction hits both.

Four things are done in an order that matters:

  1. **Saturation.** If the uncertainty column takes a handful of distinct
     values it has no usable variance and every number below is noise. On
     pedestrian crops `max_sim` took exactly two values and the correlations
     computed from it were meaningless. This gate runs first and says STOP.

  2. **Cluster bootstrap by panorama.** Crops on one panorama are not
     independent -- same lighting, same camera, often the same pavement.
     Resampling crops would understate the interval; resampling panoramas is
     the honest version.

  3. **Partial correlation.** `unsure_adj` is exactly what a resolution
     detector would predict, and `severity` alone already tracks it at +0.145
     in the full export. Every association is reported raw AND with the
     image-quality covariates regressed out. The partialled figure is the
     result.

  4. **Range-restriction correction.** Assigning correct/incorrect needs
     consensus, so contested labels are under-represented among the crops:
     `split_adj` keeps ~77% of its full-export spread and `unsure_adj` only
     ~66%. The restriction is asymmetric and biases AGAINST the positive half
     of the dissociation, so correcting it is not generosity to our own
     hypothesis. Thorndike case II, reported alongside the uncorrected value,
     never instead of it.

Usage:
    python handoff/roleb.py --run handoff/runs/obstacle_baseline_consensus_pano_resnet34_s0
    python handoff/roleb.py --run <dir> --uncertainty u_epistemic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

COVARIATES = ["severity", "n_val", "image_age_days", "label_age_days",
              "zoom", "pano_width"]
TARGETS = ["split_adj", "unsure_adj"]


def saturation(u: np.ndarray, log=print) -> bool:
    """Does the uncertainty measure vary enough to correlate with anything?"""
    n_uniq = len(np.unique(np.round(u, 6)))
    iqr = float(np.subtract(*np.percentile(u, [75, 25])))
    log("saturation check")
    log(f"  distinct values {n_uniq:,} of {len(u):,}   sd {u.std():.4f}   "
        f"IQR {iqr:.4f}")
    log(f"  range [{u.min():.4f}, {u.max():.4f}]")
    ok = n_uniq >= 50 and u.std() > 1e-3
    log("  PASS -- measure varies" if ok else
        "  STOP -- measure is saturated; nothing below is interpretable")
    return ok


def _ranks(x: pd.Series) -> np.ndarray:
    return stats.rankdata(x.to_numpy(float))


def partial_resid(y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residual of y after least-squares removal of columns C (with intercept)."""
    X = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def rho(d: pd.DataFrame, u: str, t: str, cov: list[str] | None = None):
    """Spearman, optionally partialled on `cov`. Rank-then-residualise."""
    ok = d[u].notna() & d[t].notna()
    if cov:
        for c in cov:
            ok &= d[c].notna()
    s = d[ok]
    if len(s) < 50:
        return float("nan"), len(s)
    a, b = _ranks(s[u]), _ranks(s[t])
    if cov:
        C = np.column_stack([_ranks(s[c]) for c in cov])
        a, b = partial_resid(a, C), partial_resid(b, C)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan"), len(s)
    return float(np.corrcoef(a, b)[0, 1]), int(len(s))


def cluster_boot(d: pd.DataFrame, u: str, t: str, cov: list[str] | None,
                 n_boot: int, seed: int = 0) -> tuple[float, float]:
    """Percentile CI, resampling whole panoramas."""
    rng = np.random.default_rng(seed)
    panos = d.pano_id.dropna().unique()
    idx = {p: g.index.to_numpy() for p, g in d.groupby("pano_id")}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(panos, len(panos), replace=True)
        rows = np.concatenate([idx[p] for p in pick])
        r, _ = rho(d.loc[rows], u, t, cov)
        if np.isfinite(r):
            out.append(r)
    if len(out) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def dissociation(d: pd.DataFrame, u: str, cov: list[str] | None,
                 n_boot: int, seed: int = 0):
    """delta = r(u, unsure_adj) - r(u, split_adj), with a cluster-bootstrap CI.

    This is the paper's headline. A CI excluding zero says the two kinds of
    disagreement behave differently under the same model and the same
    restriction -- which no single correlation can establish.
    """
    r_u, _ = rho(d, u, "unsure_adj", cov)
    r_s, _ = rho(d, u, "split_adj", cov)
    delta = r_u - r_s

    rng = np.random.default_rng(seed)
    panos = d.pano_id.dropna().unique()
    idx = {p: g.index.to_numpy() for p, g in d.groupby("pano_id")}
    out = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[p] for p in
                               rng.choice(panos, len(panos), replace=True)])
        s = d.loc[rows]
        a, _ = rho(s, u, "unsure_adj", cov)
        b, _ = rho(s, u, "split_adj", cov)
        if np.isfinite(a) and np.isfinite(b):
            out.append(a - b)
    lo, hi = ((np.percentile(out, 2.5), np.percentile(out, 97.5))
              if len(out) >= n_boot // 2 else (np.nan, np.nan))
    return delta, r_u, r_s, float(lo), float(hi)


def thorndike(r: float, sd_restricted: float, sd_full: float) -> float:
    """Case II correction for restriction on the target.

    Returns the correlation expected had the imaged subset spanned the full
    export's spread. Assumes linearity and that restriction is the only
    difference -- both are approximations, which is why the raw value is always
    reported next to it.
    """
    if not np.isfinite(r) or sd_restricted <= 0 or sd_full <= 0:
        return float("nan")
    k = sd_full / sd_restricted
    denom = np.sqrt(1 + r**2 * (k**2 - 1))
    return float(np.clip(r * k / denom, -1, 1))


def min_detectable(n: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest |r| detectable at this n -- the yardstick a null is read against."""
    if n < 10:
        return float("nan")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(z / np.sqrt(n - 3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--targets", type=Path,
                    default=ROOT / "handoff/data/targets.parquet")
    ap.add_argument("--uncertainty", default="uncertainty",
                    help="column to test; try u_epistemic / u_aleatoric")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_parquet(a.run / "test_preds.parquet").reset_index(drop=True)
    if a.uncertainty not in d.columns:
        raise SystemExit(f"no column {a.uncertainty!r}; have "
                         f"{[c for c in d.columns if c.startswith('u')]}")
    lines: list[str] = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log(f"=== Role B: {a.run.name} ===")
    log(f"{len(d):,} test crops, {d.pano_id.nunique():,} panoramas, "
        f"{d.city.nunique()} cities")
    log(f"uncertainty column: {a.uncertainty}")
    log(f"contested {int(d.contested.sum()):,}   unclear {int(d.unclear.sum()):,}")
    log()

    if not saturation(d[a.uncertainty].to_numpy(float), log):
        (a.run / "roleb.txt").write_text("\n".join(lines))
        raise SystemExit(1)
    log()

    cov = [c for c in COVARIATES if c in d.columns and d[c].notna().any()
           and d[c].nunique() > 2]
    log(f"covariates partialled: {', '.join(cov)}")
    log(f"minimum detectable |r| at n={len(d):,}, 80% power: "
        f"{min_detectable(len(d)):.3f}")
    log()

    full = pd.read_parquet(a.targets)
    res = {}

    log(f"  {'target':11s} {'raw r':>8} {'95% CI':>18} {'partial r':>10} "
        f"{'95% CI':>18} {'restr-corr':>11}")
    for t in TARGETS:
        r_raw, n = rho(d, a.uncertainty, t)
        lo1, hi1 = cluster_boot(d, a.uncertainty, t, None, a.n_boot, a.seed)
        r_par, _ = rho(d, a.uncertainty, t, cov)
        lo2, hi2 = cluster_boot(d, a.uncertainty, t, cov, a.n_boot, a.seed)
        corr = thorndike(r_par, d[t].std(), full[t].std())
        log(f"  {t:11s} {r_raw:+8.3f} [{lo1:+.3f},{hi1:+.3f}] {r_par:+10.3f} "
            f"[{lo2:+.3f},{hi2:+.3f}] {corr:+11.3f}")
        res[t] = {"n": n, "raw": r_raw, "raw_ci": [lo1, hi1],
                  "partial": r_par, "partial_ci": [lo2, hi2],
                  "restriction_corrected": corr,
                  "sd_test": float(d[t].std()),
                  "sd_full_export": float(full[t].std())}

    log()
    log("range restriction (imaged subset vs full export)")
    for t in TARGETS:
        k = res[t]["sd_test"] / res[t]["sd_full_export"]
        log(f"  {t:11s} sd {res[t]['sd_test']:.4f} vs "
            f"{res[t]['sd_full_export']:.4f}  -> {k:.1%} of spread retained")
    log("  restriction drags both correlations toward zero, which is why the")
    log("  dissociation below -- a within-sample contrast -- is the headline")
    log()

    for tag, c in (("raw", None), ("partialled", cov)):
        delta, r_u, r_s, lo, hi = dissociation(d, a.uncertainty, c,
                                               a.n_boot, a.seed)
        sig = "excludes 0" if np.isfinite(lo) and (lo > 0 or hi < 0) else \
              "includes 0"
        log(f"DISSOCIATION ({tag}): r(unsure) - r(split) = "
            f"{r_u:+.3f} - {r_s:+.3f} = {delta:+.3f}")
        log(f"  95% CI [{lo:+.3f}, {hi:+.3f}]  -- {sig}")
        res[f"dissociation_{tag}"] = {"delta": delta, "r_unsure": r_u,
                                      "r_split": r_s, "ci": [lo, hi]}
    log()
    log("reading: a positive delta whose CI excludes zero says the model")
    log("recovers 'reviewers could not see it' but not 'reviewers disagreed',")
    log("which is the paper's claim. A delta near zero with both correlations")
    log("near zero is the HICSS-shaped negative and is still publishable, but")
    log("weaker -- restriction cannot then be ruled out as the cause.")

    (a.run / "roleb.json").write_text(json.dumps(res, indent=2))
    (a.run / "roleb.txt").write_text("\n".join(lines))
    print(f"\nwrote {a.run/'roleb.json'}")


if __name__ == "__main__":
    main()
