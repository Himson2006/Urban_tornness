"""Build the Role B target table: what the model is asked to predict about
reviewer disagreement, with reviewer tendency removed.

This is the analogue of LIDC's radiologist-rating standard deviation, but it is
two quantities rather than one, and that is the point of the study:

  split_adj    reviewers took opposite positions. A disagreement about where
               the accessibility standard sits. NOT expected to be recoverable
               from the image -- it is a statement about people, not pixels.

  unsure_adj   reviewers individually declined to judge. A disagreement about
               what is visible. EXPECTED to be recoverable -- occlusion, low
               resolution and stale imagery are image properties.

A method whose uncertainty tracks the second and not the first is the result
the paper is after: a dissociation, not a flat null.

ONE CORRECTION APPLIES HERE. `label_id` in the Project Sidewalk exports is
assigned per deployment, so it is unique only within a city: 34,753 ids in this
pull appear in two or more cities, e.g. id=8 is a CurbRamp in Amsterdam, a
CurbRamp in Newberg and a Crosswalk in Taipei. Grouping on `label_id` alone --
as sidewalk/decompose.py does -- fuses those into a single pseudo-label and
fits one difficulty effect across unrelated sidewalks in different countries.
Every key here is therefore `city:label_id`. The `--compare-pooled` flag
refits the old way and reports how far the two disagree, because the affected
numbers also appear in paper/main.tex.

Usage:
    python handoff/targets.py
    python handoff/targets.py --compare-pooled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidewalk"))

from decompose import shrink, two_way_fit  # noqa: E402

# The contested pairs. Named to match xbd/dataset.py TASKS so the two studies
# can be read against each other: a "middle" task is a threshold judgement,
# an "extremes" task is a category difference and acts as the control.
TASKS = {
    # is this corner's ramp adequate, or is it effectively missing? The call a
    # city actually pays for.
    "ramp": ["CurbRamp", "NoCurbRamp"],
    # is a heaved slab a bad surface or a thing in the way? Genuinely contested
    # -- the same physical defect gets both labels.
    "surface": ["SurfaceProblem", "Obstacle"],
    # control: visually unrelated. Tornness here should be markedly lower for
    # the same architecture, or the measure is not tracking ambiguity.
    "extremes": ["CurbRamp", "Signal"],
}


def add_uid(d: pd.DataFrame) -> pd.DataFrame:
    """City-scoped label key. See the module docstring."""
    if "city" not in d.columns:
        raise SystemExit("frame has no `city` column; cannot build a safe key")
    d = d.copy()
    d["uid"] = d.city.astype(str) + ":" + d.label_id.astype(str)
    return d


def fit_effects(v: pd.DataFrame, outcome: str, key: str,
                mask: np.ndarray | None = None) -> pd.DataFrame:
    """Per-label propensity for `outcome`, with reviewer tendency partialled out.

    Identical in form to sidewalk/decompose.py:analyse, but keyed on `key` so
    the caller chooses between the corrected and pooled groupings.
    """
    d = v if mask is None else v[mask]
    y = (d.vote == outcome).to_numpy(float)
    lcat = d[key].astype("category")
    rcat = d.validator_id.astype("category")
    li, ri = lcat.cat.codes.to_numpy(), rcat.cat.codes.to_numpy()

    mu, a, b, lab_n, _ = two_way_fit(
        y, li, ri, len(lcat.cat.categories), len(rcat.cat.categories))
    resid_var = (y - mu - a[li] - b[ri]).var()

    var_lab, var_rev = a[li].var(), b[ri].var()
    tot = var_lab + var_rev + resid_var
    print(f"  {outcome:9s} base={mu:.3f}  label={var_lab/tot:5.1%}  "
          f"reviewer={var_rev/tot:5.1%}  unexplained={resid_var/tot:5.1%}  "
          f"({len(lcat.cat.categories):,} labels)")

    tag = outcome.lower()
    return pd.DataFrame({
        key: lcat.cat.categories,
        f"{tag}_raw": mu + a,
        f"{tag}_adj": mu + shrink(a, lab_n, resid_var),
        f"{tag}_n": lab_n,
    })


def build(lab: pd.DataFrame, val: pd.DataFrame, key: str) -> pd.DataFrame:
    print(f"\nfitting on `{key}`")
    uns = fit_effects(val, "Unsure", key)
    took = (val.vote != "Unsure").to_numpy()
    dis = fit_effects(val, "Disagree", key, took)
    eff = uns.merge(dis, on=key, how="outer")

    out = lab.merge(eff, on=key, how="inner")
    # `split` in labels.parquet is the raw vote balance; the reviewer-adjusted
    # counterpart is the disagree propensity among reviewers who took a
    # position, which is what disagree_adj already estimates.
    #
    # A label every reviewer marked "Unsure" has no position-takers, so it gets
    # no disagree effect and split_adj is undefined -- not zero. Left as 0 it
    # would read as "reviewers agreed", the opposite of the truth. Keep it NaN
    # and let each analysis drop it explicitly.
    out["split_adj"] = 1 - (out.disagree_adj - 0.5).abs() * 2
    return out


def _rho(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Spearman on complete pairs only, with the count kept alongside."""
    from scipy import stats

    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 100 or a[ok].nunique() < 3 or b[ok].nunique() < 3:
        return float("nan"), int(ok.sum())
    return float(stats.spearmanr(a[ok], b[ok]).statistic), int(ok.sum())


def confounders(d: pd.DataFrame) -> list[str]:
    """Image-quality covariates every association must be partialled on.

    The lesson from xbd/tornness.py: a measure that only recovers "this crop is
    poor" is a resolution detector wearing a nicer name. `unsure_adj` is
    especially exposed, since bad imagery is the thing it is supposed to mean.
    """
    return [c for c in ("image_age_days", "label_age_days", "zoom", "severity",
                        "pano_width", "n_val") if c in d.columns]


def report(d: pd.DataFrame) -> None:
    print(f"\n=== {len(d):,} labels, {d.city.nunique()} cities ===")
    n_split = int(d.split_adj.notna().sum())
    print(f"split_adj defined on {n_split:,} labels "
          f"({n_split/len(d):.1%}); the rest drew no position-takers")
    rho, n = _rho(d.split_adj, d.unsure_adj)
    print(f"split_adj vs unsure_adj: rho={rho:+.3f} (n={n:,})")
    print("  (near zero = two phenomena, which is the premise of the study;")
    print("   if it were high, one target would do and the paper loses its hook)")

    print("\nrole A sampling frame -- labels available per task")
    for task, classes in TASKS.items():
        sub = d[d.label_type.isin(classes)]
        counts = sub.label_type.value_counts()
        n = " / ".join(f"{c}={counts.get(c,0):,}" for c in classes)
        print(f"  {task:9s} {len(sub):>7,}   {n}")

    print("\nrole B target spread by label type")
    print(f"  {'type':16s} {'n':>8} {'split_adj':>10} {'unsure_adj':>11}")
    for t, g in d.groupby("label_type"):
        if len(g) < 500:
            continue
        print(f"  {str(t)[:16]:16s} {len(g):8,} {g.split_adj.mean():10.3f} "
              f"{g.unsure_adj.mean():11.3f}")

    cov = confounders(d)
    print(f"\ncovariates to partial on: {', '.join(cov)}")
    print(f"  {'covariate':16s} {'vs split_adj':>13} {'vs unsure_adj':>14}")
    for c in cov:
        r1, _ = _rho(d[c], d.split_adj)
        r2, _ = _rho(d[c], d.unsure_adj)
        print(f"  {c:16s} {r1:+13.3f} {r2:+14.3f}")
    print("  a covariate that already predicts unsure_adj is doing the work the")
    print("  model would otherwise be credited with -- partial it out first")


def compare_pooled(lab: pd.DataFrame, val: pd.DataFrame,
                   corrected: pd.DataFrame) -> None:
    """Refit the old way and quantify what the collision changed."""
    from scipy import stats

    print("\n" + "=" * 62)
    print("POOLED REFIT -- grouping on bare label_id, as sidewalk/decompose.py")
    print("=" * 62)
    dup = lab.groupby("label_id").city.nunique()
    print(f"label_ids spanning >1 city: {(dup > 1).sum():,} of {len(dup):,}")

    pooled = build(lab, val, "label_id")
    m = corrected[["uid", "label_id", "city", "unsure_adj", "split_adj"]].merge(
        pooled[["label_id", "unsure_adj", "split_adj"]],
        on="label_id", how="inner", suffixes=("_fix", "_pool"))

    print(f"\ncomparable rows: {len(m):,}")
    for t in ("unsure_adj", "split_adj"):
        a, b = m[f"{t}_fix"], m[f"{t}_pool"]
        rho, n = _rho(a, b)
        delta = (a - b).abs()
        print(f"  {t:11s} rho(fixed, pooled)={rho:+.3f} (n={n:,})   "
              f"mean|delta|={delta.mean():.4f}   max|delta|={delta.max():.4f}")
    print("\nrho well below 1 means the ranking of 'difficult' labels changes.")
    print("Numbers derived from the pooled fit -- including those in")
    print("paper/main.tex -- should be regenerated before submission.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--out", type=Path, default=ROOT / "handoff/data")
    ap.add_argument("--compare-pooled", action="store_true",
                    help="refit on the colliding key and report the difference")
    a = ap.parse_args()

    lab = add_uid(pd.read_parquet(a.data / "labels.parquet"))
    val = add_uid(pd.read_parquet(a.data / "validations.parquet"))
    print(f"{len(lab):,} labels  {len(val):,} reviews  "
          f"{val.validator_id.nunique():,} reviewers")

    d = build(lab, val, "uid")
    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / "targets.parquet"
    d.to_parquet(dest, index=False)
    report(d)
    print(f"\nwrote {dest} ({len(d):,} rows)")

    if a.compare_pooled:
        compare_pooled(lab, val, d)


if __name__ == "__main__":
    main()
