"""Is the overconfidence effect specific to one learner?

calibration.py showed that training on the consensus-only record makes a
gradient-boosted model far more certain about labels reviewers split on. If
that were a quirk of boosting -- or of model capacity -- it would not be worth
much. This repeats the matched-size comparison across three families, including
a linear one.

Usage:
    python sidewalk/calibration_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidewalk"))
from calibration import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--folds", type=int, default=3)
    a = ap.parse_args()

    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = pd.read_parquet(a.data / "labels.parquet")
    d = d[d.region_id.notna()].reset_index(drop=True)
    X, y = build_features(d)
    X, y = X.to_numpy(float), y.to_numpy()
    groups = (d.city + "_" + d.region_id.astype(str)).to_numpy()
    contested = d.is_split.to_numpy()
    keep = (((d.agree_count > 2) & (d.disagree_count <= 2))
            | ((d.disagree_count > 2) & (d.agree_count <= 2))).to_numpy()

    models = {
        "gradboost": lambda: HistGradientBoostingClassifier(max_iter=150,
                                                            random_state=0),
        "randforest": lambda: RandomForestClassifier(
            n_estimators=120, min_samples_leaf=20, n_jobs=-1, random_state=0),
        "logistic": lambda: make_pipeline(
            SimpleImputer(), StandardScaler(),
            LogisticRegression(max_iter=1000)),
    }
    rng = np.random.default_rng(0)
    rows = []
    print(f"  {'model':11s} {'regime':10s} {'AUROC':>7} {'p>0.9 contested':>17}")
    for mn, mk in models.items():
        P = {k: np.full(len(d), np.nan) for k in ("filtered", "full_sub")}
        for tr, te in GroupKFold(n_splits=a.folds).split(X, y, groups):
            sub = tr[keep[tr]]
            # matched size, but drawn from the full record so the ambiguous
            # middle is present
            idx = rng.choice(tr, size=len(sub), replace=False)
            for nm, ii in (("filtered", sub), ("full_sub", idx)):
                m = mk()
                m.fit(X[ii], y[ii])
                P[nm][te] = m.predict_proba(X[te])[:, 1]
        rec = {"model": mn}
        for nm, p in P.items():
            ok = ~np.isnan(p)
            c = contested & ok
            auc, silent = roc_auc_score(y[ok], p[ok]), (p[c] > 0.9).mean()
            rec[f"{nm}_auroc"], rec[f"{nm}_silent"] = auc, silent
            print(f"  {mn:11s} {nm:10s} {auc:7.4f} {silent:17.1%}")
        rec["gap"] = rec["filtered_silent"] - rec["full_sub_silent"]
        rows.append(rec)
        print(f"  {'':11s} {'-> gap':10s} {'':7s} {rec['gap']:+17.1%}\n")

    r = pd.DataFrame(rows)
    r.to_csv(a.data / "calibration_models.csv", index=False)
    print(f"  gap positive in {(r.gap > 0).sum()}/{len(r)} model families; "
          f"filtered AUROC lower in "
          f"{(r.filtered_auroc < r.full_sub_auroc).sum()}/{len(r)}")
    print(f"\nwrote {a.data/'calibration_models.csv'}")


if __name__ == "__main__":
    main()
