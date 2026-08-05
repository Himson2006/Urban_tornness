"""Does quality filtering manufacture overconfidence?

The audit shows the platform's filter removes over 90% of labels that reviewers
contested. Downstream models are trained on what survives. If removing the
contested cases also removes the model's opportunity to learn that some cases
are genuinely uncertain, then routine data cleaning produces a system that is
confident precisely where people disagree.

That is testable here without touching an image. We predict whether a label
will be judged correct, from metadata available at labelling time, and compare
three training regimes on an identical held-out set:

  filtered  train only on labels the quality filter keeps (current practice)
  full      train on every reviewed label, hard majority target
  soft      train on every reviewed label, target = agree/(agree+disagree)

The measure that matters is the one from our earlier work on pedestrian intent:
among held-out labels that reviewers split on, what fraction does the model
call with p > 0.9? A model that has never seen a contested case has no reason
to hesitate on one.

Grouping: folds are split by region so no neighbourhood appears in both train
and test, and the labeller-history features are computed leave-one-out.

Usage:
    python sidewalk/calibration.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOP_TAGS = 25


def build_features(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Everything knowable at labelling time. No review counts -- that is the target."""
    d = d.copy()
    d["tags"] = d.tags.fillna("")
    X = pd.DataFrame(index=d.index)

    X["severity"] = pd.to_numeric(d.severity, errors="coerce")
    X["zoom"] = pd.to_numeric(d.zoom, errors="coerce")
    X["canvas_w"] = pd.to_numeric(d.canvas_width, errors="coerce")
    X["canvas_h"] = pd.to_numeric(d.canvas_height, errors="coerce")
    X["heading"] = pd.to_numeric(d.heading, errors="coerce")
    X["pitch"] = pd.to_numeric(d.pitch, errors="coerce")
    X["image_age"] = d.get("image_age_days", np.nan)
    X["label_age"] = d.get("label_age_days", np.nan)
    X["n_tags"] = d.tags.str.count(",").fillna(0) + (d.tags.str.len() > 2).astype(int)
    X["has_desc"] = d.description.notna().astype(int) if "description" in d else 0

    for c in ("label_type", "city"):
        X[c] = d[c].astype("category").cat.codes

    tags = d.tags.str.strip("[]").str.split(",").explode().str.strip()
    top = tags[tags.ne("")].value_counts().head(TOP_TAGS).index
    for t in top:
        X[f"tag_{t[:18]}"] = d.tags.str.contains(t, regex=False).astype(int)

    # Labeller history, leave-one-out so a label never informs its own feature.
    ok = (d.agree_count > d.disagree_count).astype(float)
    grp = d.groupby("user_id")
    s, n = grp.ok.transform("sum") if "ok" in d else (
        ok.groupby(d.user_id).transform("sum"), ok.groupby(d.user_id).transform("size"))
    X["labeller_n"] = n - 1
    X["labeller_rate"] = np.where(n > 1, (s - ok) / np.maximum(n - 1, 1), np.nan)

    y = (d.agree_count > d.disagree_count).astype(int)
    return X, y


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss

    d = pd.read_parquet(a.data / "labels.parquet")
    d = d[d.region_id.notna()].reset_index(drop=True)
    X, y = build_features(d)
    groups = (d.city + "_" + d.region_id.astype(str)).to_numpy()
    contested = d.is_split.to_numpy()
    soft = (d.agree_count / (d.agree_count + d.disagree_count).clip(lower=1)).to_numpy()
    # A real pipeline keeps consensus on BOTH sides and drops the ambiguous
    # middle. The published rule (agree>2 & disagree<=2) selects positives; its
    # mirror selects negatives. Labels reviewers split on satisfy neither.
    qc_pos = ((d.agree_count > 2) & (d.disagree_count <= 2)).to_numpy()
    qc_neg = ((d.disagree_count > 2) & (d.agree_count <= 2)).to_numpy()
    keep = qc_pos | qc_neg

    print(f"{len(d):,} reviewed labels, {X.shape[1]} features, "
          f"{len(np.unique(groups)):,} regions")
    print(f"  consensus-positive {qc_pos.mean():.1%}, consensus-negative "
          f"{qc_neg.mean():.1%}, ambiguous middle {1-keep.mean():.1%}")
    print(f"  filtered training set: {keep.sum():,} labels, "
          f"{y[keep].mean():.1%} correct")
    print(f"  full training set    : {len(d):,} labels, {y.mean():.1%} correct")
    print(f"  held-out contested labels are the test: {contested.sum():,}\n")

    rows = []
    gkf = GroupKFold(n_splits=a.folds)
    # The filtered set is smaller, so part of any confidence gap could be
    # sample size rather than composition. `full_sub` subsamples the full
    # record to the filtered set's size, keeping the ambiguous middle.
    rng = np.random.default_rng(0)
    preds = {k: np.full(len(d), np.nan)
             for k in ("filtered", "full_sub", "full", "soft")}
    for tr, te in gkf.split(X, y, groups):
        n_keep = int(keep[tr].sum())
        sub_idx = rng.choice(tr, size=min(n_keep, len(tr)), replace=False)
        for name in ("filtered", "full_sub", "full", "soft"):
            if name == "filtered":
                sub = tr[keep[tr]]
                if len(np.unique(y[sub])) < 2:
                    continue
                m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
                m.fit(X.iloc[sub], y[sub])
                preds[name][te] = m.predict_proba(X.iloc[te])[:, 1]
            elif name == "full_sub":
                m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
                m.fit(X.iloc[sub_idx], y[sub_idx])
                preds[name][te] = m.predict_proba(X.iloc[te])[:, 1]
            elif name == "full":
                m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
                m.fit(X.iloc[tr], y[tr])
                preds[name][te] = m.predict_proba(X.iloc[te])[:, 1]
            else:
                m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
                m.fit(X.iloc[tr], soft[tr])
                preds[name][te] = np.clip(m.predict(X.iloc[te]), 0, 1)

    print(f"{'regime':10s} {'AUROC':>7} {'Brier':>7} {'ECE':>7} "
          f"{'mean conf':>10} {'conf|contested':>15} {'p>0.9 contested':>16}")
    for name, p in preds.items():
        ok = ~np.isnan(p)
        conf = np.abs(p - 0.5) * 2
        c = contested & ok
        rows.append({
            "regime": name, "auroc": roc_auc_score(y[ok], p[ok]),
            "brier": brier_score_loss(y[ok], p[ok]), "ece": ece(p[ok], y[ok]),
            "mean_conf": conf[ok].mean(), "conf_contested": conf[c].mean(),
            "silent": (p[c] > 0.9).mean(),
        })
        r = rows[-1]
        print(f"{name:10s} {r['auroc']:7.4f} {r['brier']:7.4f} {r['ece']:7.4f} "
              f"{r['mean_conf']:10.3f} {r['conf_contested']:15.3f} "
              f"{r['silent']:16.1%}")

    res = pd.DataFrame(rows)
    res.to_csv(a.data / "calibration.csv", index=False)
    f, s = res.set_index("regime").loc["filtered"], res.set_index("regime").loc["soft"]
    print(f"\n  filtered vs soft on contested labels: "
          f"confidence {f.conf_contested:.3f} vs {s.conf_contested:.3f}, "
          f"p>0.9 rate {f.silent:.1%} vs {s.silent:.1%}")
    if f.silent > s.silent:
        print("  => training on the filtered record makes the model more certain")
        print("     about exactly the labels people could not agree on.")
    else:
        print("  => no overconfidence effect; the filtering does not appear to")
        print("     drive confidence on contested cases in this setup.")
    print(f"\nwrote {a.data/'calibration.csv'}")


if __name__ == "__main__":
    main()
