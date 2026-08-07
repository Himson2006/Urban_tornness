"""Does the pairing penalty fall on the badly-registered buildings?

Across events the penalty tracks misregistration -- Harvey 8.10 px and -0.088
AUC, Michael 5.99 px and -0.019 -- but with three events that is Spearman -0.5
at p=0.67, which establishes nothing.

The same question is answerable per building, where there are thousands rather
than three. The paired and post-only models were trained on the same folds and
scored on the same test set, so for every building we have both models'
probabilities and, from phase correlation, how far apart its two captures are.
If misregistration is the mechanism, the paired model should lose to post-only
specifically on the buildings whose captures are furthest apart, and the two
should be comparable where the pair is well aligned.

The test is a within-building paired comparison stratified by offset, which
holds the building, the event and the fold fixed and varies only the input.

Usage:
    python xbd/misreg_effect.py --event hurricane-michael
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "xbd"))


def load_probs(run: Path) -> pd.DataFrame | None:
    f = run / "test_probs.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    p = z["probs"]
    return pd.DataFrame({"uid": z["uid"].astype(str), "y": z["y"],
                         "p1": p[:, 1]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "xbd/runs")
    ap.add_argument("--crops", type=Path, default=ROOT / "xbd/data/crops")
    ap.add_argument("--event", default="")
    ap.add_argument("--bins", type=int, default=4)
    a = ap.parse_args()

    import cv2
    from registration import shift_of
    from sklearn.metrics import roc_auc_score

    # pair every paired run with the post-only run on the same event and fold
    meta = {}
    for p in sorted(a.runs.glob("*/final_test.json")):
        d = json.loads(p.read_text())
        if d["task"] != "middle" or not d.get("align", True):
            continue
        if a.event and d.get("disaster") != a.event:
            continue
        key = (d.get("disaster") or "POOLED", d["fold"], d["group"])
        meta.setdefault(key, {})["pair" if d["paired"] else "post"] = p.parent
    keys = [k for k, v in meta.items() if {"pair", "post"} <= set(v)]
    if not keys:
        raise SystemExit("no matched pair/post runs found")
    print(f"{len(keys)} matched pair/post run(s)")

    rows = []
    for key in sorted(keys):
        v = meta[key]
        P, Q = load_probs(v["pair"]), load_probs(v["post"])
        if P is None or Q is None:
            print(f"  {key}: missing test_probs.npz; skipping")
            continue
        m = P.merge(Q, on="uid", suffixes=("_pair", "_post"))
        assert (m.y_pair == m.y_post).all(), "test sets differ"
        m["event"], m["fold"] = key[0], key[1]
        rows.append(m)
    d = pd.concat(rows, ignore_index=True)
    print(f"{len(d):,} scored buildings")

    # measure the offset for exactly these buildings, not a separate sample
    cm = pd.read_parquet(a.crops / "crop_meta.parquet").set_index("uid")
    need = [u for u in d.uid.unique() if u in cm.index]
    print(f"measuring registration for {len(need):,} buildings")
    sh = {}
    for n, u in enumerate(need, 1):
        if n % 1000 == 0:
            print(f"  {n:,}/{len(need):,}", flush=True)
        r = cm.loc[u]
        pre = cv2.imread(str(a.crops / r.pre))
        post = cv2.imread(str(a.crops / r.post))
        if pre is None or post is None:
            continue
        dx, dy, _ = shift_of(pre, post)
        if np.isfinite(dx):
            sh[u] = float(np.hypot(dx, dy)) / max(float(r.px_side), 1.0)
    d["shift_rel"] = d.uid.map(sh)
    d = d.dropna(subset=["shift_rel"])
    print(f"{len(d):,} with a measured offset\n")

    print("=== does the paired model lose where the captures disagree? ===")
    d["bin"] = pd.qcut(d.shift_rel, a.bins, labels=False, duplicates="drop")
    print(f"  {'offset (frac of building)':28s} {'n':>6} {'pair AUC':>9} "
          f"{'post AUC':>9} {'diff':>8}")
    diffs = []
    for b, g in d.groupby("bin"):
        if g.y_pair.nunique() < 2:
            continue
        ap_ = roc_auc_score(g.y_pair, g.p1_pair)
        aq_ = roc_auc_score(g.y_post, g.p1_post)
        lo, hi = g.shift_rel.min(), g.shift_rel.max()
        diffs.append((g.shift_rel.median(), ap_ - aq_))
        print(f"  {f'{lo:.3f} - {hi:.3f}':28s} {len(g):6,} {ap_:9.3f} "
              f"{aq_:9.3f} {ap_ - aq_:+8.3f}")
    if len(diffs) >= 3:
        from scipy import stats
        x, y = zip(*diffs)
        r = stats.spearmanr(x, y)
        print(f"\n  penalty vs offset across bins: rho {r.statistic:+.3f} "
              f"(n={len(diffs)} bins)")
    print("  the mechanism predicts the diff becomes more negative as the")
    print("  offset grows: a stack only helps when the channels agree on")
    print("  which pixels are the building.")

    # per-building, the sharpest form: does the pair's error concentrate where
    # the offset is large, holding the building fixed?
    d["pair_err"] = np.abs(d.y_pair - d.p1_pair)
    d["post_err"] = np.abs(d.y_post - d.p1_post)
    d["excess"] = d.pair_err - d.post_err
    from scipy import stats
    r = stats.spearmanr(d.shift_rel, d.excess, nan_policy="omit")
    print(f"\n  per-building: excess error of the pair vs offset, "
          f"rho {r.statistic:+.4f} (p={r.pvalue:.3g}, n={len(d):,})")
    w = stats.wilcoxon(d.pair_err, d.post_err)
    print(f"  pair error {d.pair_err.mean():.4f} vs post {d.post_err.mean():.4f}"
          f"  (Wilcoxon p={w.pvalue:.3g})")

    out = ROOT / "xbd/data/misreg_effect.parquet"
    d.to_parquet(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
