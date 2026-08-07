"""Pool the folds into the numbers the paper actually claims.

Three claims, each tested here the way it will have to be defended:

  1. **Reducibility is graded by failure mode.** Flood damage happens inside the
     building; wind damage happens to the roof; destruction is unmistakable. If
     that is right, AUC should order flood < wind < destruction with gaps larger
     than the spread across folds.

  2. **Channel-stacking the pair does not help.** Folds are matched -- fold k of
     an event is the same test set for both inputs -- so this is a paired
     comparison, not two independent samples. Reported as a paired difference
     with a Wilcoxon test, and alongside the misregistration that explains it.

  3. **Pooling manufactures skill.** A model trained across events beats the
     pooled majority while losing to the within-event ones. The comparison is
     between pooled runs and the per-event runs on the same task.

Everything is per event. Pooling has produced a spurious result three times in
this project, and every one of them looked tidier than the truth.

Usage:
    python xbd/aggregate.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def ci95(v: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of a t-based 95% interval; nan when n < 2."""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (float(v[0]) if len(v) else np.nan), np.nan
    from scipy import stats
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) *
                                  v.std(ddof=1) / np.sqrt(len(v)))


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(runs.glob("*/final_test.json")):
        d = json.loads(p.read_text())
        loc = None
        lp = p.parent / "localization.parquet"
        if lp.exists():
            L = pd.read_parquet(lp)
            loc = float(L[L.shown].inside.mean()) if "shown" in L else None
            chance = float(L[L.shown].chance.mean()) if "shown" in L else None
        else:
            chance = None
        tp = p.parent / "tornness.parquet"
        coact_rho = np.nan
        if tp.exists():
            T = pd.read_parquet(tp)
            if {"coact", "margin"} <= set(T.columns) and len(T) > 30:
                from scipy import stats
                coact_rho = float(stats.spearmanr(
                    T.coact, T.margin, nan_policy="omit").statistic)
        rows.append({
            "tag": d["tag"], "task": d["task"],
            # a held-out-event run is not "pooled" -- it is a different
            # experiment, and averaging it into the pooled rows produced a
            # 0.649 +- 0.348 that described nothing
            "event": (d.get("disaster")
                      or ("HELD-OUT-EVENT" if d["group"] == "disaster"
                          else "POOLED")),
            "input": "pair" if d["paired"] else "post",
            "aligned": d.get("align", True), "group": d["group"],
            "fold": d["fold"], "n_test": d["n_test"],
            "acc": d["test_acc"], "majority": d["majority"],
            "bal": d.get("balanced_acc", np.nan), "auc": d.get("auc", np.nan),
            "on_bld": loc, "loc_chance": chance, "coact_rho": coact_rho})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "xbd/runs")
    ap.add_argument("--registration", type=Path,
                    default=ROOT / "xbd/data/registration.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "xbd/data/results.csv")
    a = ap.parse_args()

    d = load(a.runs)
    if d.empty:
        raise SystemExit(f"no finished runs under {a.runs}")
    d.to_csv(a.out, index=False)
    print(f"{len(d)} runs | tasks {sorted(d.task.unique())} | "
          f"events {sorted(d.event.unique())}\n")

    print("=== 1. is reducibility graded by failure mode? ===")
    print(f"  {'task':9s} {'event':22s} {'input':6s} {'k':>2} "
          f"{'AUC':>16} {'bal acc':>16}")
    d["cond"] = d.event + np.where(d["aligned"], "", " (raw)")
    for (task, ev, inp), g in d.groupby(["task", "cond", "input"]):
        am, ac = ci95(g.auc.values)
        bm, bc = ci95(g.bal.values)
        ai = f"{am:.3f} +- {ac:.3f}" if np.isfinite(ac) else f"{am:.3f}   (n=1)"
        bi = f"{bm:.3f} +- {bc:.3f}" if np.isfinite(bc) else f"{bm:.3f}   (n=1)"
        print(f"  {task:9s} {str(ev)[:22]:22s} {inp:6s} {len(g):2d} "
              f"{ai:>16} {bi:>16}")
    print("  0.500 is chance. A flood event at chance means the evidence is")
    print("  not in the imagery -- ambiguity no model or resolution can fix.")

    print("\n=== 2. does stacking the pair help? (paired over folds) ===")
    mid = d[(d.task == "middle") & d["aligned"]]
    piv = mid.pivot_table(index=["event", "fold"], columns="input",
                          values="auc")
    piv = piv.dropna(subset=["pair", "post"]) if {"pair", "post"} <= set(piv) \
        else pd.DataFrame()
    if piv.empty:
        print("  need both inputs on matched folds; run scripts/run_xbd_events.sh")
    else:
        reg = None
        if a.registration.exists():
            R = pd.read_parquet(a.registration)
            reg = R.groupby("disaster")["shift"].median()
        print(f"  {'event':22s} {'k':>2} {'pair':>7} {'post':>7} {'diff':>8} "
              f"{'misreg px':>10}")
        for ev, g in piv.groupby(level=0):
            diff = (g["pair"] - g["post"]).values
            m, c = ci95(diff)
            r = reg.get(ev, np.nan) if reg is not None else np.nan
            print(f"  {str(ev)[:22]:22s} {len(g):2d} {g['pair'].mean():7.3f} "
                  f"{g['post'].mean():7.3f} {m:+8.3f} {r:10.2f}")
        alld = (piv["pair"] - piv["post"]).values
        m, c = ci95(alld)
        from scipy import stats
        w = stats.wilcoxon(piv["pair"], piv["post"]) if len(alld) >= 6 else None
        print(f"  {'ALL':22s} {len(alld):2d} {piv['pair'].mean():7.3f} "
              f"{piv['post'].mean():7.3f} {m:+8.3f}"
              + (f"   Wilcoxon p={w.pvalue:.3g}" if w else "   (n<6)"))
        print("  negative diff = the pre image makes discrimination worse.")
        if reg is not None and len(piv.groupby(level=0)) >= 3:
            per = piv.groupby(level=0).apply(
                lambda g: (g["pair"] - g["post"]).mean())
            common = [e for e in per.index if e in reg.index]
            if len(common) >= 3:
                rho = stats.spearmanr(per[common], reg[common])
                print(f"  penalty vs misregistration across events: "
                      f"rho {rho.statistic:+.3f} (p={rho.pvalue:.3g}, "
                      f"n={len(common)})")
                print("  the prediction: worse alignment, bigger penalty")

    print("\n=== 3. does pooling manufacture skill? ===")
    pool = d[(d.task == "middle") & (d.event == "POOLED") & d["aligned"] &
             (d.group == "scene")]
    within = d[(d.task == "middle") & (d.event != "POOLED") & d["aligned"]]
    for inp in ["pair", "post"]:
        p, w = pool[pool.input == inp], within[within.input == inp]
        if p.empty or w.empty:
            continue
        pm, pc = ci95(p.auc.values)
        # weight events by test size: a per-event mean would let Florence count
        # as much as Harvey
        wm = float((w.auc * w.n_test).sum() / w.n_test.sum())
        print(f"  {inp:5s} pooled AUC {pm:.3f}"
              + (f" +- {pc:.3f}" if np.isfinite(pc) else "")
              + f"   within-event (size-weighted) {wm:.3f}"
              f"   inflation {pm - wm:+.3f}")
    held = d[(d.group == "disaster")]
    if not held.empty:
        hm, hc = ci95(held.auc.values)
        print(f"  held-out event AUC {hm:.3f}"
              + (f" +- {hc:.3f}" if np.isfinite(hc) else "")
              + "   (0.500 = no transfer at all)")

    print("\n=== 4. do the explanations land on the building? ===")
    if d["on_bld"].notna().any():
        print(f"  {'task':9s} {'event':22s} {'input':6s} {'on-building':>13} "
              f"{'chance':>8} {'coact rho':>10}")
        for (task, ev, inp), g in d[d["on_bld"].notna()].groupby(
                ["task", "event", "input"]):
            lm, lc = ci95(g["on_bld"].values)
            print(f"  {task:9s} {str(ev)[:22]:22s} {inp:6s} "
                  + (f"{lm:.3f} +- {lc:.3f}" if np.isfinite(lc)
                     else f"{lm:.3f}       ").rjust(13)
                  + f" {g.loc_chance.mean():8.3f} {g.coact_rho.mean():10.3f}")
        print("  coact rho is co-activation against margin: negative means")
        print("  torn buildings resemble both classes, which is the claim.")
    else:
        print("  no localization.parquet yet -- run prototype_localization.py")

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
