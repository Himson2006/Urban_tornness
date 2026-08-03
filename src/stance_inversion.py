"""The urban result: intent models are most confident where humans most disagree.

Runs the same analysis over any set of held-out prediction files -- ProtoPNet
tornness parquets and plain-CNN baselines alike -- so the claim can be stated
about single-frame intent models rather than about one checkpoint.

Reports per model:

  stance -> disagreement   rho(frac_standing, human_disagreement). The cue.
  confidence inversion     rho(entropy, frac_standing). NEGATIVE means the model
                           gets MORE confident as pedestrians become more
                           behaviourally ambiguous -- the headline.
  human-model coupling     rho(entropy, human_disagreement). Should be positive
                           for a well-aligned model; near zero or negative is
                           the failure.
  silent failure rate      among pedestrians annotators genuinely split on
                           (intention_prob in [0.4, 0.6]), the fraction the
                           model calls with p>0.9. This is the abstract number.

Usage:
    python src/stance_inversion.py
    python src/stance_inversion.py --glob 'runs*/**/*.parquet'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def pedestrian_level(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.in_exp_window] if "in_exp_window" in df else df
    g = d.groupby("ped_id")
    out = pd.DataFrame({
        "entropy": g.entropy.mean(),
        "p_cross": g.p_cross.mean(),
        "human_disagreement": g.human_disagreement.first(),
        "intention_prob": g.intention_prob.first(),
        "intent_binary": g.intent_binary.first(),
        "frac_standing": g.action.apply(lambda s: (s == "standing").mean()),
        "frac_looking": g.look.apply(lambda s: (s == "looking").mean()),
        "bbox_h": g.bbox_h.mean(),
    })
    for c in ("coact_min", "global_max_sim"):
        if c in d.columns:
            out[c] = g[c].mean()
    out["conf"] = (out.p_cross - 0.5).abs() * 2       # 0 = torn, 1 = certain
    return out


def analyse(name: str, kind: str, df: pd.DataFrame) -> dict:
    p = pedestrian_level(df)
    r = lambda x, y: stats.spearmanr(p[x], p[y])
    a = r("frac_standing", "human_disagreement")
    b = r("entropy", "frac_standing")
    c = r("entropy", "human_disagreement")
    split = p[p.intention_prob.between(0.4, 0.6)]
    silent = float((split.conf > 0.8).mean()) if len(split) else np.nan
    row = {"model": name, "kind": kind, "n_ped": len(p),
           "stance_to_disagree": a.statistic, "p_a": a.pvalue,
           "entropy_to_stance": b.statistic, "p_b": b.pvalue,
           "entropy_to_disagree": c.statistic, "p_c": c.pvalue,
           "n_split": len(split), "silent_failure": silent}
    if "coact_min" in p:
        d = r("coact_min", "frac_standing")
        row["coact_to_stance"] = d.statistic
        row["p_d"] = d.pvalue
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto-glob", default="runs/resnet*_fold*/tornness_fold*.parquet")
    ap.add_argument("--plain-glob", default="runs_plain/*/predictions.parquet")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    proto = sorted(ROOT.glob(a.proto_glob))
    if proto:
        df = pd.concat([pd.read_parquet(f) for f in proto], ignore_index=True)
        arch = proto[0].parent.name.split("_")[0]
        rows.append(analyse(f"ProtoPNet-{arch}", "prototype", df))
        print(f"ProtoPNet: pooled {len(proto)} folds")

    plains = {}
    for f in sorted(ROOT.glob(a.plain_glob)):
        plains.setdefault(f.parent.name.split("_")[0], []).append(f)
    for arch, fs in plains.items():
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        rows.append(analyse(f"plain-{arch}", "plain", df))
        print(f"plain-{arch}: pooled {len(fs)} folds")

    if not rows:
        raise SystemExit("no prediction files found; check --proto-glob/--plain-glob")

    res = pd.DataFrame(rows)
    res.to_csv(a.out / "stance_inversion.csv", index=False)

    def star(p):
        return "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else " "

    print("\n=== stance drives human disagreement ===")
    for _, x in res.iterrows():
        print(f"  {x.model:20s} rho={x.stance_to_disagree:+.3f}{star(x.p_a)}"
              f"  (n={x.n_ped:,})")

    print("\n=== CONFIDENCE INVERSION: entropy vs frac_standing ===")
    print("    negative => model gets MORE confident as ambiguity rises\n")
    for _, x in res.iterrows():
        flag = "  <-- INVERTED" if x.entropy_to_stance < 0 and x.p_b < 0.05 else ""
        print(f"  {x.model:20s} rho={x.entropy_to_stance:+.3f}{star(x.p_b)}{flag}")

    print("\n=== model uncertainty vs human uncertainty ===")
    for _, x in res.iterrows():
        print(f"  {x.model:20s} rho={x.entropy_to_disagree:+.3f}{star(x.p_c)}")

    print("\n=== silent failure rate ===")
    print("    pedestrians annotators split on (intention_prob 0.4-0.6)")
    print("    that the model calls with p>0.9\n")
    for _, x in res.iterrows():
        print(f"  {x.model:20s} {x.silent_failure:.1%}  (n={int(x.n_split)})")

    if "coact_to_stance" in res:
        print("\n=== does the prototype layer encode stance at all? ===")
        for _, x in res.dropna(subset=["coact_to_stance"]).iterrows():
            print(f"  {x.model:20s} coact vs stance rho="
                  f"{x.coact_to_stance:+.3f}{star(x.p_d)}")

    inv = res[(res.entropy_to_stance < 0) & (res.p_b < 0.05)]
    print(f"\n=== read ===")
    print(f"  inverted in {len(inv)} of {len(res)} models "
          f"({', '.join(inv.kind.unique()) if len(inv) else 'none'})")
    if len(inv) and set(inv.kind) >= {"prototype", "plain"}:
        print("  Holds for plain CNNs too => a property of single-frame intent")
        print("  models, not of ProtoPNet or of one checkpoint.")
    elif len(inv) and set(inv.kind) == {"prototype"}:
        print("  Only the prototype model inverts => cannot claim this is general.")
    print(f"\nwrote {a.out/'stance_inversion.csv'}")


if __name__ == "__main__":
    main()
