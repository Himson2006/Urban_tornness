"""CIFAR-10H: the only dataset here that can test the *shape* claim.

PIE releases one aggregated `intention_prob` per pedestrian, so "annotators split
into two camps" and "every annotator was individually unsure" collapse to the
same number. That is exactly the distinction the thesis is about, and Experiment
1's null on PIE cannot separate them.

CIFAR-10H gives ~51 individual annotator votes per image over the CIFAR-10 test
set, so the full human label distribution is recoverable. From it:

  disagree    1 - top1 share            how much humans disagreed (PIE has this)
  top2_mass   top1 + top2               mass in exactly two classes
  split_ratio top2 / top1               how evenly those two are matched
  n_eff       exp(entropy)              how many classes are genuinely in play

BIMODAL (the dual-match analogue) = high top2_mass AND high split_ratio: two
strong competing readings. DIFFUSE (the weak-match analogue) = low top2_mass:
mass smeared over many classes, nobody has a second opinion so much as no
opinion. `disagree` alone cannot tell these apart -- that is the whole point.

Usage:
    python src/cifar10h_data.py
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COUNTS_URL = ("https://github.com/jcpeterson/cifar-10h/raw/master/data/"
              "cifar10h-counts.npy")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def human_metrics(counts: np.ndarray) -> pd.DataFrame:
    q = counts / counts.sum(1, keepdims=True)
    order = np.argsort(-q, axis=1)
    s = np.take_along_axis(q, order, axis=1)
    top1, top2 = s[:, 0], s[:, 1]
    safe = np.where(q > 0, q, 1.0)
    ent = -(q * np.log(safe)).sum(1)
    return pd.DataFrame({
        "idx": np.arange(len(q)),
        "n_annotators": counts.sum(1).astype(int),
        "human_label": order[:, 0],
        "runner_up": order[:, 1],
        "top1": top1,
        "top2": top2,
        "disagree": 1 - top1,
        "top2_mass": top1 + top2,
        "split_ratio": np.divide(top2, top1, out=np.zeros_like(top2),
                                 where=top1 > 0),
        "human_entropy": ent,
        "n_eff": np.exp(ent),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/cifar10h")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    npy = a.out / "cifar10h-counts.npy"
    if not npy.exists():
        print(f"downloading {COUNTS_URL}")
        urllib.request.urlretrieve(COUNTS_URL, npy)
    counts = np.load(npy).astype(float)
    print(f"counts {counts.shape}, {int(np.median(counts.sum(1)))} annotators/image "
          f"(median)")

    df = human_metrics(counts)
    df["human_label_name"] = [CLASSES[i] for i in df.human_label]
    df["runner_up_name"] = [CLASSES[i] for i in df.runner_up]
    df.to_parquet(a.out / "human_labels.parquet", index=False)

    print("\n=== disagreement population ===")
    for t in (0.99, 0.95, 0.90, 0.80, 0.70):
        n = int((df.top1 < t).sum())
        print(f"  top1 < {t:.2f}: {n:>6,}  ({n/len(df):6.1%})")

    con = df[df.top1 < 0.90]
    print(f"\n=== among the {len(con):,} contested images ===")
    bim = con[(con.split_ratio > 0.5) & (con.top2_mass > 0.90)]
    dif = con[con.top2_mass < 0.70]
    print(f"  BIMODAL (two strong competitors): {len(bim):>5,}"
          f"   mean human_entropy {bim.human_entropy.mean():.3f}")
    print(f"  DIFFUSE (mass over many classes): {len(dif):>5,}"
          f"   mean human_entropy {dif.human_entropy.mean():.3f}")
    print(f"  in between:                       {len(con)-len(bim)-len(dif):>5,}")
    print("\n  Hard bins are for reporting only -- the analysis uses top2_mass and")
    print("  split_ratio as continuous outcomes, so all "
          f"{len(con):,} contested images count.")

    print("\n=== most contested class pairs (bimodal) ===")
    pairs = (bim.assign(pair=[tuple(sorted((a_, b_))) for a_, b_ in
                              zip(bim.human_label_name, bim.runner_up_name)])
             .pair.value_counts().head(8))
    for (x, y), n in pairs.items():
        print(f"  {x:11s} vs {y:11s} {n}")
    print("\n  These are semantic confusions, not image degradation -- the")
    print("  dual-match population PIE appears to lack.")
    print(f"\nwrote {a.out/'human_labels.parquet'}")


if __name__ == "__main__":
    main()
