"""Figures for the UrbanAI'26 paper.

Three figures, each doing a job the tables cannot:

  fig1  what the filter does across the whole disagreement plane, rather than
        at the two thresholds the tables report
  fig2  per-region retention within each city, so the spatial spread is visible
        as a distribution rather than a min-max range
  fig3  where predicted probabilities pile up on contested labels, which is the
        harm and the remedy in one panel

Palette is the validated four-slot categorical set (blue/orange/aqua/yellow);
sequential magnitude uses a single blue ramp. Output is PDF at single-column
width for acmart sigconf.

Usage:
    python sidewalk/figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
COL_W = 3.33          # acmart sigconf single column, inches


def style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def fig1(d: pd.DataFrame, out: Path):
    """Retention against each axis of disagreement, holding the other low.

    A 2-D plane was the first attempt and it was the wrong form: the cells are
    sparse and the interaction between the two axes obscured the finding. The
    marginals show it plainly -- the filter is not a gradient but a cliff.
    """
    import matplotlib.pyplot as plt

    def marg(sub, col, edges):
        out_ = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = sub[(sub[col] >= lo) & (sub[col] < hi)]
            if len(g) >= 200:
                out_.append((f"{lo:.2f}" if hi - lo > .01 else "0",
                             g.qc_kept.mean(), len(g)))
        return out_

    a = marg(d[d.unsure < 0.2], "split", [0, .001, .2, .4, .6, .8, 1.01])
    b = marg(d[d.split < 0.2], "unsure", [0, .001, .15, .3, .45, 1.01])
    overall = d.qc_kept.mean()

    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.06, 1.95), sharey=True)
    for ax, dat, lab, ticks in (
            (axes[0], a, "how evenly reviewers split",
             ["none", "0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"]),
            (axes[1], b, "share answering “not sure”",
             ["none", "0–.15", ".15–.3", ".3–.45", ".45+"])):
        x = np.arange(len(dat))
        ax.bar(x, [v for _, v, _ in dat], width=0.66, color=BLUE, linewidth=0)
        ax.axhline(overall, color=ORANGE, lw=1.2, ls=(0, (4, 2)), zorder=3)
        for xi, (_, v, _n) in zip(x, dat):
            ax.text(xi, v + 0.03, f"{v*100:.0f}%", ha="center", fontsize=6.5,
                    color=INK)
        ax.set_xticks(x)
        # counts live in the tick label; on a 3%-tall bar they collide with the
        # percentage if placed inside the mark
        ax.set_xticklabels([f"{t}\n{n/1000:.0f}k" for t, (_, _v, n)
                            in zip(ticks[:len(dat)], dat)], fontsize=6.5)
        ax.set_xlabel(lab)
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("labels retained")
    axes[0].text(len(a) - 0.5, overall + 0.035, "all labels", ha="right",
                 fontsize=6.5, color=ORANGE)
    fig.suptitle("The filter is a cliff, not a gradient: retention collapses "
                 "once either kind of disagreement passes a threshold",
                 fontsize=8, y=1.04)
    fig.savefig(out / "fig1_cliff.pdf"); fig.savefig(out / "fig1_cliff.png", dpi=220)
    plt.close(fig)


def fig2(sp: pd.DataFrame, out: Path):
    """Per-region retention within each city. Identity is the city; one hue."""
    import matplotlib.pyplot as plt
    sp = sp[sp.n >= 100].copy()
    order = sp.groupby("city").actual.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        v = sp[sp.city == c].actual.to_numpy()
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)),
                   s=5, color=BLUE, alpha=0.45, linewidths=0)
        ax.plot([np.median(v)], [i], marker="|", ms=11, mew=1.4, color=ORANGE)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of a neighbourhood's labels retained")
    ax.grid(axis="x", color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=5, color=BLUE, label="neighbourhood")
    ax.plot([], [], marker="|", ls="none", ms=9, mew=1.4, color=ORANGE,
            label="city median")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              frameon=False, handletextpad=0.4, columnspacing=1.2)
    ax.set_title("Retention varies widely within every city", pad=16, loc="right")
    fig.savefig(out / "fig2_spatial.pdf"); fig.savefig(out / "fig2_spatial.png", dpi=220)
    plt.close(fig)


def fig3(cal: pd.DataFrame, out: Path):
    """Predicted probability on contested labels: the harm and the remedy."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    bins = np.linspace(0, 1, 26)
    for name, colr, lab in (("filtered", ORANGE, "trained on filtered record"),
                            ("soft_clf", BLUE, "trained on review distribution")):
        v = cal[name].dropna().to_numpy()
        ax.hist(v, bins=bins, histtype="step", lw=1.6, color=colr, label=lab,
                density=True)
    ax.axvspan(0.9, 1.0, color="#f2f1ec", zorder=0)
    # the shaded band is explained in the caption; an in-axes annotation
    # collided with either the legend or the x label at this figure width
    ax.set_xlabel("predicted probability the label is correct")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.5)
    ax.set_title("On labels reviewers split on", pad=4)
    fig.savefig(out / "fig3_calibration.pdf"); fig.savefig(out / "fig3_calibration.png", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "sidewalk/data")
    ap.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()

    d = pd.read_parquet(a.data / "labels.parquet")
    fig1(d, a.out)
    print("  fig1_cliff.pdf")

    sp = pd.read_csv(a.data / "spatial_retention.csv")
    fig2(sp, a.out)
    print("  fig2_spatial.pdf")

    cal = a.data / "contested_preds.parquet"
    if cal.exists():
        fig3(pd.read_parquet(cal), a.out)
        print("  fig3_calibration.pdf")
    else:
        print("  fig3 skipped: run calibration.py with --save-preds first")
    print(f"\nwrote to {a.out}")


if __name__ == "__main__":
    main()
