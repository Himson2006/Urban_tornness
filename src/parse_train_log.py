"""Reconstruct metrics.csv from an existing ProtoPNet train.log.

Runs launched before per-epoch CSV logging existed still have everything we need
in their text log -- this recovers it without retraining. Emits the same schema
train_pie.py now writes live, so downstream plotting code does not care which
run it is reading.

Note on `test_acc`: ProtoPNet's `tnt.test()` logs the marker "\ttest" whatever
loader it is given. In runs from before val-based selection existed that column
really is test accuracy; in newer runs the in-loop evaluations are on VAL, and
the single honest test number lives in the run's final_test.txt.

Usage:
    python src/parse_train_log.py runs/resnet34_official/train.log
    python src/parse_train_log.py runs/*/train.log        # all runs at once
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

EPOCH = re.compile(r"^epoch\s+(\d+)")
ACCU = re.compile(r"^\taccu:\s*([\d.]+)%")
# section markers emitted by train_and_test.py / push.py
MARKS = {"\ttrain": "train", "\ttest": "test", "\twarm": "warm",
         "\tjoint": "joint", "\tlast layer": "last", "\tpush": "push"}


def parse(log_path: Path) -> pd.DataFrame:
    epoch, mode, section = -1, "warm", None
    pushed, last_i, pending_train = 0, 0, None
    rows = []

    for line in log_path.read_text(errors="replace").splitlines():
        m = EPOCH.match(line)
        if m:
            epoch = int(m.group(1))
            pushed, last_i = 0, 0
            continue

        stripped = line.rstrip()
        if stripped in MARKS:
            mark = MARKS[stripped]
            if mark in ("warm", "joint", "last"):
                mode = mark              # optimiser phase
            elif mark == "push":
                pushed = 1               # everything after this is post-projection
            section = mark if mark in ("train", "test") else section
            continue

        a = ACCU.match(line)
        if not a or epoch < 0:
            continue
        acc = float(a.group(1)) / 100.0

        if section == "train":
            pending_train = acc
        elif section == "test":
            # a test accu closes one (train, test) pair
            phase = mode if mode != "last" else f"last_{last_i}"
            if mode == "last":
                last_i += 1
            rows.append({"epoch": epoch, "phase": phase,
                         "train_acc": pending_train, "test_acc": acc,
                         "pushed": pushed})
            pending_train = None

    df = pd.DataFrame(rows)
    if not df.empty:
        df["best"] = df.test_acc.cummax()
    return df


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    for p in paths:
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        df = parse(p)
        if df.empty:
            print(f"{p}: no epochs parsed yet")
            continue
        dest = p.parent / "metrics.csv"
        df.to_csv(dest, index=False)
        post = df[df.pushed == 1]
        print(f"{p.parent.name}: {len(df)} rows, epochs 0-{df.epoch.max()}, "
              f"best test {df.test_acc.max():.4f}"
              + (f", best post-push {post.test_acc.max():.4f}" if len(post) else "")
              + f"  -> {dest}")


if __name__ == "__main__":
    main()
