#!/usr/bin/env bash
# Collect everything needed to judge the xBD runs into one pasteable report.
#
#   ./scripts/xbd_report.sh > xbd_report.txt
#
# Order matters and is enforced here: localisation first, because if the shown
# prototypes are not on the building then the tornness numbers describe whatever
# they landed on instead, and reading them first invites believing them.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "################ 1. accuracy, pooled and per disaster ################"
python - <<'PY'
import json
from pathlib import Path
runs = sorted(Path("xbd/runs").glob("*/final_test.json"))
if not runs:
    raise SystemExit("no finished runs")
print(f"{'run':40s} {'test':>7} {'major':>7} {'val':>7} {'n_test':>8}")
for p in runs:
    d = json.loads(p.read_text())
    print(f"{d['tag']:40s} {d['test_acc']:7.4f} {d['majority']:7.4f} "
          f"{d['val_acc']:7.4f} {d['n_test']:8,}")
print()
for p in runs:
    d = json.loads(p.read_text())
    bd = d.get("by_disaster", {})
    if not bd:
        continue
    print(f"--- {d['tag']} ---")
    print(f"  {'disaster':24s} {'n':>7} {'acc':>7} {'majority':>9} {'lift':>7}")
    for k, v in sorted(bd.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {k[:24]:24s} {v['n']:7,} {v['acc']:7.4f} "
              f"{v['majority']:9.4f} {v['acc']-v['majority']:+7.4f}")
    print()
PY

echo
echo "################ 2. localisation (gates everything below) ############"
for r in xbd/runs/*/; do
  [ -f "${r}DONE" ] || continue
  echo "=== $(basename "$r") ==="
  if [ -f "logs/prototype_localization_$(basename "$r").out" ]; then
    cat "logs/prototype_localization_$(basename "$r").out"
  else
    python xbd/prototype_localization.py --run "$r" 2>&1 \
      || echo "  FAILED -- see above"
  fi
  echo
done

echo
echo "################ 3. tornness ########################################"
for r in xbd/runs/*/; do
  [ -f "${r}DONE" ] || continue
  echo "=== $(basename "$r") ==="
  if [ -f "logs/tornness_$(basename "$r").out" ]; then
    cat "logs/tornness_$(basename "$r").out"
  else
    python xbd/tornness.py --run "$r" 2>&1 || echo "  FAILED -- see above"
  fi
  echo
done

echo
echo "################ 4. training sanity ##################################"
for r in xbd/runs/*/; do
  [ -f "${r}metrics.csv" ] || continue
  echo "=== $(basename "$r") ==="
  python - "$r" <<'PY'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1] + "metrics.csv")
print(f"  {len(d)} epochs | best val {d.val_acc.max():.4f} at epoch "
      f"{int(d.val_acc.idxmax())} | last val {d.val_acc.iloc[-1]:.4f}")
print(f"  lr {d.lr.iloc[0]:.2e} -> {d.lr.iloc[-1]:.2e}")
# a val curve that never moves means the model learned nothing; one that peaks
# at epoch 0 means it learned nothing after the warm phase
print(f"  val range [{d.val_acc.min():.4f}, {d.val_acc.max():.4f}] "
      f"(flat curve = nothing learned)")
PY
done
