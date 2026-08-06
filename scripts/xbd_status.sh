#!/usr/bin/env bash
# Is the xBD crop extraction working, and how far along is it?
#
#   ./scripts/xbd_status.sh          # one look
#   ./scripts/xbd_status.sh -w       # refresh every 60s
#
# "Working" means three things at once: the process is alive, the scene counter
# is advancing, and crops are landing on disk. A live process with a stalled
# counter is the interesting failure -- it means the Hub is throttling, which
# looks identical to progress from the outside.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="xbd/extract.log"

report() {
  echo "=== xBD extraction  $(date '+%H:%M:%S') ==="

  if pgrep -f "extract_crops" > /dev/null; then
    echo "process    RUNNING (pid $(pgrep -f extract_crops | tr '\n' ' '))"
  else
    echo "process    NOT RUNNING"
    if grep -q "paired crops from" "$LOG" 2>/dev/null; then
      echo "           finished cleanly:"
      grep "paired crops from" "$LOG" | tail -1 | sed 's/^/           /'
    else
      echo "           did NOT finish -- resume with:"
      echo "           python -u xbd/extract_crops.py --min-side 24 2>&1 | tee -a $LOG"
    fi
  fi

  local last done total pct
  last="$(grep -E '^  \(' "$LOG" 2>/dev/null | tail -1)"
  if [ -n "$last" ]; then
    done="$(echo "$last" | sed -E 's/^  \(([0-9]+)\/([0-9]+)\).*/\1/')"
    total="$(echo "$last" | sed -E 's/^  \(([0-9]+)\/([0-9]+)\).*/\2/')"
    pct=$(( 100 * done / total ))
    echo "scenes     ${done}/${total}  (${pct}%)"
  else
    echo "scenes     no progress line yet"
  fi

  echo "disk       $(du -sh xbd/data/crops 2>/dev/null | cut -f1)"

  # throttling looks exactly like slow progress; count it explicitly
  local rl
  # grep -c prints 0 and exits 1 when there are no matches; `|| echo 0` would
  # append a second line and break the arithmetic below
  rl="$(grep -c "Rate limited" "$LOG" 2>/dev/null)"
  rl="${rl:-0}"
  echo "throttled  ${rl} backoffs so far$([ "$rl" -gt 40 ] && echo '  <- set HF_TOKEN to speed this up')"

  python - <<'PY' 2>/dev/null
import pandas as pd
from pathlib import Path
p = Path("xbd/data/crops/crop_meta.parquet")
if not p.exists():
    raise SystemExit("crops      metadata not written yet (first save is at scene 50)")
m = pd.read_parquet(p)
mid = m.damage.isin(["minor-damage", "major-damage"]).sum()
print(f"crops      {len(m):,} paired  ({mid:,} contested)  "
      f"from {m.scene.nunique():,} scenes")
print("           " + "  ".join(
    f"{k.split('-')[0]} {v:,}" for k, v in m.damage.value_counts().items()))
PY
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; report; sleep 60; done
else
  report
fi
