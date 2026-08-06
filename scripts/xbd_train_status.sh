#!/usr/bin/env bash
# Has the xBD training finished, and is it safe to run the analysis?
#
#   ./scripts/xbd_train_status.sh        # one look
#   ./scripts/xbd_train_status.sh -w     # refresh every 60s
#
# "Finished" means a DONE marker exists for a run. That marker is written only
# after the test set has been scored and final_test.json saved, so it is the
# same condition the analysis scripts need -- a run without it has no best.pth
# worth reading.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

report() {
  echo "=== xBD training  $(date '+%H:%M:%S') ==="

  local procs
  procs="$(pgrep -fc "xbd/train.py" 2>/dev/null)"; procs="${procs:-0}"
  echo "training procs   ${procs}"

  if [ ! -d xbd/runs ] || [ -z "$(ls -A xbd/runs 2>/dev/null)" ]; then
    echo "runs             none yet"
    return
  fi

  local total=0 done=0
  printf "\n%-40s %-9s %s\n" "run" "state" "progress"
  for r in xbd/runs/*/; do
    [ -d "$r" ] || continue
    total=$((total + 1))
    local name state prog
    name="$(basename "$r")"
    if [ -f "${r}DONE" ]; then
      state="DONE"; done=$((done + 1))
      prog="$(python -c "
import json,sys
try:
    d=json.load(open('${r}final_test.json'))
    print(f\"test {d['test_acc']:.4f}  majority {d['majority']:.4f}\")
except Exception: print('')" 2>/dev/null)"
    elif [ -f "${r}metrics.csv" ]; then
      state="running"
      prog="epoch $(($(wc -l < "${r}metrics.csv") - 1))"
    else
      state="starting"; prog=""
    fi
    printf "%-40s %-9s %s\n" "$name" "$state" "$prog"
  done

  echo
  if [ "$procs" -eq 0 ] && [ "$done" -eq "$total" ] && [ "$total" -gt 0 ]; then
    echo "ALL ${done}/${total} FINISHED -- safe to run the analysis:"
    echo "  for r in xbd/runs/*/; do python xbd/prototype_localization.py --run \"\$r\"; done"
    echo "  for r in xbd/runs/*/; do python xbd/tornness.py --run \"\$r\"; done"
  elif [ "$procs" -eq 0 ]; then
    echo "${done}/${total} finished but nothing is running -- some runs died."
    echo "Check logs/xbd_*.out, then re-run ./scripts/run_xbd.sh (finished"
    echo "runs are skipped via their DONE marker, so this costs nothing)."
  else
    echo "${done}/${total} finished, ${procs} still training. Wait."
  fi
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; report; sleep 60; done
else
  report
fi
