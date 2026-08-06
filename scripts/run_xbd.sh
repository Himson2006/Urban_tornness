#!/usr/bin/env bash
# xBD: five configurations that between them decide whether the study holds.
#
#   ./scripts/run_xbd.sh                 # sequential, one GPU
#   ./scripts/run_xbd.sh 0 1 2 3 4       # five jobs, one GPU each, one wave
#   ./scripts/run_xbd.sh 0 1 2 3 4 --folds-too   # then folds 1-4 of the primary
#
# The five are not variations on a theme. Four of them exist to kill the study:
#
#   primary   minor vs major, paired, scene-grouped    -- the claim
#   postonly  same, post-disaster image alone          -- is the pair load-bearing?
#   raw       same as primary, no radiometric alignment -- was it the satellite pass?
#   extremes  no-damage vs destroyed, paired           -- does tornness track ambiguity?
#   disaster  primary, whole events held out           -- does it survive a new event?
#
# If `postonly` matches `primary`, the pairing bought nothing. If `raw` beats
# `primary`, the model was reading capture conditions. If `extremes` shows the
# same co-activation as `primary`, the measure is not about ambiguity.
#
# RESUMABLE: finished runs are skipped via their DONE marker. Re-run the same
# command after any interruption.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOLDS_TOO=0
GPUS=()
for arg in "$@"; do
  case "$arg" in
    --folds-too) FOLDS_TOO=1 ;;
    *) GPUS+=("$arg") ;;
  esac
done
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0)

EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-128}"
mkdir -p xbd/runs logs

# Fail early and legibly rather than 20 minutes into the first run.
python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "xbd")
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA visible -- ProtoPNet's train loop requires a GPU")
for p in ["xbd/data/crops/crop_meta.parquet", "xbd/data/buildings.parquet",
          "xbd/data/scene_radiometry.parquet"]:
    if not Path(p).exists():
        raise SystemExit(f"missing {p}")
from dataset import load_meta
m = load_meta(Path("xbd/data/crops"), "middle", 24.0,
              Path("xbd/data/buildings.parquet"),
              Path("xbd/data/scene_radiometry.parquet"))
cov = m.aligned.mean()
print(f"{len(m):,} contested crops, {m.scene.nunique():,} scenes, "
      f"{m.disaster.nunique()} disasters, alignment {cov:.1%}, "
      f"{torch.cuda.device_count()} GPU(s)")
if cov < 0.99:
    raise SystemExit("run: python xbd/radiometry.py")
PY

# name : extra flags
CONFIGS=(
  "primary  :"
  "postonly : --no-paired"
  "raw      : --no-align"
  "extremes : --task extremes"
  "disaster : --group disaster"
)

launch() {   # launch <gpu> <name> <flags...>
  local gpu="$1" name="$2"; shift 2
  echo "  GPU $gpu  <- $name  $*"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u xbd/train.py \
    --epochs "$EPOCHS" --workers "$WORKERS" --batch "$BATCH" "$@" \
    > "logs/xbd_${name}.out" 2>&1 &
}

echo "=== wave 1: the five configurations ==="
i=0
for cfg in "${CONFIGS[@]}"; do
  name="$(echo "${cfg%%:*}" | tr -d ' ')"
  flags="${cfg#*:}"
  launch "${GPUS[$((i % ${#GPUS[@]}))]}" "$name" --fold 0 ${flags}
  i=$((i + 1))
  # with fewer GPUs than jobs, wait for the wave to drain before oversubscribing
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "wave 1 done"

if [ "$FOLDS_TOO" -eq 1 ]; then
  echo "=== wave 2: folds 1-4 of the primary ==="
  i=0
  for k in 1 2 3 4; do
    launch "${GPUS[$((i % ${#GPUS[@]}))]}" "primary_f${k}" --fold "$k"
    i=$((i + 1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
  done
  wait
  echo "wave 2 done"
fi

echo
echo "=== results ==="
python - <<'PY'
import json
from pathlib import Path
rows = sorted(Path("xbd/runs").glob("*/final_test.json"))
if not rows:
    raise SystemExit("no finished runs")
print(f"{'run':38s} {'test':>7} {'majority':>9} {'val':>7} {'n_test':>8}")
for p in rows:
    d = json.loads(p.read_text())
    print(f"{d['tag']:38s} {d['test_acc']:7.4f} {d['majority']:9.4f} "
          f"{d['val_acc']:7.4f} {d['n_test']:8,}")
PY

echo
echo "next: tornness, on every finished run"
echo "  for r in xbd/runs/*/; do python xbd/tornness.py --run \"\$r\"; done"
