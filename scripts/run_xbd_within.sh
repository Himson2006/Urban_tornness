#!/usr/bin/env bash
# Wave 2: train and evaluate inside a single event.
#
#   ./scripts/run_xbd_within.sh 0 1 2 3 4
#   ./scripts/run_xbd_within.sh 0 1 2 3 4 --analyse
#
# Wave 1 showed a model that beat the pooled majority (0.696 vs 0.542) while
# losing to every within-event majority, and collapsed to 0.362 against a 0.743
# baseline when a whole event was held out. It had learned which disaster it was
# looking at, not how damaged the building was. Pooling across events is what
# made that look like skill.
#
# So: one event at a time, no pooling available to hide behind.
#
#   michael  6,765 contested, 26% major -- the change signal separates cleanly
#            here (24.2/26.3/28.1/35.1 mean |pre-post| by class)
#   harvey  11,660 contested, 74% major -- flat signal (20.8/20.1/20.1/20.8).
#            A flood damages the inside, not the roof. Expect this to fail, and
#            that failure is the finding: ambiguity no imagery can resolve.
#
# Both also run post-only, because the localisation gap between paired and
# post-only (45% vs 19% on-building) was wave 1's one real positive and it needs
# to survive a setting where accuracy is measured honestly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ANALYSE=0
GPUS=()
for arg in "$@"; do
  case "$arg" in
    --analyse|--analyze) ANALYSE=1 ;;
    *) GPUS+=("$arg") ;;
  esac
done
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0)

EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-64}"        # 224x224 inputs now, not 96x96
mkdir -p xbd/runs logs

python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "xbd")
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA visible")
from dataset import load_meta
m = load_meta(Path("xbd/data/crops"), "middle", 24.0,
              Path("xbd/data/buildings.parquet"),
              Path("xbd/data/scene_radiometry.parquet"))
print(f"{torch.cuda.device_count()} GPU(s)")
print(f"{'event':22s} {'contested':>10} {'scenes':>8} {'major':>7}")
for d, g in m.groupby("disaster"):
    if len(g) < 2000:
        continue
    print(f"{str(d):22s} {len(g):10,} {g.scene.nunique():8,} {g.label.mean():7.3f}")
PY

CONFIGS=(
  "michael_pair : --disaster hurricane-michael"
  "michael_post : --disaster hurricane-michael --no-paired"
  "harvey_pair  : --disaster hurricane-harvey"
  "harvey_post  : --disaster hurricane-harvey --no-paired"
)

launch() {
  local gpu="$1" name="$2"; shift 2
  echo "  GPU $gpu  <- $name  $*"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u xbd/train.py \
    --epochs "$EPOCHS" --workers "$WORKERS" --batch "$BATCH" --fold 0 "$@" \
    > "logs/xbdw_${name}.out" 2>&1 &
}

echo "=== within-event runs ==="
i=0
for cfg in "${CONFIGS[@]}"; do
  name="$(echo "${cfg%%:*}" | tr -d ' ')"
  launch "${GPUS[$((i % ${#GPUS[@]}))]}" "$name" ${cfg#*:}
  i=$((i + 1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "done"

if [ "$ANALYSE" -eq 1 ]; then
  for stage in prototype_localization tornness; do
    echo; echo "=== ${stage} ==="
    for r in xbd/runs/*/; do
      [ -f "${r}DONE" ] || continue
      echo "--- $(basename "$r") ---"
      python "xbd/${stage}.py" --run "$r" 2>&1 \
        | tee "logs/${stage}_$(basename "$r").out" \
        || echo "  FAILED on $(basename "$r")"
    done
  done
fi

echo
echo "collect everything with:  ./scripts/xbd_report.sh > xbd_report.txt"
