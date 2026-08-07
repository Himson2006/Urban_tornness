#!/usr/bin/env bash
# The paper's experimental backbone: every event, both inputs, all five folds.
#
#   ./scripts/run_xbd_events.sh 0 1 2 3 4 5 6 7
#   EVENTS="hurricane-michael" ./scripts/run_xbd_events.sh 0 1     # a subset
#
# Everything so far is fold 0 at one seed. The three claims the paper rests on
# each need error bars before anyone should believe them:
#
#   1. Reducibility is graded by failure mode. Harvey (flood) 0.511, Michael
#      (wind) 0.666, destruction 0.964. With one fold each, that gradient is
#      three points and no spread.
#   2. Channel-stacking the pair does not help. Post-only won 3/3, but each of
#      those was a single run against a single run.
#   3. Pooling manufactures skill. Pooled 0.740 against within-event 0.666 and
#      0.511 -- again, one fold.
#
# Five folds per cell turns each of those into a paired comparison over matched
# test sets, which is the weakest design a reviewer will accept.
#
# RESUMABLE: DONE markers mean a re-run costs nothing for finished cells.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS=("$@")
[ ${#GPUS[@]} -eq 0 ] && GPUS=(0)

EVENTS="${EVENTS:-hurricane-harvey hurricane-michael hurricane-matthew}"
TASKS="${TASKS:-middle extremes}"
FOLDS="${FOLDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-64}"
mkdir -p xbd/runs logs

python - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "xbd")
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA visible")
from dataset import load_meta
print(f"{torch.cuda.device_count()} GPU(s) visible")
for task in os.environ.get("TASKS", "middle extremes").split():
    m = load_meta(Path("xbd/data/crops"), task, 24.0,
                  Path("xbd/data/buildings.parquet"),
                  Path("xbd/data/scene_radiometry.parquet"))
    print(f"\n{task}:")
    for e in os.environ.get("EVENTS", "").split():
        g = m[m.disaster == e]
        if len(g) == 0:
            print(f"  {e:22s} ABSENT"); continue
        print(f"  {e:22s} {len(g):7,} crops  {g.scene.nunique():4,} scenes  "
              f"minority {min(g.label.mean(), 1-g.label.mean()):.3f}")
PY
export EVENTS TASKS

JOBS=()
for task in $TASKS; do
  for ev in $EVENTS; do
    for k in $FOLDS; do
      JOBS+=("$task|$ev|$k|pair|")
      # post-only only on `middle`: the pair-vs-post comparison is about the
      # contested boundary, and doubling the extremes arm buys nothing
      [ "$task" = "middle" ] && JOBS+=("$task|$ev|$k|post|--no-paired")
    done
  done
done
echo
echo "${#JOBS[@]} cells over ${#GPUS[@]} GPU(s)"

i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r task ev k inp flags <<< "$job"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  name="${task}_${inp}_${ev#hurricane-}_f${k}"
  echo "  GPU $gpu  <- $name"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u xbd/train.py \
    --task "$task" --disaster "$ev" --fold "$k" ${flags} \
    --epochs "$EPOCHS" --workers "$WORKERS" --batch "$BATCH" \
    > "logs/xbde_${name}.out" 2>&1 &
  i=$((i + 1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait

echo
echo "training done. Analysis over the new runs:"
for stage in prototype_localization tornness; do
  echo "=== ${stage} ==="
  for r in xbd/runs/*/; do
    [ -f "${r}DONE" ] || continue
    b="$(basename "$r")"
    [ -f "logs/${stage}_${b}.out" ] && continue     # already analysed
    python "xbd/${stage}.py" --run "$r" > "logs/${stage}_${b}.out" 2>&1 \
      && echo "  ok   $b" || echo "  FAIL $b"
  done
done

echo
echo "now:  python xbd/aggregate.py"
