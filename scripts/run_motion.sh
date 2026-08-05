#!/usr/bin/env bash
# Motion-input experiment: does giving the model temporal context fix the
# stance blindness and the confidence inversion?
#
#   ./scripts/run_motion.sh 0 1 2 3 4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0 1 2 3 4)
NF="${NF:-3}"; GAP="${GAP:-5}"
mkdir -p runs_motion results
python -c "import pyarrow" || { echo "wrong conda env?"; exit 1; }

for k in 0 1 2 3 4; do
  g="${GPUS[$(( k % ${#GPUS[@]} ))]}"
  python -u src/train_pie_motion.py --fold $k --n-frames "$NF" --gap "$GAP" \
         --tf32 --gpu "$g" >> "runs_motion/fold${k}.out" 2>&1 &
  sleep 3
done
wait

for k in 0 1 2 3 4; do
  g="${GPUS[$(( k % ${#GPUS[@]} ))]}"
  python src/tornness.py --ckpt "runs_motion/resnet34_f${NF}g${GAP}_fold${k}/best.pth" \
         --split kfold --fold $k --n-frames "$NF" --gap "$GAP" --gpu "$g" \
         > "runs_motion/torn${k}.out" 2>&1 &
done
wait

echo "=== motion vs single-frame ==="
python src/stance_inversion.py \
  --proto-glob "runs_motion/resnet34_f${NF}g${GAP}_fold*/tornness_fold*.parquet" \
  2>&1 | tee results/stance_motion.out
