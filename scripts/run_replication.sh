#!/usr/bin/env bash
# Replication harness for the urban claim: is the confidence inversion a
# property of single-frame intent models, or just of one ProtoPNet checkpoint?
#
#   ./scripts/run_replication.sh 0 1 2 3 4 5 6 7
#
# Trains plain CNNs (no prototype layer) across architectures on the same folds,
# then runs the stance-inversion analysis over everything found on disk.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPUS=("$@"); [[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0)
ARCHS="${ARCHS:-resnet18 resnet34 resnet50}"
FOLDS="${FOLDS:-0 1 2 3 4}"
mkdir -p runs_plain results
python -c "import pyarrow" || { echo "pyarrow missing -- wrong conda env?"; exit 1; }

slot_pid=(); for i in "${!GPUS[@]}"; do slot_pid[$i]=""; done
for arch in $ARCHS; do
  for k in $FOLDS; do
    [[ -f "runs_plain/${arch}_fold${k}/predictions.parquet" ]] && \
      { echo "  ${arch} fold${k} done, skipping"; continue; }
    slot=-1
    while [[ $slot -lt 0 ]]; do
      for i in "${!GPUS[@]}"; do
        pid="${slot_pid[$i]}"
        if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then slot=$i; break; fi
      done
      [[ $slot -lt 0 ]] && sleep 15
    done
    g="${GPUS[$slot]}"
    echo "  ${arch} fold${k} -> GPU $g"
    python -u src/train_plain.py --arch "$arch" --split kfold --fold "$k" \
           --gpu "$g" >> "runs_plain/${arch}_fold${k}.out" 2>&1 &
    slot_pid[$slot]=$!
    sleep 3
  done
done
wait

echo "=== stance inversion across all models ==="
python src/stance_inversion.py 2>&1 | tee results/stance_inversion.out
