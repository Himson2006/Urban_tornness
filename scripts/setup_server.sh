#!/usr/bin/env bash
# One-shot server setup: annotations -> manifest -> folds. No video, no GPU.
# Run from the repo root after activating your python env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== python / cuda ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

if [[ ! -f data/pie_manifest/peds.parquet ]]; then
    echo "=== downloading PIE annotations (~40 MB) ==="
    mkdir -p data/pie_raw && cd data/pie_raw
    for f in annotations.zip annotations_attributes.zip annotations_vehicle.zip README.md; do
        [[ -f "$f" ]] || curl -sLO "https://raw.githubusercontent.com/aras62/PIE/master/annotations/$f"
    done
    unzip -q -o 'annotations*.zip'
    cd "$ROOT"

    echo "=== building manifest ==="
    python src/pie_manifest.py
else
    echo "=== manifest already present, skipping ==="
fi

if [[ ! -f data/pie_manifest/folds_k5_video_seed0.parquet ]]; then
    echo "=== assigning video-grouped folds (seed 0) ==="
    python -c "
import sys, pandas as pd; sys.path.insert(0,'src')
from pie_dataset import kfold_assign
p = pd.read_parquet('data/pie_manifest/peds.parquet')
kfold_assign(p, n_folds=5, group='video', seed=0).to_parquet(
    'data/pie_manifest/folds_k5_video_seed0.parquet', index=False)
print('folds written')"
fi

chmod +x scripts/*.sh
echo
echo "SETUP DONE. Next:  ./scripts/run_set.sh set05 --delete"
