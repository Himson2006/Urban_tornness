# Server runbook — PIE tornness experiments

## The short version

```bash
tmux new -s pie
conda create -n pie python=3.10 -y && conda activate pie
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyarrow opencv-python-headless pillow scikit-learn scipy matplotlib statsmodels

./scripts/setup_server.sh                       # annotations + manifest + folds (~2 min)

./scripts/run_set.sh set05 --delete             # smoke test: 2 clips. LOOK at the contact sheet.
for S in set01 set02 set06 set04 set03; do      # ~3-5 h total, CPU only
  ./scripts/run_set.sh $S --delete
done
python src/pie_dataset.py                       # sanity summary

./scripts/train_all.sh 0                        # single GPU (~12-24 h); or: ./scripts/train_all.sh 0 1 2 3
```

The rest of this file explains each step, what to check, and what can go wrong.

Everything below is written for a CUDA box. Steps 1–6 are ready to run; step 7
(the four experiments) still needs analysis code — see the bottom.

**Total wall time:** ~3–5 h extraction (CPU-bound, no GPU), ~2–4 h per training
fold, minutes for tornness extraction.
**Peak disk:** one video set (~23 GB worst case) + 3.1 GB crops + checkpoints.
Budget **40 GB free** and you never come close.

---

## 1. Prerequisites

```bash
nvidia-smi                       # confirm GPU + driver
df -h .                          # need >=40 GB free on the working volume
python3 -V                       # 3.9+
ffmpeg -version                  # not required, but useful for debugging clips
tmux new -s pie                  # do everything inside tmux; downloads are long
```

## 2. Get the code onto the server

```bash
git clone <your-remote>/urban-project.git && cd urban-project
```

If you have no remote yet, from the laptop:

```bash
rsync -avz --exclude data/pie_crops --exclude runs \
      "/Users/himsonchapagain/Documents/RAI Lab/Urban Project/" \
      user@server:~/urban-project/
```

Do **not** rsync `data/pie_crops` — it is faster to regenerate on the server than
to ship. `data/pie_manifest/` is small (a few MB) and worth copying so the fold
assignment is byte-identical to the laptop's; otherwise regenerate it in step 4.

## 3. Environment

```bash
conda create -n pie python=3.10 -y && conda activate pie
# match the CUDA build to the server's driver (cu121 shown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pyarrow opencv-python-headless pillow scikit-learn \
            scipy matplotlib statsmodels
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use `opencv-python-headless` on a server — plain `opencv-python` pulls GUI libs
that are usually missing and fail at import.

## 4. Rebuild the manifest (no video needed, ~1 min)

Skip if you rsynced `data/pie_manifest/`.

```bash
cd data/pie_raw
for f in annotations.zip annotations_attributes.zip annotations_vehicle.zip README.md; do
  curl -sLO "https://raw.githubusercontent.com/aras62/PIE/master/annotations/$f"
done
unzip -q -o 'annotations*.zip'
cd ../.. && python src/pie_manifest.py
```

Expect: 1,842 pedestrian tracks, 740,901 boxes, and the `intention_prob`
histogram. Then fix the folds (deterministic, seed 0):

```bash
python -c "
import sys, pandas as pd; sys.path.insert(0,'src')
from pie_dataset import kfold_assign
p = pd.read_parquet('data/pie_manifest/peds.parquet')
kfold_assign(p, n_folds=5, group='video', seed=0).to_parquet(
    'data/pie_manifest/folds_k5_video_seed0.parquet', index=False)
print('folds written')"
```

## 5. Rolling download + extraction

This is the storage-disciplined loop: **one set on disk at a time.** All 53 clips
are needed (~64 GB total download, ~1.2 GB per 10-min clip); the per-set URL list
is in `data/pie_manifest/videos_needed.txt`.

Order deliberately starts with **set05** — 2 clips, 0.02 GB of crops. It is a live
smoke test on real PIE footage that finishes in minutes, so decoder problems
surface before you spend 23 GB on set03.

| set | clips | download | crops | split |
|---|---|---|---|---|
| set05 | 2 | ~2.4 GB | 0.02 GB | val |
| set01 | 4 | ~4.8 GB | 0.21 GB | train |
| set02 | 3 | ~3.6 GB | 0.16 GB | train |
| set06 | 9 | ~10.8 GB | 0.28 GB | val |
| set04 | 16 | ~19.2 GB | 1.27 GB | train |
| set03 | 19 | ~22.8 GB | 1.15 GB | test |

For each set, in order:

```bash
SET=set05
mkdir -p PIE_clips/$SET
grep "/$SET/" data/pie_manifest/videos_needed.txt \
  | xargs -n1 -P4 wget -q --show-progress -c -P PIE_clips/$SET

python src/pie_extract.py --set $SET --clips PIE_clips
```

The script decodes each clip once (sequential `grab`, never seeks), checkpoints
metadata after every clip so an interrupt costs at most one video, verifies crop
counts and pedestrian coverage, writes a 20-crop contact sheet, and prints either

```
SAFE TO DELETE VIDEOS FOR set05
```

or a failure. **It never deletes anything.** On success:

```bash
open data/pie_crops/_spotcheck_$SET.jpg   # or scp it down — LOOK at it once
rm -rf PIE_clips/$SET
```

Look at the contact sheet at least for set05. Crops are labelled with
`action/look` and `t±frames_to_critical`; if boxes are off the pedestrian or the
labels look shuffled, stop — that is the failure mode that would silently poison
every downstream result.

Re-running `pie_extract.py` for a set is safe and resumes from `_state_<set>.json`.
`--verify-only` re-runs the checks without touching video.

Once all six are done:

```bash
python src/pie_dataset.py     # sanity summary across the whole crop store
```

Expect ~225k crops, 1,842 pedestrians, intent balance ~1374/468, and 140
contested pedestrians at 0.4–0.6.

## 6. Train

Headline model on PIE's official split (comparable to prior PIE work):

```bash
python src/train_pie.py --split official --arch resnet34 --epochs 40 --gpu 0
```

Then the five grouped folds that the contested-case analyses run on:

```bash
for k in 0 1 2 3 4; do
  python src/train_pie.py --split kfold --fold $k --arch resnet34 --epochs 40 --gpu 0
done
```

With multiple GPUs, run folds concurrently — one fold per GPU, `--gpu 1`, `--gpu 2`, …

Outputs land in `runs/resnet34_fold<k>/`: `train.log`, `best.pth`, `final.pth`,
and `img/` containing every projected prototype as a real pedestrian patch with
its bounding box. Those images *are* the paper's exemplars.

Notes on the defaults:

- `--protos-per-class 20` (40 total). CUB uses 2000 for 200 classes; here the
  point is that co-activation across two readable prototype sets is legible.
- `--frame-stride 5` — adjacent frames of one pedestrian are near-duplicates, so
  training on all of them inflates epochs without adding information. Tornness
  extraction in step 7 still scores **every** frame.
- Class imbalance (75/25) is handled by a `WeightedRandomSampler`, not by editing
  ProtoPNet's loss. The upstream source is untouched.
- Most of the runtime is the upstream recipe's 20 last-layer epochs after each
  push. `--push-every 20` roughly halves total time if you are iterating.

Sanity check before trusting anything: test accuracy should clear the majority-
class baseline of **0.744**. A model sitting at 0.744 has collapsed to "everyone
intends to cross" and its prototypes are meaningless.

## 7. Extract tornness features

```bash
for k in 0 1 2 3 4; do
  python src/tornness.py --ckpt runs/resnet34_fold0/best.pth --fold $k --gpu 0
done
```

Each run scores only its held-out fold and writes
`runs/.../tornness_fold<k>.parquet` — one row per (pedestrian, frame) with
`coact_min`, `coact_prod`, `global_max_sim`, `margin`, `entropy`,
`mc_dropout_std`, `typing_score`, plus every degradation covariate — and a
`.protosims.npy` of the full per-prototype similarity vectors (the fallback
feature if 2-class co-activation proves too coarse).

It prints the Experiment 1 headline immediately: Spearman correlations of each
uncertainty measure against `human_disagreement`, on torn cases only. Concatenate
the five folds for the full 300-contested-pedestrian analysis.

---

## What is not written yet

Steps 1–7 produce the measurements. The four experiments still need analysis code:

1. **Signature → disagreement.** Partly there — `tornness.py` prints the raw
   Spearmans. Still needed: the confound controls (regress out `bbox_h`,
   `blur_var_laplacian`, `occluded_flag`, `truncated`), bootstrap CIs, and the
   comparison table against entropy / margin / MC-dropout.
2. **Signature → resolvability.** Group `tornness_fold*.parquet` by `ped_id`,
   align on `frames_to_critical`, and plot tornness decay stratified by
   `typing_score`. Anchor on `intent_binary`, **not** on whether they physically
   crossed — see RESULTS.md.
3. **Signature → human reasons.** PIE has no free text. The substitute is the
   per-frame `action`/`look`/`cross`/`gesture` tags already in every row.
4. **Typed deferral.** Needs `intention_prob` as the simulated human and matched
   deferral budgets.

Also unbuilt: the CIFAR-10H arm, which is the only place the bimodal-vs-diffuse
*shape* claim can be tested, since PIE releases only aggregate `intention_prob`.
Remember to upsample CIFAR to 224 — at 32x32 a prototype's receptive field covers
most of the image.

## Verified vs. unverified

Verified on the laptop, no GPU: manifest build against all 53 real annotation
files; extraction end-to-end on a synthetic clip (62/62 crops, checks pass);
frame-index alignment (solid-grey frames, luma error <=2, so no off-by-one);
ProtoPNet constructed with `num_classes=2`, 40 prototypes, 20 per class, correct
similarity and MC-dropout tensor shapes on CPU.

Not yet verified: decoding real PIE MP4s (B-frames or variable frame rate could
behave differently from the synthetic test — this is exactly why set05 goes
first, and why you should look at the contact sheet), and any CUDA path.
