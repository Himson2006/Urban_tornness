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

---

# xBD: competing prototypes over pre/post change

A separate study from the PIE work above, sharing its discipline but not its
data. The question is *when does competing-prototype explanation work, and where
is ambiguity concentrated in disaster damage assessment?*

## Why the input is a pair

Buildings in xBD have a median footprint of 26 pixels; only 15% reach 40. On
appearance alone that is thinner evidence than the pedestrian crops that already
failed. But the pre- and post-disaster images are co-registered, so the same
pixel polygon addresses the same building in both, and the model can be given
the pair. The explanation's object stops being *what does this roof look like*
and becomes *what changed here* — and change survives low resolution far better
than texture.

The stem conv is inflated from 3 to 6 channels with the pretrained kernel
halved, so a building that did not change reproduces the pretrained single-image
response exactly (verified to 7e-6). "No change" is the origin of the feature
space rather than an arbitrary point in it.

## The population (all 3,732 label files, verified)

| | |
|---|---|
| buildings | 217,649 across 10 disasters |
| ordinal middle (minor + major) | 37,789 (17.4%) |
| middle at >= 24 px | 22,771 — 10,433 minor / 12,338 major |
| scenes holding them | 1,283 (~4.2 GB of imagery) |
| control task (no-damage vs destroyed, >= 24 px) | 93,818 |

Ambiguity is not spread evenly across events, and the pattern is not noise:
hurricanes carry it (Matthew 61.6% middle, Harvey 46.0%, Michael 31.3%) while
fire, earthquake and tsunami barely do (Santa Rosa 1.0% middle but 25.8%
destroyed; Mexico earthquake 0.6%). Wind and water damage buildings by degrees;
fire and ground failure tend not to. That is a finding, not a nuisance.

**Two confounds to keep in view.** Major-damage footprints are larger than minor
(32 vs 25 px median) and destroyed are smallest (20 px) — rubble is traced
smaller. So building size carries label information directly, and every
association below is reported partialled on size, off-nadir, GSD and sun
elevation. Second, one event (Harvey) is half the contested set, which is why
`--group disaster` exists.

## The radiometric confound, found by looking

The first contact sheet turned up something no accuracy number would have shown.
The pre and post images of a scene come from different satellite passes, and the
whole frame shifts in tone between them: on the scenes measured so far the post
capture is 11 levels darker on a 0-255 scale and carries **35% less contrast**
(median ratio 0.65). On the major-damage examples the pre crop was blue-cast and
the post nearly white, across the entire crop including the grass.

Handed that, a six-channel model learns "the post image is brighter", and scores
well, because capture conditions correlate with scene, scene with disaster, and
disaster with damage. It is the same failure as the pedestrian prototypes
landing on the curb — a real signal, and the wrong one.

`xbd/radiometry.py` estimates the shift from the whole 1024x1024 frame, where a
single building cannot move the statistics, and stores a per-scene gain and
offset. `dataset.py` applies it to the pre crop at load time. It is stored
rather than baked in, so `--no-align` measures how much of the model's
performance was the artifact.

After alignment, mean |pre − post| is still ordered along the damage scale —
no-damage 16.8, minor 21.7, major 26.9 — which is the premise of the study
holding up before any training. Treat that as provisional: it is 50 scenes of
one hurricane so far.

## Steps

```bash
# 1. labels only, ~200 MB -- decides go/no-go before any imagery is pulled
python xbd/fetch_labels.py
python xbd/manifest.py

# 2. paired crops. Visits only scenes holding contested buildings; every
#    building in a visited scene is extracted, so the control task is free.
#    ~2.5 h on a home connection, resumable, ~4 GB.
python -u xbd/extract_crops.py --min-side 24 2>&1 | tee xbd/extract.log

# 3. per-scene radiometric alignment (needs the imagery from step 2 cached)
python xbd/radiometry.py

# 4. LOOK AT THE CROPS before training on them. Rows are buildings, columns
#    are pre / post / |difference|, with the difference at fixed gain so panels
#    are comparable. If major-damage pairs are indistinguishable, the premise is
#    wrong and no training run will rescue it.
python xbd/contact_sheet.py --per-class 8
python xbd/contact_sheet.py --per-class 8 --no-align --out xbd/data/sheet_raw.png

# 5. train (GPU). Checkpoint on val, test scored once.
python xbd/train.py --fold 0                      # contested boundary, paired
python xbd/train.py --fold 0 --no-paired          # ablation: is the pair load-bearing?
python xbd/train.py --fold 0 --task extremes      # control: easy boundary
python xbd/train.py --fold 0 --group disaster     # held-out event

# 6. tornness, with the confound checks in front
python xbd/tornness.py --run xbd/runs/middle_pair_scene_f0_resnet34
```

Set `HF_TOKEN` before step 2 if you have one — unauthenticated fetches hit 429
throttling and the download takes several times longer.

## What would kill this study

In the order the analysis checks them:

1. **Saturation.** If `max_sim` takes a handful of distinct values, co-activation
   has no variance and every statistic built on it is noise. This is precisely
   how the pedestrian experiment failed, undetected, for weeks. `xbd/tornness.py`
   refuses to interpret anything until this passes.
2. **Degradation.** If co-activation survives raw but dies once partialled on
   size and viewing geometry, the measure is a resolution detector.
3. **The control task.** Co-activation on no-damage vs destroyed should be
   markedly lower than on minor vs major. If it is not, it is not tracking
   ambiguity.
4. **The ablation.** If `--no-paired` matches `--paired`, the change signal
   bought nothing and the study's distinguishing claim is gone.

Any of these coming out badly is publishable as a negative result, in the way the
LIDC arm of the HICSS paper was — but only if it is reported, not discovered
later.
