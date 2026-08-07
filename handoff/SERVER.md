# Running on a GPU server

Setup for `handoff/`. See `DESIGN.md` for what the experiments are and why.

**Read §0 first** — two of those gotchas silently produce wrong numbers rather
than errors.

---

## 0. Four things that bite

1. **Run every command from the repo root.** `ProtoPNet/resnet_features.py`
   sets `model_dir = './pretrained_models'` — a path relative to the working
   directory, not the module. From anywhere else the ImageNet weights get
   re-downloaded into a stray directory. Not fatal, just confusing.
2. **Do not re-run `sidewalk/fetch_cities.py` or `handoff/targets.py` on the
   server.** Project Sidewalk accumulates labels continuously, so a fresh fetch
   returns a *different, larger* export. Reviewer-adjusted effects, splits and
   every number in `DESIGN.md` would shift, and results across machines would
   stop being comparable. Copy the parquets instead (§2). Regenerate only if
   you deliberately want a newer snapshot, and then regenerate everything.
3. **Set `HF_TOKEN`.** Unauthenticated crop downloads hit HTTP 429 with ~230s
   backoffs; the 8.5 GB pull takes hours instead of minutes. A free read token
   from <https://huggingface.co/settings/tokens> is enough.
4. **`handoff/data/` and `handoff/runs/` are gitignored** (`.gitignore` lines 30
   and 32, which match any directory named `data/` or `runs/` at any depth).
   Code arrives via git; data does not.

---

## 1. Code onto the server

`handoff/` is currently untracked. From the laptop:

```bash
cd "/Users/himsonchapagain/Documents/RAI Lab/Urban Project"
git add handoff/ && git commit -m "handoff: Sidewalk accessibility hand-off study"
git push
```

Then on the server:

```bash
git clone <your-remote> urban && cd urban
```

No remote? `rsync -av --exclude data --exclude runs --exclude __pycache__ \
  "/Users/himsonchapagain/Documents/RAI Lab/Urban Project/" server:~/urban/`

---

## 2. Data onto the server (42 MB)

The targets and manifests are small and must match exactly (§0.2). From the
laptop:

```bash
cd "/Users/himsonchapagain/Documents/RAI Lab/Urban Project"
rsync -av handoff/data/*.parquet server:~/urban/handoff/data/
```

That is `targets.parquet` (36 MB) plus the four `manifest_*.parquet`. It carries
the reviewer-adjusted disagreement effects, the pano-grouped split assignments
and the city-scoped `uid` key — everything except pixels.

---

## 3. Environment

```bash
cd ~/urban
python -m venv .venv && source .venv/bin/activate

# torch first, matched to the server's CUDA. Check with `nvidia-smi`.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r handoff/requirements.txt

python -c "import torch,cv2;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`cv2` and `matplotlib` are not optional — ProtoPNet's push step imports both.

---

## 4. Crops (8.5 GB)

```bash
export HF_TOKEN=hf_...
cd ~/urban
for t in nocurbramp obstacle surfaceproblem crosswalk; do
  python handoff/data.py --type "$t"
done
```

`data.py` re-derives the manifests after downloading. Because it reads the
`targets.parquet` you copied, the splits come out identical to the laptop's.
It also re-verifies the release integrity and will stop if the published
`label_type` of any task has changed.

Expected, and worth checking against:

```
nocurbramp      7,433 crops   0 panos straddling our splits
obstacle        4,662 crops   0
surfaceproblem  5,343 crops   0
crosswalk         915 crops   0
```

### Optional: pre-resize (recommended)

Crops are 640×640 WebP and get resized to 224 every epoch, so **data loading,
not the GPU, is the bottleneck**. If epochs look CPU-bound:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
for p in Path("handoff/data/hf").rglob("*.webp"):
    im = Image.open(p)
    if max(im.size) > 256:
        im.convert("RGB").resize((256, 256), Image.BILINEAR).save(p, quality=95)
PY
```

This overwrites in place and is destructive — re-run §4 to undo. 256 leaves
room for the 224 resize. Skip it if `--workers 16` already saturates the GPU.

---

## 5. Smoke test

```bash
cd ~/urban
python handoff/train.py --task crosswalk --dry-run --workers 4
```

Expect: split sizes, a `WARNING` that crosswalk is too small for Role B (that
is correct — it is the Role A control), then `dry run OK` with batch
`(64, 3, 224, 224) -> logits (64, 2), min-dists (64, 20)`.

---

## 6. Train

```bash
cd ~/urban
python handoff/train.py --task obstacle    --batch 128 --workers 16
python handoff/train.py --task nocurbramp  --batch 128 --workers 16
```

**Run `obstacle` first.** Its test split carries 242 unclear against 169
contested — the only task where the two kinds of disagreement are close to
balanced, which is what the dissociation test in `DESIGN.md` §5 actually needs.
`nocurbramp` is contested-heavy (230 vs 63): it tests the negative half well and
the positive half weakly.

Each run writes to `handoff/runs/<tag>/`:

| file | contents |
|---|---|
| `train.log` | per-epoch val accuracy, phase transitions, push events |
| `best.pth` | checkpoint, selected on val |
| `scores.json` | test accuracy, balanced accuracy, AUC, majority baseline |
| `test_preds.parquet` | **per-crop `uncertainty`, `split_adj`, `unsure_adj`** — the Role B input |
| `protos/` | pushed prototype images, for the Role A figures |

A finished run drops a `DONE` file and re-running skips it; delete `DONE` to
retrain.

### Seeds and the ensemble baseline

A Deep Ensemble is just several seeds averaged, so it comes almost free:

```bash
for s in 0 1 2; do
  python handoff/train.py --task obstacle --seed $s --batch 128 --workers 16
done
```

Three seeds also give the mean±std that `DESIGN.md` reports throughout.

### Control and ablation

```bash
python handoff/train.py --task crosswalk  --batch 128   # Role A control
python handoff/train.py --task obstacle --all-train     # ablation: circular
```

`--all-train` trains on contested labels too. It exists to show the number you
get when Role B *is* circular — do not report it as a main result.

### Held-out city

```bash
python handoff/train.py --task obstacle --held-out-city chicago
```

Not pooled runs. Report them separately, per the xBD work.

---

## 7. What is NOT built yet

Being explicit so nothing looks finished that isn't:

- **The Role B analysis script.** `test_preds.parquet` has every column it
  needs — `uncertainty`, `split_adj`, `unsure_adj`, `severity`, `n_val`,
  `city` — but the partial correlations, the range-restriction correction and
  the dissociation contrast (`DESIGN.md` §5) are not written.
- **The saturation check.** `xbd/tornness.py` has the procedure. **No Role B
  correlation should be read before this passes** — on pedestrian crops
  `max_sim` took two distinct values and every correlation from it was
  meaningless.
- **MC-Dropout.** Needs a code change (dropout at inference, N stochastic
  passes). Vanilla ProtoPNet and Deep Ensemble are reachable with existing
  flags; MC-Dropout is not.
- **The competing-prototype figures.** `src/competing_prototypes.py` renders
  them for the xBD/pedestrian runs and needs adapting to this run layout.

---

## 8. Quick reference

```bash
cd ~/urban && source .venv/bin/activate && export HF_TOKEN=hf_...

python handoff/train.py --task obstacle --dry-run --workers 4   # verify
python handoff/train.py --task obstacle --batch 128 --workers 16

tail -f handoff/runs/obstacle_consensus_pano_resnet34_s0/train.log
python -c "import json;print(json.load(open('handoff/runs/obstacle_consensus_pano_resnet34_s0/scores.json')))"
```

Useful flags: `--arch resnet50`, `--epochs`, `--protos-per-class` (default 10),
`--features-lr 1e-4` to unfreeze the backbone (it is frozen by default — an
unfrozen backbone collapsed the feature space in the pedestrian work).
