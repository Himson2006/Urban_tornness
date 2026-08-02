# RESULTS

Tornness project -- running log.

## Step A -- PIE manifest (annotations only, no video downloaded)

- pedestrian tracks with attributes: **1,842**
- tracks with `intention_prob`: **1,842**
- annotated pedestrian boxes: **740,901** (740,901 on intent-labelled tracks)

### Per set

| set | split | videos | ped tracks | w/ intent | boxes | boxes (intent peds) |
|---|---|---|---|---|---|---|
| set01 | train | 4 | 111 | 111 | 51,617 | 51,617 |
| set02 | train | 3 | 104 | 104 | 54,480 | 54,480 |
| set03 | test | 19 | 719 | 719 | 294,219 | 294,219 |
| set04 | train | 16 | 665 | 665 | 255,296 | 255,296 |
| set05 | val | 2 | 16 | 16 | 4,656 | 4,656 |
| set06 | val | 9 | 227 | 227 | 80,633 | 80,633 |

### intention_prob distribution (the torn-case population)

- mean 0.712, median 0.850, n unique values 61
- values are multiples of 1/60 -> consistent with ~15 annotators on a 5-point scale (61 distinct levels observed)

| bin | n | % |
|---|---|---|
| 0.0-0.1 | 132 | 7.2% |
| 0.1-0.2 | 102 | 5.5% |
| 0.2-0.3 | 82 | 4.5% |
| 0.3-0.4 | 48 | 2.6% |
| 0.4-0.5 | 58 | 3.1% |
| 0.5-0.6 | 82 | 4.5% |
| 0.6-0.7 | 105 | 5.7% |
| 0.7-0.8 | 141 | 7.7% |
| 0.8-0.9 | 309 | 16.8% |
| 0.9-1.0 | 783 | 42.5% |

**Contested population** (how many pedestrians humans were split on):

| band | n | % of labelled |
|---|---|---|
| 0.40-0.60 | 140 | 7.6% |
| 0.35-0.65 | 221 | 12.0% |
| 0.30-0.70 | 300 | 16.3% |
| 0.25-0.75 | 399 | 21.7% |
| 0.20-0.80 | 548 | 29.8% |

### Behavioural outcome vs. intent

| crossing | meaning | n | mean intention_prob |
|---|---|---|---|
| -1 | irrelevant (near road, not intending) | 468 | 0.260 |
| +0 | did not cross | 855 | 0.838 |
| +1 | crossed in ego path | 519 | 0.910 |

### Track geometry (bounds the resolvability experiment)

- `n_frames`: p5=78 p25=169 med=273 p75=468 p95=1253
- `exp_window_len`: p5=30 p25=74 med=90 p75=90 p95=121
- `frames_before_critical`: p5=35 p25=95 med=166 p75=301 p95=905
- `frames_after_critical`: p5=18 p25=29 med=49 p75=186 p95=478
- `median_bbox_h`: p5=42 p25=63 med=91 p75=157 p95=297
- tracks with >=45 frames after critical_point: **1015** (55.1%)

### Per-frame behaviour tags (free cue labels for dual-match cases)

- `action`: walking=382,971, standing=357,930
- `look`: not-looking=670,750, looking=70,151
- `cross`: not-crossing=599,473, crossing=118,471, crossing-irrelevant=22,957
- `gesture`: __undefined__=739,375, other=1,151, hand_ack=167, hand_yield=91, hand_rightofway=80, nod=37

### Crop storage estimate (2x bbox, JPEG q90)
```
  all annotated ped boxes           740,901 crops  ~ 10.49 GB
  boxes on intent-labelled peds     740,901 crops  ~ 10.49 GB
```

---

## Step A findings that change the design

### 1. Label design is settled by the data: use `crossing != -1`, not `crossing == 1`

PIE separates *intention* from *action* deliberately. Mean `intention_prob` by outcome:

| grouping | negative class | positive class | separation |
|---|---|---|---|
| **intent** = `crossing != -1` | 0.260 (n=468) | 0.866 (n=1374) | clean |
| **action** = `crossing == 1` | 0.634 (n=1323) | 0.910 (n=519) | muddy |

855 pedestrians did not cross yet carry mean `intention_prob` 0.838 — they intended
to cross and were blocked (red light, no gap, vehicle not yielding). Training on the
action label puts those clear intenders in the "won't cross" prototype set, so every
dual-match would be a label artefact rather than genuine cue conflict.

**Decision: binary target is `intent_binary = (crossing != -1)`.** Class balance
1374/468 ≈ 75/25, so ProtoPNet's CE needs class weighting
(`pie_dataset.class_weights`).

Consequence for Experiment 2: the resolution anchor for resolvability curves must
also be `intent_binary`, not "did they physically cross".

### 2. Contested population is thin on the official split

| band | all | train | val | test |
|---|---|---|---|---|
| 0.40–0.60 | 140 | 62 | 18 | **60** |
| 0.30–0.70 | 300 | 138 | 37 | **125** |

`intention_prob` is heavily skewed — 42.5% of pedestrians sit in [0.9, 1.0]. The
deferral experiment on the official split would run on ~60 held-out contested
pedestrians, which is too few for meaningful deferral curves.
Mitigation: PIE's native k-fold split (`data_split_type='kfold'`) gives held-out
predictions for all 300 contested cases.

### 3. Storage is not the binding constraint — video download is

Crops for **all six sets**, experiment window + 45-frame tail, 2× bbox, JPEG q90:

| scope | crops | size |
|---|---|---|
| full tracks → critical+45 | 740,901 | ~10.5 GB |
| exp window → critical+45 | 225,350 | **~3.1 GB** |
| exp window only | 156,927 | ~1.4 GB |

Per-set crop footprint (window+45): set01 0.21, set02 0.16, set03 1.15, set04 1.27,
set05 0.02, set06 0.28 GB. Peak disk = one video set (~10–15 GB) + ≤3.1 GB of crops.
86 GB free ⇒ the whole dataset is reachable by rolling through sets.

### 4. Track geometry bounds Experiment 2

- `exp_window_len` median **90 frames (~3 s)** — the clip humans actually saw.
- `frames_after_critical`: median 49; **only 55.1% of tracks have ≥45 frames**
  past the critical point. Resolvability tails beyond critical_point are available
  for ~1015 of 1842 pedestrians.
- Within-window resolvability (exp_start → critical) is available for **all** tracks
  and is the better-matched comparison, since `intention_prob` is a judgement about
  exactly that window.

### 5. Free per-frame cue labels (substitute for PSI's text reasoning)

Every box carries `action` (walking/standing), `look` (looking/not-looking),
`cross`, `gesture`. In-window co-occurrence: standing+looking = 11.0% of frames —
this is the running example's "at the curb, checking traffic, not moving" cue
conflict, labelled, for free.

**Preliminary signal (weak but correctly signed):** 308 pedestrians are
standing >50% and looking >20% of in-window frames. Their mean |intention_prob − 0.5|
is **0.313** vs **0.358** for the rest — humans were measurably more split on
dual-cue pedestrians. Not proof, but the hypothesis is not dead on arrival.

### Threats surfaced by Step A

- **Intention–action gap** contaminates any outcome-anchored resolvability claim
  (see #1). Anchor on intent, report action separately.
- **Only aggregate `intention_prob`** is released — values are multiples of 1/60,
  consistent with ~15 annotators on a 5-point scale, but per-annotator responses are
  not in the repo. Bimodal-vs-diffuse disagreement *shape* is therefore **not**
  recoverable from PIE. That claim needs CIFAR-10H or PSI.
- **Skew**: 42.5% of tracks are near-unanimous crossers; contested cases are rare
  and may correlate with small/blurry crops — hence the degradation stats computed
  at extraction time.

## Step B / C status

- `src/pie_extract.py` — rolling per-set extraction. Sequential decode (grab/retrieve),
  resumable per video, checkpoints metadata after every clip, integrity checks +
  20-crop contact sheet, prints "safe to delete" only on pass. **Never deletes.**
- `src/pie_dataset.py` — `PIECropDataset` (ProtoPNet training) and `PIETrackDataset`
  (resolvability sequences), official splits, class weights.
- Verified without video: synthetic-clip end-to-end run (62/62 crops, checks pass)
  and a frame-alignment test (solid-grey frames, luma error ≤2 ⇒ no off-by-one).

---

## Decisions taken

**Temporal window: experiment window + 45.** `pie_extract.py` now defaults to
`exp_start_point .. critical_point + 45` (`--full-track` opts back into whole
tracks). 225,350 crops, ~3.1 GB for all six sets. Rationale: `intention_prob` is a
judgement about exactly these frames, so model and annotators see the same evidence.

**Split: official for the headline classifier, grouped k-fold for contested-case
analyses.** Fixed assignment in `data/pie_manifest/folds_k5_video_seed0.parquet`.

Note we do *not* use PIE's native k-fold. `pie_data.py::_get_kfold_pedestrian_ids`
runs a plain shuffled `KFold` over pedestrian ids, so pedestrians from the same
10-minute clip appear in both train and test. For ProtoPNet that leaks directly:
prototypes are projected onto real training patches, which can come from the test
fold's own scene. `pie_dataset.kfold_assign` uses `StratifiedGroupKFold` grouped on
video, stratified on (intent_binary x contested).

| fold | peds | videos | pos rate | contested 0.3–0.7 | contested 0.4–0.6 |
|---|---|---|---|---|---|
| 0 | 362 | 9 | 0.749 | 60 | 30 |
| 1 | 363 | 7 | 0.738 | 60 | 24 |
| 2 | 372 | 8 | 0.758 | 61 | 29 |
| 3 | 382 | 7 | 0.759 | 58 | 30 |
| 4 | 363 | 9 | 0.725 | 61 | 27 |

Videos spanning more than one fold: **0**. Held-out contested cases: **300**
(vs 125 on the official split, a 2.4x gain). Grouping on `set` instead was tested
and rejected — fold sizes range 111–719 pedestrians, contested counts 16–125.

## Next step (blocked on video download)

```bash
# 1. download ONE set from http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/
#    into <clips>/set01/video_0001.mp4 ...
python src/pie_extract.py --set set01 --clips /path/to/PIE_clips
# 2. on "SAFE TO DELETE", remove the videos yourself, then repeat for set02 ...
python src/pie_dataset.py          # sanity summary once >=1 set is extracted
```

Suggested order: set05 (0.02 GB crops, 2 videos) first as a live smoke test,
then set01, set02, set06, set03, set04.
