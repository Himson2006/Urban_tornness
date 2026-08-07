# The Accessibility Hand-Off — experimental setup

UrbanAI'26, full paper track (8–10pp, ACM sigconf). **Deadline 21 Aug 2026.**

Port of the UA-ProtoPNet structure to urban accessibility triage: an
interpretable model that explains *why* it is uncertain, plus a test of whether
human disagreement is recoverable from the image at all.

Everything below is measured against the real data. Nothing has been trained.

---

## 1. The task

**"Is this crowdsourced accessibility label correct?"**

That is the question Project Sidewalk's own reviewers answer, and it is the
hand-off this paper is about: a city inspector deciding whether to trust a
volunteer's label. Training the model on exactly the decision whose uncertainty
is later compared against reviewer disagreement is what makes the two halves of
the paper the same study rather than two studies stapled together — the
weakness of the HICSS version, where the negative result landed on a dataset the
explanation mechanism was never demonstrated on.

| Role | Requirement | Here |
|---|---|---|
| **A** — competing-prototype explanation | real exemplar patches, provenance | upheld vs. overturned label crops |
| **B** — is disagreement image-predictable? | ≥3 independent human ratings/item | 746,975 reviews, 11,480 reviewers |

### The upgrade over LIDC

LIDC gives one disagreement scalar. Sidewalk gives **two that mean opposite
things**, already separated by `sidewalk/decompose.py`:

- **`split_adj`** — reviewers took opposing positions. A disagreement about
  *where the accessibility standard sits*. Should **not** be image-predictable.
- **`unsure_adj`** — reviewers individually declined to judge. A disagreement
  about *what is visible*. Should **be** image-predictable.

The target is a **dissociation**, not a flat null. Measured
`rho(split_adj, unsure_adj) = +0.33` over the full export and `+0.21` over the
imaged subset — related but far from redundant, so the premise holds. If both
come out null you still have the HICSS-shaped negative; the paper survives
either way, which matters at two weeks out.

---

## 2. Data

### Crops are free and official

Project Sidewalk publishes the crops on HuggingFace, one dataset per label type,
already cut to the label. This removes the Street View Static API, ~$450 of
requests and an unverified zoom→fov mapping. `crops.py` is retained only as a
fallback for label types with no released dataset.

Classes are `correct`/`incorrect` — the validation outcome, not the label type.
Filenames are `{city}_{label_id}.webp`, which is exactly the `city:label_id`
key `targets.py` builds, so the join needs no fuzzy matching.

| task | crops | joined | contested | unclear | role |
|---|---|---|---|---|---|
| `nocurbramp` | 9,150 | 7,433 (81%) | 1,547 (20.8%) | 472 | **primary** |
| `surfaceproblem` | 6,811 | 5,343 (78%) | 665 (12.4%) | 835 | secondary |
| `obstacle` | 5,110 | 4,662 (91%) | 1,080 (23.2%) | **1,728** | unsure-rich |
| `crosswalk` | 1,585 | 915 (58%) | 70 (7.7%) | 75 | Role A control only |

`obstacle` carries by far the most `unclear` mass — 37% of its labels — which is
the image-quality signal Role B predicts *should* be recoverable. `crosswalk` is
the cleanest control (7.7% contested) but is too small for Role B; `train.py`
refuses to run it and says so.

Visual check on `crosswalk`: upheld crops show unmistakable zebra markings;
overturned crops show brick paving, a drainage ditch, a lane arrow, broken
asphalt. The crops are well-centred on the label and the distinction is real.

### Four defects in the release, all handled

1. **There are no CurbRamp crops.** `...-dataset-curbramp` is a byte-identical
   copy of `...-dataset-crosswalk`: same splits, classes and 1,585 filenames.
   All 915 of its rows that join to the export land on `label_type ==
   Crosswalk`. The repo is mislabelled, not merely duplicated. CurbRamp —
   54,676 labels, the highest-volume type and the call a city actually budgets
   against — has **no imagery**. `data.py` omits the repo and verifies the
   `label_type` of every task at load time.
2. **The released splits leak.** 1,547 panoramas, 24.4% of crops, appear in more
   than one of train/val/test. Splits are regenerated grouped on `pano_id`; the
   released ones are kept as `hf_split` for reference only.
3. **The imaged subset is range-restricted on the target.** Assigning
   `correct`/`incorrect` needs consensus, so contested labels are
   under-represented: 18.3% of the imaged subset has `split_adj > 0.5` against
   30.2% of the full export (sd 0.224 vs 0.285). Disagreement survives —
   ~3,360 contested labels — but see §5.
4. **Rate limiting.** Unauthenticated HF pulls hit 429 with ~230s backoffs.
   `data.py` forces the classic HTTP path; **set `HF_TOKEN` to avoid this.**

### A correction to the label key

`label_id` is assigned per deployment, so it is unique only *within* a city:
34,753 ids appear in ≥2 cities (id=8 is a CurbRamp in Amsterdam, a CurbRamp in
Newberg and a Crosswalk in Taipei). Everything here keys on `city:label_id`.
`targets.py --compare-pooled` quantifies what the collision does:

```
                  label var.  reviewer var.        vs pooled fit
Unsure                 27.5%          23.8%    was 14.3% / 20.8%
Disagree               52.6%           8.8%    was 31.2% /  9.5%

unsure_adj  rho(fixed, pooled) = +0.49   mean|delta| = 0.075
split_adj   rho(fixed, pooled) = +0.41   mean|delta| = 0.242
```

The pooled fit roughly halves the apparent label variance. `validator_id` is a
UUID and is unaffected.

---

## 3. Splits

- **Grouped by `pano_id`** (default). Two labels on one panorama share lighting,
  camera and often the same stretch of pavement.
- **Held-out city** (`--held-out-city`). The honest generalisation test and the
  analogue of xBD's held-out-event runs. Not pooled runs; reported apart.

---

## 4. Role A — competing-prototype explanation

UA-ProtoPNet, ResNet-34 backbone frozen by default, K=10 prototypes/class,
standard schedule (warm → joint → push → last-layer). Competing prototypes:
top-two classes by logit, and for each, `argmax_k a_{c,k} W_{c,k}`.

The explanation reads: *"the model is split between this confirmed-genuine
missing curb ramp and this confirmed-spurious one."* Provenance is richer than
HAM10000's — city, severity, `image_capture_date`, and the full reviewer tally.

Baselines: vanilla ProtoPNet, MC-Dropout (30 passes), Deep Ensemble.
Metrics: accuracy, balanced accuracy, AUC, ECE, uncertainty-AUROC.

**Train on consensus, evaluate on contest.** Training excludes contested and
unclear labels by default. A label whose reviewers were split has no trustworthy
target, and training on it teaches the model the noise it is meant to be
uncertain about — which would make Role B circular. This is the LIDC
arrangement: train on the confident nodules, hold out the indeterminate ones.
`--all-train` is the ablation.

**Run the saturation check first.** On pedestrian crops `max_sim` took exactly
two distinct values and every correlation computed from it was meaningless.
Nothing in Role B is worth reading until the measure is shown to vary
(`xbd/tornness.py` has the procedure).

---

## 5. Role B — the dissociation test

For each method, correlate uncertainty against `split_adj` and `unsure_adj`
separately. Prediction: near-zero for `split_adj`, positive for `unsure_adj`.

Every association reported **twice** — raw, and partialled on the confounders,
because `unsure_adj` is exactly the target a resolution detector would fake.
Measured baseline associations over the full export:

```
covariate          vs split_adj   vs unsure_adj
severity                 +0.129         +0.145
n_val                    +0.083         +0.046
pano_width               +0.017         +0.052
image_age_days           -0.012         -0.027
zoom                     -0.029         +0.003
```

`severity` already predicts `unsure_adj` at +0.145, so a model reaching +0.15
has demonstrated nothing. **The partialled figure is the result.**

### The range-restriction caveat

This is the main threat to the paper and must be stated in it. Because the
imaged subset is truncated on the target (§2), a **null is weaker evidence than
a positive**: a flat correlation could be genuine unpredictability or merely
restricted range. Three mitigations, in order of preference:

1. **Lead with the dissociation, not either absolute value.** Restriction
   affects both targets similarly, so the *contrast* between them is far more
   robust than either coefficient alone. This is the main reason to prefer the
   two-target design.
2. Report the range-restriction-corrected correlation alongside the raw one.
3. Report the observable ceiling — the correlation achievable given the observed
   spread — so a null can be read against what was detectable.

---

## 6. Files

| file | does |
|---|---|
| `targets.py` | reviewer-adjusted disagreement targets, city-scoped key |
| `data.py` | fetch HF crops, join, pano-grouped splits, integrity checks |
| `dataset.py` | torch dataset; consensus-only training filter |
| `train.py` | UA-ProtoPNet training; Role B guard; per-crop test outputs |
| `crops.py` | Street View fallback for types with no released crops |

`crops.py` is superseded but kept: it is the only route to CurbRamp imagery, and
its geometry is solved. `heading = camera_heading + pano_x/pano_width*360 − 180`,
`pitch = 90 − pano_y/pano_height*180`. Both heading corrections are
load-bearing — without them `pano_x` and `heading` correlate at r=0.008. With
them the residual has median −0.8° and correlates 0.93 with the label's canvas
offset, and the independent camera-pose derivation agrees to a median 2.2°.
The zoom→fov map remains unverified.

---

## 7. Risks

1. **Range restriction** (§5) — the real one.
2. **No CurbRamp imagery.** The highest-volume type is unavailable without
   ~$170 of Street View requests via `crops.py`.
3. **Scale.** 7,433 crops for the primary task against HAM10000's 10,015 — fine
   for ProtoPNet, thin for strong accuracy claims. Lead with interpretability.
4. **Two weeks**, and no ProtoPNet in this project has been trained end-to-end
   yet (`xbd/runs/` is empty). The schedule risk is training, not data.
5. **`unsure_adj` may be too easy.** If it is mostly "the image is blurry", the
   positive half is unsurprising. The partialled figure decides this.

---

## 8. Order of work

1. `python handoff/targets.py --compare-pooled` — **done.**
2. `python handoff/data.py --type nocurbramp` (also `obstacle`,
   `surfaceproblem`, `crosswalk`) — **in progress**; set `HF_TOKEN` first.
3. `python handoff/train.py --task crosswalk --dry-run` — **done**, plumbing OK
   on MPS.
4. Train `nocurbramp`, pano-grouped. Saturation check before anything else.
5. Baselines: vanilla, MC-Dropout, Deep Ensemble, 3 seeds.
6. Role B correlations, raw and partialled, plus the restriction correction.
7. `obstacle` (the unsure-rich test of the dissociation) and `crosswalk`
   (the Role A control).
8. Held-out-city runs, reported separately.
