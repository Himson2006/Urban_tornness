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
