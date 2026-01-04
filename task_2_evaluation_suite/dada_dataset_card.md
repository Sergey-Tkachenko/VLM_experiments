# DADA-2000 — Dataset card (computed from XLSX annotations)

> Generated from `dada_text_annotations.xlsx` (sheets `Sheet1` + `text`).

## 0. Purpose

DADA-2000 is a dashcam accident dataset intended for **accident understanding/anticipation** and **driver attention modeling**. It provides accident categories, temporal accident windows, crash-object localization (in the full dataset), and driver-attention signals. This card focuses on what’s present in the provided XLSX annotations.

**References**

- Paper (original benchmark): https://arxiv.org/abs/1904.12634

- Fang, Jianwu, Dingxin Yan, Jiahuan Qiao, Jianru Xue, and Hongkai Yu. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE Transactions on Intelligent Transportation Systems 23, no. 6 (2022): 4959-4971.

- GitHub / download page (LOTVS-DADA): https://github.com/JWFangit/LOTVS-DADA

- Mirror I used: https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/main

## 1. Dataset structure and annotation file contents

The DADA-2000 dataset is organized as follows:

### File and Folder Structure

The dataset contains a large collection of road accident video samples, each represented as a folder of sequential images (frames). The directory structure is illustrated below:

```
/workspace/datasets/mm-au/Origin/DADA2000/DADA2000/
└── 1/                           # Example subset/fold
    └── 001/                     # Sample ID
        └── images/
            ├── 0001.png
            ├── 0002.png
            ├── 0003.png
            ├── ...
            └── 0409.png        # Last frame in this sample
```

Each `images/` directory contains frame images in PNG format, named with sequential numbering.

*Note:* The original dataset release also includes segmentation masks, driver attention masks, and other modalities in sibling folders to the images, but these are **out of scope** for this evaluation.

### Annotation spreadsheet

Annotations are stored in an XLSX file:
```
/workspace/datasets/mm-au/dada_text_annotations.xlsx
```
with the following important sheets:

- **`Sheet1`**: Per-sample metadata including weather, lighting, scene, road type, accident event `type`, accident flag, frame markers (`abnormal start`, `accident frame`, `abnormal end`), and text fields (`texts`, `causes`, `measures`).
- **`text`**: Lookup dictionary of candidate text descriptions, grouped by ID (38 groups in this file).

**Only the image sequences and spreadsheet annotations are currently considered; segmentation/attention masks and other auxiliary files from the original dataset are not used in this evaluation.**

## 2. General stats

- **Samples in this XLSX (`Sheet1` rows):** 1,962
- **Unique `video` IDs in this XLSX:** 255 *(note: `video` repeats; treat each row as one annotated sample)*
- **Accident present:** 1,945 / 1,962 (99.13%)
- **No-accident samples:** 17 / 1,962 (0.87%)
- **Total frames (sum of `total frames` across samples):** 649,399

**Dataset release notes (not computable from XLSX):**

- **Video resolution:** commonly reported as **1584×660**

- **FPS:** commonly evaluated at **30 fps**

- **Compressed size:** GitHub release note mentions ~53 GB for a train/test release (and ~116 GB for a “full benchmark”).

### Clip length (`total frames`)

- **All samples** (frames): mean 330.99, median 322, p05 160, p25 240.25, p75 414.75, p95 519.90 (≈ seconds @ 30fps: mean 11.03s, median 10.73s, p25 8.01s, p75 13.82s)

- **Accident samples only** (frames): mean 331.08, median 322, p05 160, p25 240, p75 415, p95 520 (≈ seconds @ 30fps: mean 11.04s, median 10.73s, p25 8.00s, p75 13.83s)

- **No-accident samples only** (frames): mean 321.06, median 340, p05 151.60, p25 280, p75 410, p95 432 (≈ seconds @ 30fps: mean 10.70s, median 11.33s, p25 9.33s, p75 13.67s)

### Temporal structure (derived from frame markers)

- **Pre-abnormal context (`t_ai`)** (frames): mean 133.28, median 115, p05 1, p25 50, p75 186, p95 330 (≈ seconds @ 30fps: mean 4.44s, median 3.83s, p25 1.67s, p75 6.20s)

- **Abnormal window length (`t_ae - t_ai`)** (frames): mean 96.91, median 90, p05 54, p25 75, p75 112, p95 158 (≈ seconds @ 30fps: mean 3.23s, median 3.00s, p25 2.50s, p75 3.73s)

- **Time-to-collision inside abnormal window (`t_co - t_ai`, accident only)** (frames): mean 49.28, median 45, p05 21, p25 34, p75 60, p95 90 (≈ seconds @ 30fps: mean 1.64s, median 1.50s, p25 1.13s, p75 2.00s)

- **Post-collision part inside abnormal window (`t_ae - t_co`, accident only)** (frames): mean 47.64, median 42, p05 22, p25 33, p75 56, p95 92 (≈ seconds @ 30fps: mean 1.59s, median 1.40s, p25 1.10s, p75 1.87s)

- **Tail after abnormal window (`end - t_ae`)** (frames): mean 100.80, median 85, p05 6, p25 46, p75 135, p95 250 (≈ seconds @ 30fps: mean 3.36s, median 2.83s, p25 1.53s, p75 4.50s)

Additional ratios (computed):

- Mean abnormal-window fraction of the clip: 0.324 (median 0.297)
- Mean accident-frame position in the clip (accident samples): 0.536 (median 0.540)

## 3. Metadata distributions

### Weather

| weather | count | share |
| --- | --- | --- |
| sunny | 1824 | 92.97% |
| rainy | 129 | 6.57% |
| snowy | 3 | 0.15% |
| foggy | 6 | 0.31% |

### Light

| light | count | share |
| --- | --- | --- |
| day | 1771 | 90.27% |
| night | 191 | 9.73% |

### Scene

| scene | count | share |
| --- | --- | --- |
| highway | 184 | 9.38% |
| tunnel | 12 | 0.61% |
| mountain | 55 | 2.80% |
| urban | 1319 | 67.23% |
| rural | 392 | 19.98% |

### Road geometry (`linear`)

| road_type | count | share |
| --- | --- | --- |
| arterials | 840 | 42.81% |
| curve | 99 | 5.05% |
| intersection | 610 | 31.09% |
| T-junction | 391 | 19.93% |
| ramp | 22 | 1.12% |

## 4. GT categories (`type`) + accident rate

- **# unique `type` IDs in this XLSX:** 52

Per-type distribution with accident/non-accident breakdown:

| type_id | total | accident | non_accident | accident_rate | share |
| --- | --- | --- | --- | --- | --- |
| 11 | 252 | 249 | 3 | 98.81% | 12.84% |
| 43 | 209 | 209 | 0 | 100.00% | 10.65% |
| 50 | 201 | 201 | 0 | 100.00% | 10.24% |
| 10 | 167 | 166 | 1 | 99.40% | 8.51% |
| 5 | 157 | 157 | 0 | 100.00% | 8.00% |
| 6 | 119 | 119 | 0 | 100.00% | 6.07% |
| 37 | 85 | 84 | 1 | 98.82% | 4.33% |
| 48 | 82 | 82 | 0 | 100.00% | 4.18% |
| 38 | 67 | 67 | 0 | 100.00% | 3.41% |
| 8 | 57 | 53 | 4 | 92.98% | 2.91% |
| 1 | 53 | 48 | 5 | 90.57% | 2.70% |
| 57 | 45 | 45 | 0 | 100.00% | 2.29% |
| 12 | 42 | 42 | 0 | 100.00% | 2.14% |
| 49 | 40 | 39 | 1 | 97.50% | 2.04% |
| 56 | 37 | 37 | 0 | 100.00% | 1.89% |
| 14 | 31 | 31 | 0 | 100.00% | 1.58% |
| 39 | 30 | 30 | 0 | 100.00% | 1.53% |
| 42 | 23 | 23 | 0 | 100.00% | 1.17% |
| 9 | 20 | 20 | 0 | 100.00% | 1.02% |
| 51 | 18 | 18 | 0 | 100.00% | 0.92% |
| 41 | 18 | 18 | 0 | 100.00% | 0.92% |
| 61 | 18 | 18 | 0 | 100.00% | 0.92% |
| 3 | 17 | 17 | 0 | 100.00% | 0.87% |
| 59 | 17 | 17 | 0 | 100.00% | 0.87% |
| 24 | 17 | 17 | 0 | 100.00% | 0.87% |
| 40 | 16 | 15 | 1 | 93.75% | 0.82% |
| 4 | 13 | 13 | 0 | 100.00% | 0.66% |
| 7 | 12 | 12 | 0 | 100.00% | 0.61% |
| 21 | 12 | 12 | 0 | 100.00% | 0.61% |
| 53 | 11 | 11 | 0 | 100.00% | 0.56% |
| 52 | 11 | 11 | 0 | 100.00% | 0.56% |
| 18 | 10 | 10 | 0 | 100.00% | 0.51% |
| 45 | 9 | 9 | 0 | 100.00% | 0.46% |
| 13 | 9 | 9 | 0 | 100.00% | 0.46% |
| 2 | 6 | 6 | 0 | 100.00% | 0.31% |
| 58 | 4 | 4 | 0 | 100.00% | 0.20% |
| 55 | 3 | 3 | 0 | 100.00% | 0.15% |
| 44 | 3 | 3 | 0 | 100.00% | 0.15% |
| 54 | 3 | 2 | 1 | 66.67% | 0.15% |
| 15 | 3 | 3 | 0 | 100.00% | 0.15% |
| 60 | 3 | 3 | 0 | 100.00% | 0.15% |
| 36 | 2 | 2 | 0 | 100.00% | 0.10% |
| 23 | 1 | 1 | 0 | 100.00% | 0.05% |
| 47 | 1 | 1 | 0 | 100.00% | 0.05% |
| 16 | 1 | 1 | 0 | 100.00% | 0.05% |
| 17 | 1 | 1 | 0 | 100.00% | 0.05% |
| 19 | 1 | 1 | 0 | 100.00% | 0.05% |
| 20 | 1 | 1 | 0 | 100.00% | 0.05% |
| 22 | 1 | 1 | 0 | 100.00% | 0.05% |
| 33 | 1 | 1 | 0 | 100.00% | 0.05% |
| 30 | 1 | 1 | 0 | 100.00% | 0.05% |
| 34 | 1 | 1 | 0 | 100.00% | 0.05% |

## 5. Text fields / data quality notes

- `texts` missing: 1 row(s); `measures` missing: 10 row(s).
- Unique normalized `texts` strings (after `.strip()`): 96.
- Many `texts` entries contain trailing spaces; strip them before grouping/deduplicating.

## 6. Suggested VLM evaluation targets (aligned to this XLSX)

- **Event type classification**: predict `type` as a categorical label (macro-F1 / per-class accuracy).
- **Temporal grounding**: predict `{abnormal_start, accident_frame, abnormal_end}` (MAE in frames/seconds; optional IoU over abnormal window).
- **Structured “safety report” generation**: output JSON with `{type, accident_happened, cause, measure, timings}` and track parsing success rate.


## 7. Qualitiative analysis

### Summary

I reviewed 1% of the dataset -- 20 samples. Annotation is mostly correct, with only 1 sample of 20 contained ambiguous annotation, + hitting/crossing concent is a bit unclear.

2 of 20 has some artifacts in videos -- "leaps" and concatenated videos. Both does not expected to affect training, however, it is unclear how many of such "artifacts" are present on the vide.

Further analysis may include more videos (~100) to ensure artifacts are rare.


#### Most common types

### Type = 11
#### 001

Annotation reflects the video: weather is rainy, accident happended and falls into specified category.

Detection abnormal frames is still unclear yet I guess it can be based on distance to object when hitting.

#### 200

Annotation: type is good, weather and day/night correct. However, I personally see the highway while annotation suggests rural (also acceptable).

Abnormal frames: clear, accident frame can be detected cleanly.


#### 12/040 (adjacent)

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

#### 10/050 (adjacent)

Annotation: mostly correct, however, I am not sure about cause, because I can not see if the ego-car turned the signal lights or not. Also I am unsure that rules says about such manevour. However, it seems that it is the ego-driver who is responsible for this, so I mark it as OK.00s

Abnormal frames: clear, accident frame can be detected cleanly.

### Type = 43

This category is effectively car hitting another car.

#### 43/001

Annotation: type is good, weather and day/night correct.

Abnormal frames: clear, accident frame can be detected cleanly.

#### 43/011

Tunnel/night setting, interesting

Annotation: type is good, weather and day/night correct.

Abnormal frames: clear, accident frame can be detected cleanly.

#### 43/085

Annotation: type is good, weather and day/night correct, ramp label seem also to be valid.

Abnormal frames: cover almost full video, not sure about them.

### Type = 50

#### 001

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

#### 070

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.


#### 096

Rural setting

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.


### Type = 10

#### 001

Annotation: not sure about T junction. Also it is strange because car actually hits, why is it crossing instead of hitting -- unclear.

Abnormal frames: human can see it cleanly, however, for VLM it will be hard -- it is clean only due to the ego-vehicle gets shaken.

Also this is strange clip since it is actually two clips -- one with annotation and other with completely different settings.

#### 015

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

Interestingly, everything happens in the parking.

#### 130

Annotation: completely unclear -- I was not able to understand if something is incorrect, it looked like sudden stop.

Abnormal frames: hard to identify for very same reason.

### Type = 5

### 001

Nigth setting

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

Interestinly, it seem some parts of the clip were "cut", i.e. there are visible "leaps" in video.


### 002

Day setting, rural

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### 008

Rainy weather

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### Type 6

#### 040

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### Type 37

#### 030

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### Type 48

#### 023

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### Type 38

#### 015

Annotation: correct

Abnormal frames: clear, accident frame can be detected cleanly.

### Type 8

#### 007

Annotation: correct

Abnormal frames: unclear, at which moment truck violated the rules.

### Type = 1

#### 003

Annotation: correct

Abnormal frames: unclear, at which moment truck violated the rules.

## 8. License

License is not clearly stated in the provided XLSX; verify the dataset’s usage constraints on the official release page.
