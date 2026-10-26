# experiment3 — plan

A clean-room rebuild of the crop stress detector. **No code is copied from
`hybrid_pipeline*`.** Only the dataset is shared. Everything — features, models,
evaluation, figures, reports — is produced inside `experiment3/`.

Goal: same project, better results, and a methodology that survives review.

---

## 1. Why the results should improve

Three findings from auditing the existing pipeline drive this rebuild.

### The raw archive is 9-band, not one band

`dataset_clean_2/` holds only derived `_MSAVI.tif`. But
`archive (1)/lombardia/data2019/<tile>/YYYYMMDD.tif` is **9-band multispectral** —
B2, B3, B4, B5, B6, B7, B8, B11, B12. Verified against a July acquisition: blue 624,
green 962, red 1030, then the red-edge climb 1532 → 2833 → 3352, NIR plateau 3349, SWIR
falloff 2745 → 1717. Textbook healthy vegetation.

So four indices are available instead of one:

| Index | Formula | What it senses |
|---|---|---|
| MSAVI | soil-adjusted NIR/red | canopy greenness, soil-corrected |
| NDVI | (NIR−Red)/(NIR+Red) | canopy greenness |
| NDRE | (NIR−RE1)/(NIR+RE1) | chlorophyll content — **reacts earlier than NDVI** |
| NDMI | (NIR−SWIR1)/(NIR+SWIR1) | **canopy water content** |

Validated on real tiles: NDVI spans 0.0–0.83, peaks in July for a summer crop and April
for a winter cereal. NDRE tracks NDVI at lower magnitude, as it should.

**Why this matters for results:** the current pipeline's worst case is *gradual onset*
(F1 0.565 against 0.718 for a sudden drop). Gradual decline is exactly what water stress
looks like, and NDMI measures water content directly. NDRE responds to chlorophyll loss
before it is visible in NDVI. These target the weakness head-on.

### Two-thirds of the temporal resolution is being discarded

Each tile has **71 raw acquisition dates**. The current pipeline selects 25. More dates
means finer percentile resolution — the existing threshold sweep goes degenerate above P94
purely because 25 points cannot resolve a higher percentile — and more reliable
persistence statistics.

### Cloud screening can be spectral instead of heuristic

The current filter infers cloud from image blurriness (spatial gradient). With 9 bands
available, cloud can be detected directly: clouds are bright in blue and bright in SWIR,
where vegetation is dark. That is a far more reliable discriminator, and better input data
raises every downstream number.

---

## 2. What gets fixed

Problems identified in the existing pipeline, and how experiment3 avoids each.

| Problem in current pipeline | Fix here |
|---|---|
| **No baseline comparison** — nothing showed ML beats a simple rule | Three rule-based baselines are first-class citizens, evaluated in every experiment |
| **No held-out split** — models fitted on all 3,328 tiles, then scored on those same tiles | Tiles split 60/40 into fit and evaluation sets. Nothing is ever scored on a tile it was fitted on |
| **Threshold computed on the post-injection batch** | Thresholds calibrated on **clean fit-set data only**, then applied unchanged. Removes the leakage caveat entirely |
| **PCA fitted but never used** | Not included. If dimensionality reduction is not in the scoring path it does not go in the pipeline |
| **`contamination` inert but presented as tuned** | Not used. Decisions come from percentile thresholds on raw scores |
| **`n_neighbors=10` never swept** | Swept, with the choice justified by a figure |
| **Sliding-window persistence peeks at the future** | Causal rules only. Nothing looks past the current observation |
| **No significance testing** | Bootstrap confidence intervals on every headline metric; paired tests between configurations |
| **Spatial overlay was decorative, not model output** | Either genuinely per-pixel or labelled plainly as a diagnostic |
| **cp1252 encoding, stale figures, captions contradicting tables** | One command regenerates every figure and table from current results. UTF-8 throughout |

---

## 3. Pipeline design

```
raw 9-band GeoTIFF
   → spectral cloud/quality screening
   → 4 vegetation indices per acquisition
   → per-tile per-date summary statistics
   → temporal feature engineering (per index)
   → standardise
   → LOF  (fitted on CLEAN FIT-SET tiles only)
   → per-field adaptive threshold (calibrated on clean data)
   → causal persistence rule
   → alarm
```

### Stage detail

**S1 · Ingest.** Read each raw raster, build a valid-pixel mask, screen for cloud
spectrally (bright blue + bright SWIR + low NDVI), compute the four indices, and reduce
each to summary statistics: mean, std, and the 10/25/50/75/90th percentiles. Parallelised
across 8 workers — benchmarked at ~11.6 ms per raster, so ~7 minutes for all 236,288
rasters. Cached to Parquet so it runs once.

**S2 · Date selection.** Keep dates passing quality screening, targeting ~40 per tile with
a minimum spacing constraint so the season stays evenly covered. Tiles with too few usable
dates are dropped rather than padded.

**S3 · Features.** Per index: Savitzky-Golay smoothed level, change since previous pass,
rate of change, rolling median and dispersion, within-field spread (IQR), and deviation
from the same-date median across all tiles. Plus cross-index terms — the NDMI/NDVI ratio
separates water stress from chlorophyll loss, which single-index pipelines cannot do.

**S4 · Split.** 60% of tiles for fitting, 40% held out for evaluation. Split by tile, not
by row, so no field appears in both.

**S5 · Detect.** LOF in novelty mode as the primary detector; Isolation Forest and a
PyTorch autoencoder for comparison; three rule-based baselines.

**S6 · Decide.** Per-field threshold from the clean fit-set distribution, then a causal
N-consecutive persistence rule.

---

## 4. Experiments

| # | Question | Output |
|---|---|---|
| E1 | **Does multispectral beat MSAVI-only?** | The headline result |
| E2 | Which detector wins, and does it beat the naive rules? | Model + baseline comparison |
| E3 | What is the right `n_neighbors`? | Sensitivity curve |
| E4 | Per-field vs global thresholding, and at what percentile? | Operating-point curve |
| E5 | Which persistence rule? | Comparison, causal only |
| E6 | What are the detection limits? | Severity × duration heatmap |
| E7 | Does it hold for gradual and spectral-shift anomalies? | Robustness comparison |
| E8 | Does it generalise to unseen fields? | Held-out evaluation |

Every experiment: 5 seeds, mean ± bootstrap CI.

**Injection model.** Physically coherent across indices — a stress event lowers the
greenness indices (MSAVI, NDVI, NDRE) and lowers NDMI, rather than perturbing one channel
in isolation. Three shapes: sudden drop, gradual onset, spectral shift.

---

## 5. Deliverables

```
experiment3/
├── PLAN.md              ← this file
├── README.md            ← how to run, what everything is
├── config.py            ← every parameter, single source
├── src/
│   ├── indices.py       ← band maths + spectral cloud screening
│   ├── ingest.py        ← raw rasters → statistics (parallel, cached)
│   ├── features.py      ← temporal feature engineering
│   ├── detectors.py     ← LOF / IForest / autoencoder / baselines
│   ├── decision.py      ← thresholding + causal persistence
│   ├── evaluation.py    ← injection, metrics, bootstrap CIs
│   └── viz.py           ← figure style + all plotting
├── run_ingest.py        ← stage 1–2 (slow, run once)
├── run_experiments.py   ← stages 3–6 + all experiments
├── make_figures.py      ← every figure and table
├── data/                ← cached features (Parquet)
├── models/              ← fitted artefacts
└── outputs/
    ├── figures/         ← PNG 400 dpi + vector PDF
    ├── tables/          ← UTF-8 markdown + CSV
    └── REPORT.md        ← full write-up with results
```

### Figures for the paper

1. Pipeline architecture
2. Spectral signature — why 9 bands beat 1
3. **Index ablation — MSAVI-only vs multispectral** *(the headline)*
4. Model comparison including naive baselines
5. `n_neighbors` sensitivity
6. Threshold operating curve
7. Persistence comparison
8. Detection limits heatmap
9. Anomaly-type robustness
10. Score separability
11. PR + ROC with operating point
12. Confusion matrix
13. Held-out generalisation
14. Detection walkthrough on one field
15. Spatial stress map

---

## 6. Honest expectations

Multispectral features, better cloud screening and more dates should raise the numbers,
and the gradual-onset case has the most room to improve.

Working the other way: a held-out split and thresholds calibrated on clean data only are
**stricter** than the current setup. Some metrics may come in lower than the existing
0.83 / 0.63 for that reason alone.

If that happens, the lower number is the more trustworthy one and is what belongs in the
paper. A result that survives a held-out split is worth more than a higher number that
never faced one. Both will be reported, with the difference explained.

---

## 7. Order of work

1. Scaffold + config + indices, validated on a handful of tiles
2. Parallel ingest of all 3,328 tiles (~7 min), cached
3. Features + split
4. Detectors + baselines + decision logic
5. Experiments E1–E8
6. Figures + tables + REPORT.md
