# experiment3

A clean-room rebuild of the crop stress detector. **No code is shared with
`hybrid_pipeline*`** — only the dataset. Everything the paper needs is produced
inside this folder.

See [PLAN.md](PLAN.md) for the design rationale.

---

## Running it

The project `.venv` is empty; use the global Python 3.14.

```bash
cd experiment3
python run_ingest.py          # ~9 min, once — raw rasters -> data/observations.csv
python run_experiments.py     # all experiments -> outputs/tables/
python make_figures.py        # all figures -> outputs/figures/
```

`run_ingest.py --limit 20` gives a fast smoke test.
`run_experiments.py --quick` uses fewer seeds.

## What is different from the earlier pipeline

### It reads the raw multispectral archive, not the derived MSAVI product

`dataset_clean_2/` contains only single-band `_MSAVI.tif`. The raw archive at
`archive (1)/lombardia/data2019/` is **9-band** — B2, B3, B4, B5, B6, B7, B8,
B11, B12 — which the earlier study's audit had incorrectly recorded as
unavailable.

That gives four indices instead of one:

| Index | Senses |
|---|---|
| MSAVI | canopy greenness, soil-corrected |
| NDVI | canopy greenness |
| NDRE | chlorophyll — reacts before NDVI does |
| NDMI | **canopy water content** |

NDMI correlates only 0.81–0.85 with the greenness indices, so it carries
genuinely independent information.

### More of the season survives

71 raw acquisitions exist per field. The earlier pipeline kept 25; this one
keeps up to 40 after screening — **132,568 observations across 3,317 fields**,
against 83,200 before.

### Cloud screening is spectral, not a blur heuristic

Cloud is detected as *bright in blue **and** bright in SWIR **and** no
vegetation signal* — a combination canopy never produces. The earlier pipeline
inferred cloud from image blurriness, which cannot separate a cloud from a
genuinely smooth field.

### The evaluation is stricter

| | earlier pipeline | experiment3 |
|---|---|---|
| Model fitting | all 3,328 fields | 60% of fields; evaluated on the other 40% |
| Threshold source | the batch being scored, injections included | each field's **clean** record |
| Persistence | included a rule that looked at the *next* observation | causal only |
| Baselines | none | three rule-based, evaluated alongside |
| `n_neighbors` | fixed at 10, never swept | swept |
| Significance | none | paired permutation tests, bootstrap CIs |
| Dead parameters | PCA fitted but unused; `contamination` inert | neither present |

## Results

Held-out fields, 40% severity over 2 observations, 5 seeds:

| | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Multispectral (4 indices)** | **0.32** | **0.40** | **0.35** | **0.78** |
| MSAVI only (1 index) | 0.28 | 0.22 | 0.24 | 0.68 |

Multispectral input raises F1 by **+0.107 (44% relative)**, permutation p = 0.065.
That is the headline finding and it is what reading the raw archive buys.

### These numbers are lower than the earlier study's 0.83 / 0.63, and that is the point

The same data, evaluated three ways:

| Protocol | ROC-AUC | F1 |
|---|---|---|
| Same fields, same season (**earlier study**) | 0.99 | 0.72 |
| Unseen fields, same season (**here**) | 0.78 | 0.35 |
| Previous season as baseline | 0.72 | 0.29 |

The earlier pipeline fits LOF in novelty mode on the very rows it later scores.
That memorises the manifold, so injected points sit far outside it and separate
almost perfectly — and the threshold was calibrated on the batch containing the
injections. Most of that 0.99 is the protocol, not the model.

**Crop rotation breaks the cross-year protocol.** A field growing maize one year
and wheat the next has a completely different normal, so last season's baseline
is actively misleading. That is why cross-field, not cross-year, is the primary
protocol here despite cross-year being closer to deployment in principle.

## Layout

```
experiment3/
├── config.py              every parameter, single source
├── src/
│   ├── indices.py         band maths + spectral cloud screening
│   ├── ingest.py          raw rasters -> observations (parallel)
│   ├── features.py        temporal + contextual feature engineering
│   ├── detectors.py       LOF / IForest / autoencoder / baselines
│   ├── decision.py        thresholding + causal persistence
│   ├── evaluation.py      injection, metrics, bootstrap, permutation tests
│   ├── pipeline.py        one leak-free evaluation run
│   └── viz.py             figure style
├── run_ingest.py
├── run_experiments.py
├── make_figures.py
├── data/observations.csv  cached (regenerate with run_ingest.py)
└── outputs/
    ├── tables/            .md + .csv per experiment
    ├── figures/           PNG 400 dpi + vector PDF
    ├── final_scored.csv   held-out predictions, for the figures
    └── results.json       headline numbers
```

## Design decisions worth defending

**Injection is coherent across indices.** A stress event lowers greenness *and*
moisture together, in the proportions the biology implies. Perturbing one
channel alone would let a multispectral model win for the wrong reason.

**Ground truth travels as a DataFrame column,** never a parallel array. Feature
building regroups rows by field, so an index-aligned array would silently
desynchronise.

**Thresholds are calibrated on each field's clean record** — a stand-in for the
previous season a deployed system would have. Calibrating on an earlier *window
of the same season* was tried first and does not work: early and late season are
different phenological regimes, so a boundary learned in March is the wrong
boundary for August.
