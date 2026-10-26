# experiment4 — a better operating point, without changing the model

Six experiments that improve what the existing detector *reports*, without
retraining anything from rasters and without touching a single file outside this
folder.

The detector is unchanged: LOF, `n_neighbors=10`, the same 16 features, the same
83,200-row feature table, the same injector, the same strict 2-consecutive
persistence rule. What changes is **where the decision threshold comes from**,
**how results are scored**, and **what gets verified rather than assumed**.

## Run it

```bash
cd experiment4 && C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe run_all.py
```

First run takes a couple of minutes (it injects five seeds and caches them);
later runs are much faster. To re-render figures without re-running the
experiments:

```bash
cd experiment4 && C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe make_figures.py
```

## What it reads, and what it writes

**Reads, never writes:**

- `hybrid_pipeline_experiments/outputs/features_timeseries.csv` — the feature table
- `hybrid_pipeline_experiments/config.py`, `utils/preparation.py`, `utils/evaluation.py`

**Writes, only inside `experiment4/`:** `cache/` (injected draws), `outputs/`.

It does **not** load the fitted models in `hybrid_pipeline_experiments/models/`
— it refits LOF on its own `fit` split, because the stored model was fitted on
every field including the ones this study evaluates on.

## The three-way split — the point of the whole thing

| Split | Fields | Used for |
|---|---|---|
| `fit` | 50% | Fitting the detector, on clean rows only |
| `cal` | 25% | Choosing the threshold percentile and *k* |
| `test` | 25% | Every reported number. Nothing is selected on it. |

Study 2 has only the equivalent of `fit`, and reports on it. It also computes
its per-field percentile on the batch that already contains the injections,
which lets the cut-off adapt to the anomaly rate it is being measured against.

Every threshold here is computed from **clean, pre-injection scores**. That is
what makes E1's improvement a result rather than a tuning artifact.

## The experiments

| | Question | File |
|---|---|---|
| **E1** | If the threshold is calibrated on clean data and the percentile chosen honestly, where does the operating point land? | `experiments.py:e1_threshold` |
| **E2** | Does event-level scoring change the picture versus observation-wise? | `:e2_range_metrics` |
| **E3** | Does a same-date cross-field peer feature reduce false alarms? | `:e3_peer_feature` |
| **E4** | Does LOF `n_neighbors` matter? (Study 2 never swept it) | `:e4_kneighbors` |
| **E5** | Is adaptive P85 actually distinguishable from global P90? | `:e5_significance` |
| **E6** | Can an extreme-value tail beat a percentile that saturates at 25 samples? | `:e6_evt` |

## Layout

```
experiment4/
├── common.py         data, splits, detector, decision rule, metrics
├── experiments.py    E1–E6
├── make_figures.py   7 figures, PNG 400dpi + vector PDF
├── make_report.py    outputs/REPORT.md
├── run_all.py        orchestrator
├── cache/            injected draws (regenerable; safe to delete)
└── outputs/
    ├── REPORT.md     ← read this
    ├── results.json  every number, machine-readable
    ├── tables/       t1…t6 as markdown
    └── figures/      fig1…fig7
```

`REPORT.md` and all figures are generated from `results.json`, so the prose,
the tables and the plots cannot drift apart.

## Honesty notes

These are the things a reviewer should check, stated up front:

- **Nothing is selected on the test split.** The threshold percentile (E1) and
  *k* (E4) are chosen on `cal`. Test curves appear in figures for reference and
  are never argmaxed.
- **Clean calibration assumes a stress-free reference period per field.** That
  is available in a retrospective seasonal analysis. For a real-time system it
  is an assumption, and `experiment3` showed that cross-year reference periods
  are broken by crop rotation.
- **Point-wise and event-level metrics are both reported everywhere.** Quoting
  only the event-level figure would flatter the result.
- **Synthetic injection is still the only ground truth**, with every caveat the
  main study records.
- The test split is a quarter of the data, so these numbers carry more sampling
  noise than Study 2's full-dataset figures.

## Relationship to the rest of the project

- `hybrid_pipeline_experiments2/` remains the submitted study. This folder does
  not replace it; it proposes a better operating point for it.
- `hybrid_pipeline_experiments2/protocol_comparison/` (the controlled protocol
  test) established that held-out fields cost ~0.02 F1, not the ~0.37 that
  `experiment3` was read as implying. E1 here builds directly on that split.
- `experiment3/` remains the independent multispectral and cross-year study.
