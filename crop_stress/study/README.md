# hybrid_pipeline_experiments3 — the unified study

**This is the folder to write the paper from.** `outputs/REPORT.md` is the
single source: every figure, table and sentence in it comes from one
`results.json`, so nothing can drift.

## Why this folder exists

Three earlier pieces of work each measured something real, but each at a
different operating point on a different subset of fields. No two of their
tables could be placed side by side, which is why the results read as a
patchwork rather than a study.

| Predecessor | What it established | Status |
|---|---|---|
| `hybrid_pipeline_experiments2/` | Ensembling is harmful; detection limits; persistence; signatures | **Re-measured here** under one protocol |
| `.../protocol_comparison/` | Held-out fields cost little | **Folded in** as section 7 |
| `experiment4/` | Clean-calibrated and EVT thresholds; `k` sweep; significance | **Folded in** as section 1 and 8 |
| `experiment3/` | Multispectral +44%; crop rotation breaks cross-year | **Still cited** — not superseded |

Nothing in those folders was modified. This one re-runs their questions on a
common footing.

## Run it

```bash
cd hybrid_pipeline_experiments3 && C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe run_all.py
```

First run is roughly 30–45 minutes: it injects the full severity × duration
grid plus two alternative signatures and trains the autoencoder. Later runs
reuse `cache/` and are far quicker. Figures alone:

```bash
cd hybrid_pipeline_experiments3 && C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe make_figures.py
```

## The protocol

| Split | Share | Role |
|---|---|---|
| `fit` | 50% | detectors fitted here, clean rows only |
| `cal` | 25% | threshold rule and hyperparameters chosen here |
| `test` | 25% | every reported number; nothing is ever selected on it |

Every threshold is calibrated on **clean, pre-injection scores**.

Study 2 computed its percentile on the batch that already contained the
injections. At a 15% injection rate that is a specific trap: the top 15% of
scores *are* the anomalies, so a P85 cut lands on their boundary and gets
dragged up into the group it is meant to separate. About half the injected
observations end up below their own field's threshold — which is why Study 2's
recall could not exceed roughly 0.6 no matter which detector was used.

## What it reads

Read-only, never written to:

- `hybrid_pipeline_experiments/outputs/features_timeseries.csv`
- `hybrid_pipeline_experiments/config.py`, `utils/`

It does **not** load the fitted models in `hybrid_pipeline_experiments/models/`,
because those were fitted on every field including the ones evaluated here. All
three detectors are refitted on the `fit` split.

## Sections

| | Question |
|---|---|
| 1 | Where should the threshold sit? |
| 2 | Which detector — and is the ensemble worth it? |
| 3 | Does it beat rules written by hand? |
| 4 | How many consecutive passes before an alarm? |
| 5 | How weak or brief an event can it see? |
| 6 | Does the shape of the stress matter? |
| 7 | What does held-out evaluation actually cost? |
| 8 | Feature set and LOF neighbourhood size |

## Layout

```
hybrid_pipeline_experiments3/
├── core.py           protocol, data, detectors, decision rules, metrics
├── study.py          sections 1–8
├── make_figures.py   9 figures, 400 dpi PNG + vector PDF
├── make_report.py    outputs/REPORT.md
├── run_all.py        orchestrator
├── cache/            injected draws (regenerable; safe to delete)
└── outputs/
    ├── REPORT.md     ← write the paper from this
    ├── results.json
    ├── tables/       t1…t9
    └── figures/      fig1…fig9
```

## Two corrections this study makes

1. **The sliding 2-of-3 persistence rule is causal.** Study 2's audit recorded
   it as non-causal, claiming it inspects observation *t+1*. The implementation
   in `utils/hybrid.py` slices `[t-2, t-1, t]` — strictly backward. The rule may
   still lose on F1, but it loses on merit.
2. **Held-out evaluation is not what caused `experiment3`'s low numbers.**
   That comparison ran across two codebases differing in input representation,
   cloud screening, observation density and injection procedure. Isolated
   properly, the protocol cost is small (section 7).

## Honesty notes

- Nothing is selected on `test`. The threshold rule and `k` are chosen on `cal`.
- Each detector gets its own clean-data threshold in section 2, because the
  three score scales are not comparable; a shared threshold would have made the
  comparison meaningless.
- Clean calibration assumes a stress-free reference period per field —
  available retrospectively, an assumption in real time.
- Part of the section 1 gain is a fix for an evaluation-design flaw specific to
  a 15% injection rate, not a pure detector improvement. Stated in the report.
- Synthetic injection is still the only ground truth.
