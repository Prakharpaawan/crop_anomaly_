# Crop Stress Detection from Sentinel-2 Imagery

Unsupervised anomaly detection that flags crop stress at field level from free
optical satellite time series, with no labelled training data.

**3,328 fields · 25 satellite passes each · 83,200 observations · Lombardy, 2019**

Current configuration: **LOF** + an extreme-value (GPD) threshold at a 3%
false-alarm risk + strict 2-consecutive persistence, measured on **832 held-out
fields** → precision **0.876**, recall **0.814**, F1 **0.844**, PR-AUC **0.904**
(40% severity over 2 observations).

---

## Where to start

| If you want to… | Read |
|---|---|
| Pick up the project in a new session | **`HANDOFF.md`** — the source of truth |
| See the current results | **`crop_stress/outputs/study/REPORT.md`** |
| Understand the folder layout | `crop_stress/README.md` |
| Work on the paper | `paper/` — read `paper/README.md` first |
| Know what came from where | `crop_stress/outputs/study/PROVENANCE.md` |

---

## Layout

```
PBL Project/
├── README.md                    ← you are here
├── HANDOFF.md                   ← full project state; read this first
│
├── crop_stress/                 ⭐ ALL THE CODE AND RESULTS
│   ├── study/                       ⭐ work here
│   ├── engine/                      shared config, utils, models, the feature table
│   ├── appendix/                    three experiments that exist nowhere else
│   ├── multispectral/               the independent 4-index study
│   ├── outputs/                     ⭐ one outputs tree
│   │   ├── study/                       16 figures, 9 tables, REPORT.md
│   │   ├── appendix/                    7 figures, 6 tables, REPORT.md
│   │   └── multispectral/               14 figures, 9 tables, REPORT.md
│   └── cache/                       regenerable; safe to delete
│
├── paper/                       📄 IEEE draft
├── dataset_clean_2/             ⛔ MSAVI rasters — never delete
├── archive (1)/                 ⛔ raw 9-band archive — never delete
└── old/                         🗄️ superseded work; nothing uses it
```

## Running it

The `.venv` at the repo root is **empty**. Use the global Python 3.14:

```bash
C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe
```

The command you usually want — rebuilds 9 tables, 16 figures, `REPORT.md` and
`results.json`. ~35 minutes the first time, much faster afterwards:

```bash
cd crop_stress/study && python run_all.py
```

Re-render the figures only, from the stored results — seconds:

```bash
cd crop_stress/study && python make_figures.py
```

## ⚠️ The one thing that will break this project

`crop_stress/study/` has **no data and no pipeline code of its own**. It imports
`config.py` and `utils/` from `crop_stress/engine/`, reads the 83,200-row feature
table from `crop_stress/engine/outputs/`, and reads the rasters from
`dataset_clean_2/`.

**Do not delete `crop_stress/engine/`, `dataset_clean_2/` or `archive (1)/`.**
`engine/` is shared machinery, not an old study.

## The protocol, in one paragraph

Fields are split three ways, disjointly: **1,664 fit** (detectors fitted here on
clean rows only), **832 calibration** (the threshold rule and *k* chosen here),
**832 test** (every reported number, and nothing is ever selected on it). Every
threshold is calibrated on clean, pre-injection scores. That last point is what
raised the headline from the earlier 0.718 — computing a percentile on the batch
that already contains the injections caps recall near 0.6 regardless of detector.

## Known state

- **The central finding survives every re-measurement.** The project was built
  around a hybrid ensemble; it is *harmful*. LOF scores F1 **0.844** against
  **0.354** for the hybrid, and the failure is in recall (0.814 → 0.230), not
  precision — averaging a strong ranking with weak ones stops the alarms firing.
- **Three earlier claims were corrected.** The sliding 2-of-3 rule is *causal*
  (the old audit said otherwise); held-out evaluation costs only 0.021 F1; and
  the multispectral study's protocol claim is withdrawn.
- **The recommended operating point is not uniformly better.** It beats its
  predecessor on 9 of 11 severity/duration cells but loses on the weakest two —
  F1 0.077 at 10% severity. That belongs in the paper as a trade-off, not a
  footnote.
- **Absolute numbers are an upper bound.** Stress is simulated by scaling six of
  the sixteen feature columns, so a simulated event is a large coherent
  displacement in the detector's own feature space. The comparisons between
  configurations are the reliable part. `paper/main.tex` §VII says this.
- **`Interview_QA.pdf` and `INTERVIEW_PREP.html` are out of date** — they teach
  the old F1 0.718 story and have not been regenerated.
- **Not under version control.** No git history, so deletions are permanent.
  Consider `git init`.

Open items and limitations: `HANDOFF.md` §10 and `crop_stress/README.md`.
