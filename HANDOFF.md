# PROJECT HANDOFF — Unsupervised Crop Stress Detection (PBL Project)

**Last updated:** 2026-08-27 — the four code folders were merged into
`crop_stress/`; this document was rewritten around the new layout.

**Purpose:** single source of truth for picking this project up in a fresh chat
session. Paste the Bootstrap Prompt (§13) into a new chat, or just point the
assistant at this file.

---

## 1. The result, in one box

**Write the paper from `crop_stress/outputs/study/REPORT.md`.**

LOF · extreme-value (GPD) threshold at a 3% false-alarm risk · strict
2-consecutive persistence, measured on **832 held-out fields**:

| Precision | Recall | F1 | PR-AUC | False alarms |
|---|---|---|---|---|
| **0.876** | **0.814** | **0.844** | 0.904 | 350 / 20,800 |

(40% severity over 2 observations, mean of 5 seeds.)

**The central finding, which has survived every re-measurement:** the project was
built around a hybrid ensemble, and the ensemble is *harmful*. Standalone LOF
reaches F1 **0.844**; the three-model hybrid reaches **0.354**. The failure is in
**recall** — 0.814 → 0.230 — not precision. Averaging a strong ranking with weak
ones does not make the alarms wrong, it stops them firing.

---

## 2. Layout

```
PBL Project/
├── crop_stress/          ⭐ ALL THE CODE AND RESULTS
│   ├── study/                ⭐ work here
│   ├── engine/               shared config, utils, models, the feature table
│   ├── appendix/             three experiments that exist nowhere else
│   ├── multispectral/        the independent 4-index study
│   ├── outputs/              ⭐ ONE outputs tree
│   │   ├── study/                16 figures, 9 tables, REPORT.md, results.json, PROVENANCE.md
│   │   ├── appendix/             7 figures, 6 tables, REPORT.md
│   │   └── multispectral/        14 figures, 9 tables, REPORT.md
│   └── cache/                regenerable; safe to delete
│
├── paper/                📄 the IEEE draft (§7)
├── dataset_clean_2/      ⛔ MSAVI rasters — never delete
├── archive (1)/          ⛔ raw 9-band archive — never delete
└── old/                  🗄️ superseded work; nothing uses it
    ├── hybrid_pipeline_experiments2/     the old "Study 2"
    ├── pre_merge_originals/              the four folders as they were before the merge
    └── gen1_prototype/, gen2_original/
```

**`crop_stress/` was merged on 2026-08-27** from `hybrid_pipeline_experiments`,
`hybrid_pipeline_experiments3`, `experiment3` and `experiment4`. Five filenames
collided across them (`config.py`, `run_experiments.py`, `make_figures.py`,
`make_report.py`, `run_all.py`), so they are separate packages under one roof
rather than a flat merge. `crop_stress/README.md` lists the five paths that were
patched. The originals are in `old/pre_merge_originals/` (2.8 GB) as a safety net
and can be deleted once you are confident.

**What each folder is for:**

- **`study/`** — the current study, and the source for the paper.
- **`engine/`** — not a study. `config.py`, `utils/`, fitted `models/`, and
  `outputs/features_timeseries.csv` (83,200 rows) which every study reads.
  `main.py` rebuilds that table from rasters — slow, rarely needed.
- **`appendix/`** — three experiments found nowhere else, one of which backs a
  claim the paper makes. See §8.
- **`multispectral/`** — independent rebuild sharing only the raw data. Sole
  source of two findings the paper cites. See §8.

---

## 3. Environment

The `.venv` at the repo root is **empty**. Use the global Python 3.14:

```
C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe
```

Installed: torch 2.11.0+cpu, scikit-learn 1.8.0, rasterio 1.5.0, numpy 2.3.5,
pandas 2.3.3, matplotlib 3.10.8, scipy 1.17.1, joblib 1.5.2, pypdf 6.16.2.

**Commands.** Rebuild the whole study — 9 tables, 16 figures, `REPORT.md`,
`results.json`. ~35 min the first time; `cache/` makes later runs far quicker:

```bash
cd crop_stress/study && python run_all.py
```

Figures only, from the stored results (seconds):

```bash
cd crop_stress/study && python make_figures.py
```

Rebuild the feature table from rasters — **rarely needed**, all 3,328 tiles:

```bash
cd crop_stress/engine && python main.py
```

> `study/` reads the feature CSV but **refits** all three detectors on its own
> `fit` split. It does not load the `.pkl`/`.pt` files in `engine/models/`,
> because those were fitted on every field including the ones it evaluates on.

---

## 4. Data and method

**Data.** Sentinel-2 over 3,328 delineated parcels in Lombardy, Italy. The raw
archive is 9-band (B2–B8, B11, B12) for 2016–2019; the study uses a derived
single-band MSAVI product for 2019.

**Screening.** Acquisitions are rejected on four criteria — minimum intensity
standard deviation, minimum valid-pixel ratio 0.7, minimum unique-value ratio
0.05, and minimum spatial gradient. Then 25 passes per parcel are kept, at least
10 days apart. That gives 3,328 × 25 = **83,200 parcel-observations**.

**Features (16).** Per pass: mask nodata, clip to the 1st/99th percentile, then
`mean, std, min, max, valid_pixel_ratio, p25, p50, p75, cv`. Per parcel over
time: `smoothed_p50` (Savitzky–Golay), `iqr`, `diff_from_global_median`,
`rolling_median`, `rolling_std`, `diff`, `percent_change`.

**Detectors.** Isolation Forest (100 trees), LOF (novelty mode, k=10),
autoencoder (16→32→4→32→16, dropout 0.2, 200 epochs), and a hybrid that
min–max normalises the three scores and sums them.

**Decision rule.** Per-field threshold, then an alarm only when a field exceeds
it on N *consecutive* passes. Both defend against cloud artefacts: a global
cut-off would spam alarms on naturally volatile fields, and a single-pass
excursion is indistinguishable from a shadow.

**Why MSAVI, not NDVI.** Early in the season the canopy is sparse and bare soil
dominates the pixel. MSAVI carries a soil-adjustment term, so it tracks living
canopy rather than dirt.

**Two inert hyperparameters.** A PCA at 95% variance is fitted but consumed by
nothing — **do not claim PCA as part of the method**. The `contamination`
parameter has no effect either: it shifts a decision offset that this scoring
path never uses, and a percentile ranking is invariant to it. That is an
advantage, not an oversight — the true contamination rate is unknown in practice
and the method does not need it.

---

## 5. The protocol — why the numbers are trustworthy

Fields are split three ways, **disjointly**:

| Split | Fields | Role |
|---|---|---|
| `fit` | 1,664 | detectors fitted here, clean rows only |
| `cal` | 832 | threshold rule and *k* chosen here |
| `test` | 832 | every reported number; nothing is ever selected on it |

**Every threshold is calibrated on clean, pre-injection scores.** This is the
single most consequential design decision in the project.

**Why the headline went UP from the old 0.718.** The previous study computed each
field's percentile on the batch that *already contained* the injections. At a 15%
injection rate the top 15% of scores **are** the anomalies, so a P85 cut lands
exactly on their boundary and gets dragged up into the group it is meant to
separate. Roughly half the injected observations end up below their own field's
threshold. That capped recall near 0.6 regardless of which detector was used.
Calibrating on clean scores removes the cap and allows a far stricter cut that
still recovers more.

**Evaluation.** No ground truth exists, so stress is injected synthetically —
sudden drop, gradual onset, or spectral shift — into 15% of observations, and
recovery is measured. Baseline scenario: 40% severity over 2 observations. Five
seeds: 42, 123, 456, 789, 1024.

---

## 6. Results summary

All from `crop_stress/outputs/study/`. Full tables in `REPORT.md`.

**Detectors** — LOF 0.844 · Autoencoder 0.528 · Hybrid 0.354 · ISO+LOF 0.354 ·
Isolation Forest 0.179.

**Against hand-written rules** — LOF 0.844 vs 0.243 for the best rule (per-field
z-score), a 3.5× margin.

**Threshold rule** — adaptive P85 post-injection 0.693 · global P90
post-injection 0.679 (but precision 0.979 with only 34 false alarms) ·
clean-calibrated P97.25 0.846 · **EVT q=0.03 → 0.844** (selected on `cal`;
ξ = −0.199).

**Detection limits (F1)**

| Duration | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|
| 1 obs | — | — | 0.365 | 0.452 | 0.500 |
| 2 obs | 0.077 | 0.482 | 0.755 | 0.844 | 0.859 |
| 3 obs | — | — | 0.786 | 0.877 | 0.898 |

**Signatures** — sudden drop 0.844 · spectral shift 0.809 · **gradual onset
0.614** (the weak case: a slow decline is partly absorbed into the rolling
baseline it is compared against).

**Persistence** — none 0.809 · **strict 2-consecutive 0.844** · sliding 2-of-3
0.530 · strict 3-consecutive 0.465. Read the last with care: it is measured at
2-observation events, so it cannot fire on the event alone.

**Ablations** — `k` spans 0.805–0.844 across k ∈ [5,75] and **k\* = 10**, the
original default. Adding trajectory features (lagged value, cumulative change,
acceleration) *hurts*: 0.844 → 0.785.

**Protocol cost** — fitting on a disjoint half instead of all fields costs
**0.021 F1** (0.714 → 0.693).

**Threshold-free** — average precision 0.900, ROC-AUC 0.974. Per-field
thresholds span a 4.0-fold range, which is what justifies calibrating per field.

**Significance** — the headline gain over the old rule is ΔF1 **+0.1505**, 95% CI
[+0.1416, +0.1590], paired permutation p < 5×10⁻⁴ over 4,160 field-seed units.

---

## 7. The paper — `paper/`

`main.tex` + `refs.bib` (26 entries) + `figures/`. Upload the folder to Overleaf
and compile with pdfLaTeX; `IEEEtran.cls` ships there. **There is no LaTeX on
this machine.** 7 figures, 7 tables, ~8.6 pages. Every number was machine-checked
against `results.json` — zero mismatches. `paper/README.md` has the build
instructions, an ordered trim list and the pre-submission checklist.

**⚠️ A float fix has not been recompiled.** The first Overleaf compile stranded
two figures inside the bibliography on pages 7–8. Three fixes were applied and
not yet verified: both late floats moved to the start of Section V,
`worked_example.pdf` reshaped from 7.2×6.6 in to 9.6×4.5 in (it was taking 54% of
a page as a full-width float, which made it unplaceable), and `stfloats` added.
**If Overleaf errors on `stfloats`, delete that line** — the other two fixes
should be enough alone.

**Before submitting:** fill in the author block, spot-check the bibliography page
numbers, and keep the two deliberately unflattering paragraphs (§V-D and §VII).

---

## 8. What lives only in `appendix/` and `multispectral/`

**`appendix/`** — three experiments found nowhere else:

- **Event-level (range-based) metrics.** Tried; changed recall by +0.002, because
  events last 2 observations and persistence requires 2 consecutive, so detection
  is all-or-nothing and there is no partial credit to award. This is the answer if
  a reviewer asks why point-wise F1 was used.
- **The same-date peer-deviation feature.** A documented failure: cost ~0.03 F1
  and raised false alarms, probably because injected fields pollute their own date
  median at a 15% injection rate.
- **Adaptive P85 vs global P90.** ΔF1 +0.0144, 95% CI [+0.0022, +0.0265],
  p = 0.018. The study's significance test answers a different question, so this
  is the only source for that comparison.

**`multispectral/`** — two findings the paper cites and nothing else can produce:

- **Four vegetation indices beat one.** F1 0.243 → 0.351 (+44% relative,
  permutation p ≈ 0.065). NDMI correlates only 0.81–0.85 with the greenness
  indices, so it carries independent signal. This is also the most promising
  attack on the gradual-onset weakness.
- **Crop rotation breaks cross-year monitoring.** Using a field's previous season
  as its baseline scores *worse* (F1 0.287) than using other fields in the current
  season (0.352), because a field under maize one year and wheat the next has a
  different normal.

Its **protocol claim is withdrawn** — see §9.

---

## 9. Corrections — do not repeat these

Three claims made earlier in this project turned out to be wrong. They are fixed
in the current study and the current paper, but they still appear in files under
`old/`, so anyone reading those will meet them again.

1. **The sliding 2-of-3 persistence rule is causal.** The old audit said it peeks
   at `j+1`. It does not — `engine/utils/hybrid.py` slices `[j-2, j-1, j]`,
   strictly backward. The rule still loses on F1, but on merit.
2. **Held-out evaluation costs 0.021 F1, not ~0.37.** The large figure came from
   comparing two separate codebases that also differed in input representation,
   cloud screening, observation density and injection procedure. Measured with
   only the protocol changing, the cost is small.
3. **`multispectral/`'s protocol claim is withdrawn.** Its multispectral and
   crop-rotation findings still stand; its explanation of the 0.72 → 0.35 gap
   does not.

---

## 10. Honest limitations

These are in the paper, and they should stay there.

- **No ground truth.** Every number rests on synthetic injection.
- **The injection is generous.** It scales six of the sixteen feature columns by
  (1−severity) but leaves `std` and `cv` untouched — describing a field that got
  darker without becoming less variable, which real pixels cannot do. Absolute
  figures are an **upper bound**; the comparisons *between* configurations are
  the reliable part. `paper/main.tex` §VII says this.
- **The recommended operating point is not uniformly better.** It wins on 9 of 11
  severity/duration cells but *loses* on the two weakest — F1 0.077 at 10%
  severity against 0.233 for the rule it replaces. That belongs in the paper as a
  stated trade-off: the risk *q* is a dial set from the cost of a missed event
  against the cost of a wasted field visit.
- **Clean calibration assumes a stress-free reference period** per field.
  Available retrospectively; an assumption for a live system, and the obvious
  source (last season) is broken by crop rotation.
- **One region, one season.** The test split is 832 fields, so these numbers carry
  more sampling noise than a whole-dataset figure.
- **Field-level only.** One score per parcel per pass; no within-field
  localisation.
- **Not under version control.** No git history, so deletions are permanent.
  Consider `git init`.

---

## 11. Known stale files

- **`Interview_QA.pdf` and `INTERVIEW_PREP.html`** teach the old F1 0.718 story
  and the pre-merge narrative. **Not regenerated. Do not revise from them.**
- **Everything under `old/`** is superseded. `old/README.md` explains what is in
  there and flags the two things in it that are actually wrong.
- **`crop_stress/engine/Final_Defense_Guide.txt`** carries a SUPERSEDED banner
  and argues a conclusion the project's own data refutes.

---

## 12. Vocabulary

- **Tile / field / parcel** — one agricultural plot. 3,328 of them.
- **Observation / pass** — one satellite overflight of one field. 25 per field.
- **MSAVI** — Modified Soil Adjusted Vegetation Index. Soil-adjusted, so it
  tracks canopy rather than bare earth early in the season.
- **Injection** — synthetic corruption of clean data to create a labelled test.
- **Severity** — % reduction in the vegetation index (10–50%).
- **Duration** — how many consecutive passes the injected stress lasts (1–3).
- **Persistence** — the rule requiring N consecutive flags before an alarm.
- **EVT / GPD** — extreme value theory; a Generalised Pareto tail fitted to each
  field's clean scores and inverted for a target false-alarm risk *q*.
- **False alarms** — raw false-positive count, out of 20,800 test observations.

---

## 13. Bootstrap Prompt for a New Chat

```
I'm continuing work on my PBL project at:
C:\Users\prakh\OneDrive\Desktop\Projects\PBL Project

Read HANDOFF.md in that directory first — it's a full handoff document
covering the project, environment, results and limitations.

Five things to know up front:
- ALL the code and results are in crop_stress/. Work in crop_stress/study/.
  Results are in crop_stress/outputs/study/REPORT.md.
- crop_stress/engine/ is shared machinery, not a study — config, utils,
  and the 83,200-row feature table everything reads. Don't delete it,
  and don't move dataset_clean_2/ or "archive (1)/".
- Headline: LOF + EVT threshold + strict 2-consecutive →
  P 0.876 / R 0.814 / F1 0.844 on 832 held-out fields.
- STALE, do not quote: anything under old/, plus Interview_QA.pdf and
  INTERVIEW_PREP.html. They teach the old F1 0.718 story.
- The .venv is empty. Use the global Python:
  C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe

Then help me with: <your task>
```
