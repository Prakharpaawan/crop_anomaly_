# crop_stress — the whole project in one folder

Merged on 2026-08-27 from four separate folders (`hybrid_pipeline_experiments`,
`hybrid_pipeline_experiments3`, `experiment3`, `experiment4`). The originals are
in `old/pre_merge_originals/` if anything needs checking against them.

## Layout

```
crop_stress/
├── engine/          the shared machinery — config, utils, fitted models, the feature table
├── study/           ⭐ THE STUDY — this is where you work
├── appendix/        three experiments that exist nowhere else
├── multispectral/   the independent 4-index study
├── outputs/         ⭐ ONE outputs tree, one subfolder per study
│   ├── study/           16 figures, 9 tables, REPORT.md, results.json, PROVENANCE.md
│   ├── appendix/        7 figures, 6 tables, REPORT.md
│   └── multispectral/   14 figures, 9 tables, REPORT.md
└── cache/           regenerable intermediate data; safe to delete
```

## Run it

```bash
cd crop_stress/study && python run_all.py
```

Rebuilds everything in `outputs/study/`: 9 tables, 16 figures, `REPORT.md`,
`results.json`. ~35 minutes the first time; `cache/` makes later runs far
quicker. Figures only, from the stored results (seconds):

```bash
cd crop_stress/study && python make_figures.py
```

The `.venv` at the repo root is **empty**. Use the global Python 3.14:
`C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe`

## What each folder is

**`engine/`** — Not a study. It holds `config.py` (every tunable parameter),
`utils/` (cleaning, feature extraction, temporal features, the three injectors,
model definitions), the fitted `models/`, and `outputs/features_timeseries.csv`
— the 83,200-row table every study reads. **Nothing here should be deleted.**
`main.py` rebuilds that table from the rasters; it is slow and rarely needed.

**`study/`** ⭐ — The current study, and the source for the paper. One protocol,
one operating point, one split. Read `outputs/study/REPORT.md`.

**`appendix/`** — Mostly folded into `study/`, but three experiments live only
here and one of them backs a claim the paper makes: event-level (range-based)
metrics, the same-date peer-deviation feature, and the adaptive-P85 vs
global-P90 significance test.

**`multispectral/`** — An independent rebuild sharing only the raw data. Sole
source of two findings the paper cites: deriving four vegetation indices instead
of one raises F1 by ~44%, and cross-year baselines fail because crop rotation
gives a field a different normal each season. Its *protocol* claim is withdrawn —
`study/` section 7 supersedes it.

## What still lives outside this folder

- `dataset_clean_2/` — the MSAVI rasters. Read by `study/extras.py` for the
  imagery figure and by `engine/main.py`.
- `archive (1)/` — the raw 9-band archive. Read by `multispectral/`.
- `paper/` — the LaTeX draft.

**Do not move those.** Paths are discovered by walking up from this folder until
the data directories are found, so nesting depth is flexible — but the data has
to be findable above `crop_stress/`.

## What changed in the merge

Five filenames collided across the four originals (`config.py`,
`run_experiments.py`, `make_figures.py`, `make_report.py`, `run_all.py`), so the
folders are kept as separate packages rather than flattened. Four path
assumptions were patched, because every one of them assumed its folder sat one
level below the repo root and it now sits two:

| File | Was | Now |
|---|---|---|
| `engine/config.py` | `PARENT_DIR = dirname(here)` | walks up to the folder holding `dataset_clean_2` |
| `study/core.py` | `../hybrid_pipeline_experiments` | `../engine`, plus `PKG` and `PROJECT` roots |
| `study/extras.py` | `dirname(BASE_DIR)/dataset_clean_2` | `PROJECT/dataset_clean_2` |
| `appendix/common.py` | `../hybrid_pipeline_experiments` | `../engine` |
| `multispectral/config.py` | `PROJECT = dirname(here)` | walks up to the folder holding `archive (1)` |

Output directories were repointed so all three studies write into the single
`outputs/` tree instead of three separate ones.

**Verified after the merge:** the feature table loads (83,200 × 16, 3,328
fields), the splits are unchanged (1,664 / 832 / 832), all 16 figures and
`REPORT.md` regenerate, and all three studies import cleanly with the originals
removed.
