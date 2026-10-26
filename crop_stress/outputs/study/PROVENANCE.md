# Provenance — what came from where

Which earlier result each section of `REPORT.md` replaces, keeps, or corrects.
Written so that nothing has to be taken on trust, and so the older folders can
be read without confusion about which of their numbers still stand.

## Sources

| Folder | Role now |
|---|---|
| `hybrid_pipeline_experiments/` | The pipeline and the feature table. Read-only dependency. Still required. |
| `hybrid_pipeline_experiments2/` | Study 2. Its **questions** are re-run here; its **numbers** are superseded. |
| `hybrid_pipeline_experiments2/protocol_comparison/` | Folded in as section 7. |
| `experiment4/` | Folded in as sections 1 and 8. |
| `experiment3/` | **Not superseded.** Independent multispectral and cross-year study. |

## Section by section

| Section | Question | Origin | What changed |
|---|---|---|---|
| 1 · Operating point | Where should the threshold sit? | `experiment4` E1, E6 | Now measured alongside every other ablation at the same split |
| 2 · Detectors | Which detector; is the ensemble worth it? | Study 2 model comparison | Re-run on held-out fields with clean-calibrated thresholds. **Conclusion unchanged.** |
| 3 · Baselines | Does it beat hand-written rules? | Study 2 Figure 9 | Re-run under the new protocol |
| 4 · Persistence | How many consecutive passes? | Study 2 persistence table | Re-run; **causality claim corrected** |
| 5 · Detection limits | How weak or brief an event? | Study 2 severity table | Re-run under the new protocol |
| 6 · Signatures | Does stress shape matter? | Study 2 anomaly-types table | Re-run under the new protocol |
| 7 · Protocol effect | What does held-out evaluation cost? | `protocol_comparison` | **Corrects `experiment3`'s claim** |
| 8 · Ablations | Features; LOF `k` | Study 2 feature ablation + `experiment4` E4 | `k` sweep closes a stated limitation |

## Results deliberately NOT carried forward

| Result | Where from | Why dropped |
|---|---|---|
| "Evaluation protocol accounts for most of the headline number" | `experiment3` finding 2 | Refuted. That comparison ran across two codebases differing in input representation, cloud screening, observation density and injection procedure. Isolated properly (section 7), the protocol cost is small. |
| "Under a clean protocol LOF barely beats a naive rule" | `experiment3` | Was attributed to protocol. Since protocol is not the cause, this is unexplained and should not be cited until isolated. |
| Sliding 2-of-3 is "non-causal" | Study 2 audit §5 | Factually wrong. `utils/hybrid.py` slices `[t-2, t-1, t]`, strictly backward. |
| Event-level (range-based) metrics | `experiment4` E2 | Genuine negative result: changed recall by +0.002. Persistence makes detection all-or-nothing when events are as long as the persistence requirement, so there is no partial credit to award. Not repeated here. |
| Same-date peer-deviation feature | `experiment4` E3 | Genuine negative result: cost ~0.03 F1 and raised false alarms. Likely because injected fields pollute their own date median at a 15% injection rate. Not repeated here. |
| PCA | `config.py` | Fitted and persisted but consumed by nothing. Never part of the method. |
| `contamination` parameter | `config.py` | Inert — it shifts a decision offset that this scoring path never uses. A percentile ranking is invariant to it. |

## Results from `experiment3` that DO still stand

Both are within-`experiment3` controlled comparisons — same code, same data,
one variable changed — which is exactly what its refuted claim was not.

1. **Multispectral input raises F1 by roughly 44% relative** (0.243 → 0.351,
   permutation p ≈ 0.065). Four derived indices instead of one. This cannot be
   reproduced in the folders that consume `dataset_clean_2`, which contains only
   MSAVI, so `experiment3` is the sole source.
2. **Crop rotation breaks cross-year monitoring.** Using the previous season as
   a field's baseline scores worse than using other fields in the current
   season, because a field under maize one year and wheat the next has a
   different normal. A genuine agronomic result, and the direct caveat on this
   study's clean-calibration assumption.

## What is safe to archive

Nothing has been deleted. If the project is tidied later:

- `hybrid_pipeline_experiments2/outputs/` — superseded twice over, and still
  contains the three contradictory figure captions.
- `hybrid_pipeline_experiments2/plots/` and `outputs/plots/` — duplicate copies
  of four charts.
- `hybrid_pipeline_experiments2/generate_paper_spatial_plots.py` — fits and
  scores a model on the same single field.

**Do not archive** `hybrid_pipeline_experiments/` or `dataset_clean_2/`: this
folder, Study 2 and `experiment4` all read from them.
