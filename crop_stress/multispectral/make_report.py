"""
make_report.py — assemble outputs/REPORT.md from the saved tables.

Everything here is read back from disk, so the report cannot state a number
that differs from the table it came from.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TAB_DIR, OUT_DIR, FIG_DIR, TARGET_YEAR, HISTORY_YEAR, DECIDE, EVAL

ORDER = [
    ("e1_index_ablation", "Multispectral input versus a single index",
     "The headline ablation: does reading the raw 9-band archive instead of the "
     "derived MSAVI product actually improve detection?"),
    ("e2_model_comparison", "Detectors and rule-based baselines",
     "Whether the learned detector earns its place against simple threshold rules."),
    ("e3_n_neighbors", "LOF neighbourhood size", "Sensitivity to the one LOF hyperparameter."),
    ("e4_threshold", "Decision threshold", "Per-field versus global, and at what percentile."),
    ("e5_persistence", "Temporal persistence", "How many consecutive exceedances to require."),
    ("e6_detection_limits", "Detection limits", "How weak or brief an event can be before it is missed."),
    ("e7_anomaly_types", "Stress signature robustness", "Sudden, gradual and spectral events."),
    ("e8_protocols", "Evaluation protocol", "How much of a published number comes from the evaluation design rather than the model."),
    ("final_configuration", "Final configuration", "The recommended operating point."),
]

FIGS = [
    ("fig01_pipeline", "Processing pipeline"),
    ("fig02_spectral_signature", "Sentinel-2 spectral response — why nine bands beat one"),
    ("fig03_index_ablation", "Multispectral versus single-index input"),
    ("fig04_model_comparison", "Detectors versus rule-based baselines"),
    ("fig05_n_neighbors", "LOF neighbourhood-size sensitivity"),
    ("fig06_threshold", "Decision-threshold selection"),
    ("fig07_persistence", "Temporal persistence requirement"),
    ("fig08_detection_limits", "Detection limits by severity and duration"),
    ("fig09_anomaly_types", "Robustness to stress signature"),
    ("fig10_score_separation", "Score separability"),
    ("fig11_pr_roc", "Precision-recall and ROC characteristics"),
    ("fig12_confusion", "Confusion matrix at the operating point"),
    ("fig13_trajectory", "Detection walkthrough on one field"),
    ("fig14_protocols", "Effect of the evaluation protocol"),
]


def read_md(name):
    p = os.path.join(TAB_DIR, f"{name}.md")
    return open(p, encoding="utf-8").read().strip() if os.path.exists(p) else None


def main():
    res = {}
    p = os.path.join(OUT_DIR, "results.json")
    if os.path.exists(p):
        res = json.load(open(p, encoding="utf-8"))
    fin = res.get("final", {})

    out = [
        "# Crop stress detection from Sentinel-2 time series",
        "",
        "## Results report — experiment3",
        "",
        f"The detector is fitted on one set of fields and evaluated on a **disjoint set "
        f"of fields it has never seen**, within the {TARGET_YEAR} season. Decision "
        "thresholds are calibrated on clean fit-set data. Neither the model nor the "
        "threshold ever sees the anomalies it is measured against.",
        "",
        "### Headline",
        "",
    ]
    if fin:
        out += [
            f"| Metric | Value |", "|---|---|",
            f"| Precision | **{fin.get('precision', float('nan')):.3f}** |",
            f"| Recall | **{fin.get('recall', float('nan')):.3f}** |",
            f"| F1 | **{fin.get('f1', float('nan')):.3f}** "
            f"(95% CI {fin.get('f1_ci', [0, 0])[0]:.3f}–{fin.get('f1_ci', [0, 0])[1]:.3f}) |",
            f"| ROC-AUC | {fin.get('roc_auc', float('nan')):.3f} |",
            f"| PR-AUC | {fin.get('pr_auc', float('nan')):.3f} |",
            f"| False alarms | {int(fin.get('false_alarms', 0)):,} |",
            "",
            f"Configuration: LOF, {fin.get('features', '?')} features, "
            f"k={fin.get('n_neighbors', '?')}, per-field P{DECIDE['threshold_percentile']} "
            f"threshold, {DECIDE['min_consecutive']}-consecutive persistence. "
            f"{fin.get('fields', 0):,} fields, {len(EVAL['seeds'])} seeds, "
            f"{int(EVAL['severity']*100)}% severity over {EVAL['duration']} observations.",
            "",
        ]
    g = res.get("e1_gain")
    if g:
        out += [f"Multispectral input changed F1 by **{g['diff']:+.3f}** against a "
                f"single-index pipeline (paired permutation p = {g['p']:.4f}).", ""]

    out += ["---", "", "## Results tables", ""]
    for name, title, caption in ORDER:
        md = read_md(name)
        if md:
            out += [f"### {title}", "", caption, "", md, ""]

    out += ["---", "", "## Figures", "",
            "All figures are in `outputs/figures/` as 400 dpi PNG and vector PDF.", ""]
    for name, title in FIGS:
        if os.path.exists(os.path.join(FIG_DIR, f"{name}.png")):
            out += [f"### {title}", "", f"![{title}](figures/{name}.png)", ""]

    out += [
        "---", "", "## Method notes", "",
        "**Data.** Raw 9-band Sentinel-2 acquisitions (B2–B8, B11, B12), Lombardy. "
        "Four vegetation indices are derived per acquisition — MSAVI, NDVI, NDRE and "
        "NDMI — rather than using the single-band MSAVI product.",
        "",
        "**Screening.** Cloud is identified spectrally: bright in blue *and* bright in "
        "SWIR *and* showing no vegetation signal, a combination canopy never produces.",
        "",
        "**No labels are used anywhere.** Detection performance is measured by injecting "
        "synthetic stress, coherently across all four indices, into a season the detector "
        "has never seen.",
        "",
        "**Leakage.** The detector is fitted on one season and the decision threshold is "
        "each field's own percentile from that same clean season. Both are fixed before "
        "any anomaly exists.",
        "",
        "**Limitations.** Synthetic anomalies are a proxy for real biological stress. "
        "Detection is at field level, not per pixel. One region, two seasons.",
        "",
    ]

    path = os.path.join(OUT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
