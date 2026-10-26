"""
run_all.py — run every experiment, write results.json, tables, figures, report.

  cd experiment4
  C:/Users/prakh/AppData/Local/Programs/Python/Python314/python.exe run_all.py

Nothing outside experiment4/ is written.
"""

import json
import os
import time

import numpy as np

import common as C
import experiments as E
import make_figures
import make_report


def _fmt(m, keys, pct=None):
    """One markdown row: mean±sd for score keys, plain int for counts."""
    cells = []
    for k in keys:
        v = m.get(k)
        if v is None:
            cells.append("—")
        elif k in ("false_alarms", "missed", "n_true_events", "n_pred_events"):
            cells.append(f"{int(round(v)):,}")
        else:
            sd = m.get(k + "_sd")
            cells.append(f"{v:.4f}±{sd:.3f}" if sd is not None else f"{v:.4f}")
    return " | ".join(cells)


def write_table(name, header, rows):
    path = os.path.join(C.TABLE_DIR, f"{name}.md")
    md = "| " + " | ".join(header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(r) + " |\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  wrote tables/{name}.md")
    return md


def main():
    t0 = time.time()
    C.ensure_dirs()
    R = {}

    print("=" * 66)
    print("experiment4 — building study (fit / calibrate / test splits)")
    print("=" * 66)
    study = E.Study()
    n = {k: int(m.sum()) for k, m in study.mask.items()}
    R["setup"] = {
        "n_tiles": {k: len(v) for k, v in study.splits.items()},
        "n_rows": n,
        "seeds": C.SEEDS,
        "injection_ratio": C.INJECTION_RATIO,
        "severity": C.SEVERITY,
        "duration": C.DURATION,
        "min_consecutive": C.MIN_CONSECUTIVE,
        "split_seed": C.SPLIT_SEED,
        "baseline_pctl": C.BASELINE_PCTL,
        "k_default": C.LOF["n_neighbors"],
    }
    for k in ("fit", "cal", "test"):
        print(f"  {k:5s}: {len(study.splits[k]):>5,} tiles  {n[k]:>7,} rows")

    # warm the injection cache
    for s in C.SEEDS:
        study.draw(s)
    print(f"  injections ready ({time.time() - t0:.0f}s)")

    # ── E1 ───────────────────────────────────────────────────────────
    print("\n[E1] clean-calibrated threshold, percentile chosen on `cal`")
    e1 = E.e1_threshold(study)
    print(f"  p* = P{e1['p_star']:g}")
    for k, v in e1["results"].items():
        print(f"  {k:24s} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")

    # ── E2 ───────────────────────────────────────────────────────────
    print("\n[E2] event-level (range-based) metrics")
    e2 = E.e2_range_metrics(study, e1)
    for k in sorted(e2):
        v = e2[k]
        print(f"  {k:38s} P={v['range_precision']:.4f} "
              f"R={v['range_recall']:.4f} F1={v['range_f1']:.4f}")

    R["e1"] = {k: v for k, v in e1.items() if k != "_preds"}
    R["e2"] = e2

    # ── E5 (uses the same study object) ──────────────────────────────
    print("\n[E5] adaptive P85 vs global P90 — significance")
    e5 = E.e5_significance(study)
    print(f"  ΔF1 = {e5['delta_f1']:+.4f}  95% CI "
          f"[{e5['ci95'][0]:+.4f}, {e5['ci95'][1]:+.4f}]  p = {e5['p_value']:.4f}")
    R["e5"] = e5

    # ── E6 ───────────────────────────────────────────────────────────
    print("\n[E6] extreme-value (POT/GPD) thresholding")
    try:
        e6 = E.e6_evt(study)
        print(f"  q* = {e6['q_star']}  xi = {e6['xi']:.4f}  "
              f"test F1 = {e6['test']['f1']:.4f} "
              f"FA = {int(e6['test']['false_alarms']):,}")
        R["e6"] = e6
    except Exception as exc:      # a tail fit on 25 points is allowed to fail
        print(f"  EVT failed: {exc}")
        R["e6"] = None

    del study

    # ── E3 (refits the detector, so runs on its own) ─────────────────
    print("\n[E3] same-date peer-deviation feature")
    e3 = E.e3_peer_feature(p_star=R["e1"]["p_star"])
    for k, v in e3.items():
        print(f"  {k:8s} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"F1={v['f1']:.4f} FA={int(v['false_alarms']):,}")
    R["e3"] = e3

    # ── E4 ───────────────────────────────────────────────────────────
    print("\n[E4] LOF neighbourhood size sweep")
    e4 = E.e4_kneighbors(p_star=R["e1"]["p_star"])
    for r in e4["test"]:
        print(f"  k={r['k']:>3d}  F1={r['f1']:.4f}  FA={int(r['false_alarms']):,}")
    print(f"  k* (chosen on cal) = {e4['k_star']}")
    R["e4"] = e4

    # ── persist ──────────────────────────────────────────────────────
    R["runtime_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(C.OUT_DIR, "results.json"), "w",
              encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(f"\nWrote outputs/results.json")

    # ── tables ───────────────────────────────────────────────────────
    print("\nWriting tables...")
    ps = R["e1"]["p_star"]
    label = {"baseline_p85_postinj": f"Study 2 baseline (adaptive P{C.BASELINE_PCTL}, post-injection)",
             "clean_p85": f"Clean-calibrated P{C.BASELINE_PCTL}",
             "clean_pstar": f"**Clean-calibrated P{ps:g}** (chosen on cal)"}
    cols = ["precision", "recall", "f1", "pr_auc", "specificity", "false_alarms"]
    write_table("t1_threshold", ["Threshold rule", "Precision", "Recall", "F1",
                                 "PR-AUC", "Specificity", "False alarms"],
                [[label[k], *_fmt(R["e1"]["results"][k], cols).split(" | ")]
                 for k in label])

    rows = []
    for k in label:
        pt = R["e1"]["results"][k]
        rg = R["e2"][f"{k}@alpha0.5"]
        rows.append([label[k],
                     f"{pt['precision']:.4f}", f"{pt['recall']:.4f}",
                     f"{pt['f1']:.4f}",
                     f"{rg['range_precision']:.4f}", f"{rg['range_recall']:.4f}",
                     f"{rg['range_f1']:.4f}"])
    write_table("t2_point_vs_event",
                ["Threshold rule", "P (point)", "R (point)", "F1 (point)",
                 "P (event)", "R (event)", "F1 (event)"], rows)

    write_table("t3_peer_feature",
                ["Feature set", "Precision", "Recall", "F1", "PR-AUC",
                 "False alarms"],
                [["16 features (Study 2)",
                  *_fmt(R["e3"]["base_16"], ["precision", "recall", "f1",
                                             "pr_auc", "false_alarms"]).split(" | ")],
                 ["17 features (+ peer deviation)",
                  *_fmt(R["e3"]["peer_17"], ["precision", "recall", "f1",
                                             "pr_auc", "false_alarms"]).split(" | ")]])

    write_table("t4_kneighbors",
                ["k", "Precision", "Recall", "F1", "PR-AUC", "False alarms"],
                [[str(r["k"]), *_fmt(r, ["precision", "recall", "f1", "pr_auc",
                                         "false_alarms"]).split(" | ")]
                 for r in R["e4"]["test"]])

    write_table("t5_significance",
                ["Rule", "Precision", "Recall", "F1", "False alarms"],
                [["Adaptive P85",
                  *_fmt(R["e5"]["adaptive_p85"], ["precision", "recall", "f1",
                                                  "false_alarms"]).split(" | ")],
                 ["Global P90",
                  *_fmt(R["e5"]["global_p90"], ["precision", "recall", "f1",
                                                "false_alarms"]).split(" | ")]])

    if R.get("e6"):
        write_table("t6_evt",
                    ["Rule", "Precision", "Recall", "F1", "PR-AUC",
                     "False alarms"],
                    [[f"Clean-calibrated P{ps:g}",
                      *_fmt(R["e1"]["results"]["clean_pstar"],
                            ["precision", "recall", "f1", "pr_auc",
                             "false_alarms"]).split(" | ")],
                     [f"EVT / GPD (q={R['e6']['q_star']})",
                      *_fmt(R["e6"]["test"], ["precision", "recall", "f1",
                                              "pr_auc", "false_alarms"]).split(" | ")]])

    # ── figures + report ─────────────────────────────────────────────
    print()
    make_figures.main()
    print()
    make_report.main()

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
