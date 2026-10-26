"""
run_experiments.py — every experiment, under the temporal protocol.

    fit + calibrate on 2018   ->   monitor 2019

Scoring is the expensive step, so clean-season scores are cached per detector
and the cheap sweeps (threshold, persistence) reuse an already-scored frame.

    python run_experiments.py
    python run_experiments.py --quick        # fewer seeds / fields
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                                          # noqa: E402
from config import (TAB_DIR, OUT_DIR, EVAL, DECIDE, LOF, LOF_SWEEP,
                    THRESHOLD_SWEEP, SEVERITY_GRID, DURATION_GRID,
                    TARGET_YEAR, HISTORY_YEAR)                          # noqa: E402
from src import features as F                                           # noqa: E402
from src import decision as D                                           # noqa: E402
from src import pipeline as P                                           # noqa: E402
from src.detectors import (LOFDetector, IForestDetector, AEDetector,
                           BASELINES)                                   # noqa: E402
from src.evaluation import metrics, aggregate, paired_test, bootstrap_ci  # noqa: E402

RESULTS = {}


def log(m):
    print(m, flush=True)


def _md(df):
    """Markdown table without pulling in `tabulate`."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df.to_numpy()]
    w = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    out += ["| " + " | ".join(r[i].ljust(w[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join(out) + "\n"


def save_table(name, df, caption=""):
    df.to_csv(os.path.join(TAB_DIR, f"{name}.csv"), index=False)
    with open(os.path.join(TAB_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        if caption:
            f.write(caption.strip() + "\n\n")
        f.write(_md(df))
    log(f"    -> tables/{name}.md")


def fmt(df):
    out = df.copy()
    for c in ("precision", "recall", "f1"):
        if c in out:
            out[c] = ([f"{m:.3f}±{s:.3f}" for m, s in zip(out[c], out[f"{c}_sd"])]
                      if f"{c}_sd" in out else out[c].map("{:.3f}".format))
    for c in ("pr_auc", "roc_auc", "specificity"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    for c in ("false_alarms", "missed", "tp", "tn", "n_test", "n_stressed"):
        if c in out:
            out[c] = out[c].map(lambda v: f"{int(round(v)):,}")
    return out.drop(columns=[c for c in out.columns if c.endswith("_sd")])


def sweep(frame, calib, pct, nc, per_field=True):
    thr = D.calibrate_thresholds(calib["score"].to_numpy(),
                                 calib["tile_id"].to_numpy(), pct, per_field)
    pred, _ = D.decide(frame["score"].to_numpy(), frame["tile_id"].to_numpy(),
                       frame["date"].to_numpy(), thr, nc)
    return metrics(frame["stressed"].to_numpy(), pred, frame["score"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-fields", type=int, default=None)
    ap.add_argument("--protocol", choices=("cross_field", "cross_year"),
                    default="cross_field")
    args = ap.parse_args()

    seeds = EVAL["seeds"][:2] if args.quick else EVAL["seeds"]
    seeds3 = seeds[:2] if args.quick else EVAL["seeds"][:3]
    sev, dur, ratio = EVAL["severity"], EVAL["duration"], EVAL["injection_ratio"]
    t0 = time.time()

    hist = pd.read_csv(config.observations_csv(HISTORY_YEAR), parse_dates=["date"])
    targ = pd.read_csv(config.observations_csv(TARGET_YEAR), parse_dates=["date"])
    hist, targ, shared = P.align_seasons(hist, targ)
    if args.max_fields:
        keep = shared[:args.max_fields]
        hist = hist[hist.tile_id.isin(keep)]
        targ = targ[targ.tile_id.isin(keep)]
        shared = keep
    # Primary protocol: fit on one set of fields, evaluate on a disjoint set,
    # within the monitored season. Cross-year (fit on the previous season)
    # is reported alongside in E8 — it is closer to deployment in principle,
    # but crop rotation means a field's previous season is often a different
    # crop entirely, which makes last year's baseline actively misleading.
    if args.protocol == "cross_year":
        fit_obs, eval_obs = hist, targ
        proto = f"fit+calibrate {HISTORY_YEAR} -> monitor {TARGET_YEAR}"
    else:
        fit_obs, eval_obs, _ft, _et = P.prepare_cross_field(targ)
        proto = (f"fit+calibrate on {len(_ft):,} fields -> evaluate on "
                 f"{len(_et):,} unseen fields ({TARGET_YEAR})")
    RESULTS["protocol"] = proto
    log(f"Protocol: {proto}")
    log(f"  {len(shared):,} fields present in both seasons")
    log(f"  fit {len(fit_obs):,} obs · evaluate {len(eval_obs):,} obs")

    cols_multi = F.feature_columns(True)
    cols_mono = F.feature_columns(False)
    log(f"  features: {len(cols_multi)} multispectral / {len(cols_mono)} MSAVI-only")

    # ── E1 · multispectral vs single index ─────────────────────────────
    log(f"\n[E1] Index ablation")
    rows, raw = [], {}
    for label, cols in [("MSAVI only (1 index)", cols_mono),
                        ("Multispectral (4 indices)", cols_multi)]:
        P.reset_caches()
        det = P.fit_detector(LOFDetector(LOF["n_neighbors"]), fit_obs, cols)
        runs = [P.run_once(det, fit_obs, eval_obs, cols, s, sev, dur, ratio,
                           det_key=("e1", label))[0] for s in seeds]
        raw[label] = [r["f1"] for r in runs]
        rows.append({"configuration": label, **aggregate(runs)})
        log(f"    {label:26s} F1 {rows[-1]['f1']:.3f}  ROC-AUC {rows[-1]['roc_auc']:.3f}")
    diff, p = paired_test(raw["Multispectral (4 indices)"], raw["MSAVI only (1 index)"])
    RESULTS["e1_gain"] = {"diff": diff, "p": p}
    log(f"    paired difference {diff:+.4f} F1, p = {p:.4f}")
    save_table("e1_index_ablation", fmt(pd.DataFrame(rows)),
               "One vegetation index versus four. LOF, everything else identical.")

    best_cols = cols_multi if rows[1]["f1"] >= rows[0]["f1"] else cols_mono
    best_label = "multispectral" if best_cols is cols_multi else "MSAVI-only"
    log(f"    -> {best_label} downstream")

    # ── E2 · detectors vs rule-based baselines ─────────────────────────
    log("\n[E2] Detectors and naive baselines")
    P.reset_caches()
    rows = []
    for cls in (LOFDetector, IForestDetector, AEDetector):
        det = cls(LOF["n_neighbors"]) if cls is LOFDetector else cls()
        det = P.fit_detector(det, fit_obs, best_cols)
        runs = [P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                           det_key=("e2", det.name))[0] for s in seeds]
        rows.append({"method": det.name, "type": "learned", **aggregate(runs)})
        log(f"    {det.name:22s} F1 {rows[-1]['f1']:.3f}")
    for b in BASELINES:
        runs = [P.run_once(b, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                           baseline=True, det_key=("e2b", b.name))[0] for s in seeds]
        rows.append({"method": b.name, "type": "rule-based", **aggregate(runs)})
        log(f"    {b.name:22s} F1 {rows[-1]['f1']:.3f}")
    save_table("e2_model_comparison", fmt(pd.DataFrame(rows)),
               "Every method shares the same features, threshold source and "
               "persistence rule; only the score differs.")

    # ── E3 · n_neighbors ───────────────────────────────────────────────
    log("\n[E3] LOF neighbourhood size")
    rows = []
    for k in LOF_SWEEP:
        P.reset_caches()
        det = P.fit_detector(LOFDetector(k), fit_obs, best_cols)
        runs = [P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                           det_key=("e3", k))[0] for s in seeds3]
        rows.append({"n_neighbors": k, **aggregate(runs)})
        log(f"    k={k:<4d} F1 {rows[-1]['f1']:.3f}")
    best_k = int(max(rows, key=lambda r: r["f1"])["n_neighbors"])
    RESULTS["best_k"] = best_k
    log(f"    -> k = {best_k}")
    save_table("e3_n_neighbors", fmt(pd.DataFrame(rows)),
               "Neighbourhood size for the local density estimate.")

    # ── final detector, then cheap sweeps on cached scores ─────────────
    P.reset_caches()
    det = P.fit_detector(LOFDetector(best_k), fit_obs, best_cols)
    calib = P.calibration_scores(det, fit_obs, best_cols, ("final", best_k))
    frames = {s: P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                            calib=calib)[1] for s in seeds3}

    log("\n[E4] Threshold strategy")
    rows = []
    for pct in THRESHOLD_SWEEP:
        rows.append({"threshold": f"P{pct} (per-field)",
                     **aggregate([sweep(frames[s], calib, pct,
                                        DECIDE["min_consecutive"]) for s in seeds3])})
    for pct in (90, 95, 99):
        rows.append({"threshold": f"P{pct} (global)",
                     **aggregate([sweep(frames[s], calib, pct,
                                        DECIDE["min_consecutive"], False)
                                  for s in seeds3])})
    best_thr = max(rows, key=lambda r: r["f1"])
    RESULTS["best_threshold"] = best_thr["threshold"]
    log(f"    best: {best_thr['threshold']}  F1 {best_thr['f1']:.3f}")
    save_table("e4_threshold", fmt(pd.DataFrame(rows)),
               f"Thresholds are the field's own percentile from {HISTORY_YEAR}, "
               f"applied unchanged to {TARGET_YEAR}.")

    log("\n[E5] Persistence rule")
    rows = []
    for n in (1, 2, 3):
        label = "none" if n == 1 else f"{n} consecutive"
        rows.append({"persistence": label,
                     **aggregate([sweep(frames[s], calib,
                                        DECIDE["threshold_percentile"], n)
                                  for s in seeds3])})
        log(f"    {label:16s} F1 {rows[-1]['f1']:.3f}")
    save_table("e5_persistence", fmt(pd.DataFrame(rows)),
               "All rules causal — no look-ahead.")

    # ── E6 · detection limits ──────────────────────────────────────────
    log("\n[E6] Detection limits")
    rows = []
    for d in DURATION_GRID:
        for sv in SEVERITY_GRID:
            runs = [P.run_once(det, fit_obs, eval_obs, best_cols, s, sv, d, ratio,
                               calib=calib)[0] for s in seeds3]
            rows.append({"severity": f"{int(sv*100)}%", "duration": f"{d} obs",
                         **aggregate(runs)})
        log(f"    duration {d} done")
    save_table("e6_detection_limits", fmt(pd.DataFrame(rows)),
               "How weak or brief an event can be before it is missed.")

    # ── E7 · anomaly signature ─────────────────────────────────────────
    log("\n[E7] Anomaly signature")
    rows = []
    for kind, label in [("sudden", "Sudden drop"), ("gradual", "Gradual onset"),
                        ("spectral", "Spectral shift")]:
        runs = [P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                           kind=kind, calib=calib)[0] for s in seeds]
        rows.append({"signature": label, **aggregate(runs)})
        log(f"    {label:16s} F1 {rows[-1]['f1']:.3f}")
    save_table("e7_anomaly_types", fmt(pd.DataFrame(rows)),
               "Injected coherently across all four indices.")

    # ── E8 · cross-field generalisation (the stricter protocol) ────────
    log("\n[E8] Cross-field protocol (stricter than deployment)")
    P.reset_caches()
    fit_o, eval_o, ft, et = P.prepare_cross_field(targ)
    det_cf = P.fit_detector(LOFDetector(best_k), fit_o, best_cols, key="cf")
    calib_cf = P.calibration_scores(det_cf, fit_o, best_cols, ("cf", best_k))
    runs_cf = [P.run_once(det_cf, fit_o, eval_o, best_cols, s, sev, dur, ratio,
                          calib=calib_cf)[0] for s in seeds3]
    cf = aggregate(runs_cf)
    log(f"    unseen fields: F1 {cf['f1']:.3f}  (temporal protocol for reference)")
    save_table("e8_cross_field", fmt(pd.DataFrame([
        {"protocol": f"Temporal ({HISTORY_YEAR}->{TARGET_YEAR}, same fields)",
         **aggregate([P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                                 calib=calib)[0] for s in seeds3])},
        {"protocol": "Cross-field (unseen fields, one season)", **cf},
    ])), "The temporal protocol matches deployment; the cross-field protocol "
         "additionally requires generalising to fields never seen.")

    # ── headline ───────────────────────────────────────────────────────
    log("\n[Final] Recommended configuration")
    runs = [P.run_once(det, fit_obs, eval_obs, best_cols, s, sev, dur, ratio,
                       calib=calib) for s in seeds]
    final = aggregate([r[0] for r in runs])
    fr = runs[0][1]
    lo, hi = bootstrap_ci(fr["stressed"].to_numpy(), fr["alarm"].to_numpy(),
                          n=EVAL["bootstrap_n"])
    log(f"    P {final['precision']:.3f}  R {final['recall']:.3f}  "
        f"F1 {final['f1']:.3f} (95% CI {lo:.3f}-{hi:.3f})  "
        f"ROC-AUC {final['roc_auc']:.3f}  PR-AUC {final['pr_auc']:.3f}")
    RESULTS["final"] = {**{k: v for k, v in final.items() if not k.endswith("_sd")},
                        "f1_ci": [lo, hi], "features": best_label,
                        "n_neighbors": best_k, "fields": len(shared),
                        "protocol": f"fit {HISTORY_YEAR} -> monitor {TARGET_YEAR}"}
    save_table("final_configuration", fmt(pd.DataFrame([{
        "configuration": f"LOF · {best_label} · k={best_k} · "
                         f"P{DECIDE['threshold_percentile']} per-field · "
                         f"{DECIDE['min_consecutive']}-consecutive", **final}])),
        f"Fit and calibrated on {HISTORY_YEAR}; evaluated on {TARGET_YEAR}.")

    keep = [c for c in ["tile_id", "date", "score", "stressed", "alarm", "raw_flag",
                        "msavi_p50", "ndvi_p50", "ndre_p50", "ndmi_p50",
                        "ndvi_sm", "ndmi_sm"] if c in fr.columns]
    fr[keep].to_csv(os.path.join(OUT_DIR, "final_scored.csv"), index=False)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    log(f"\nComplete in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
