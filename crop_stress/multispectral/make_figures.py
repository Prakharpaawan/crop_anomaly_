"""
make_figures.py — every paper figure, from the tables run_experiments.py wrote.

Figures are generated from the saved results, never from hardcoded numbers, so
a figure cannot silently disagree with the table beside it.

    python make_figures.py
"""
import os
import sys
import glob
import json
import re
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (OUT_DIR, TAB_DIR, RAW_DIR, INDICES, DECIDE, EVAL,
                    OBSERVATIONS_CSV)                                  # noqa: E402
from src import viz as V                                               # noqa: E402


def num(s):
    """'0.812±0.004' -> 0.812 ; '1,234' -> 1234.0"""
    if isinstance(s, (int, float, np.floating)):
        return float(s)
    t = str(s).replace(",", "").split("±")[0].strip()
    try:
        return float(t)
    except ValueError:
        return np.nan


def load(name):
    """Read a results CSV back into numeric columns for plotting."""
    df = pd.read_csv(os.path.join(TAB_DIR, f"{name}.csv"))
    for c in df.columns:
        if c in ("precision", "recall", "f1", "pr_auc", "roc_auc", "specificity",
                 "false_alarms", "missed", "tp", "tn", "n_test", "n_stressed") \
           or c.endswith("_sd"):
            df[c] = df[c].map(num)
    return df


# ══════════════════════════════════════════════════════════════════════
def fig_pipeline():
    steps = [
        ("1", "Spectral screening", "cloud + shadow rejected using blue,\nSWIR and NDVI together"),
        ("2", "Four vegetation indices", "MSAVI, NDVI, NDRE, NDMI\nper acquisition"),
        ("3", "Temporal features", "Savitzky-Golay level, change,\nspread, cross-field context"),
        ("4", "Standardise", "zero mean, unit variance\nover all features"),
        ("5", "LOF scoring", "novelty mode, fitted on\nheld-out clean fields"),
        ("6", "Per-field threshold", "calibrated on each field's own\nclean calibration window"),
        ("7", "Causal persistence", "N consecutive exceedances,\nno look-ahead"),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(5.9, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(-0.95, n + 0.8); ax.axis("off")
    for i, (k, title, sub) in enumerate(steps):
        y = n - 1 - i
        ax.add_patch(plt.Rectangle((0.05, y + .13), .90, .70, facecolor=V.SOFT,
                                   edgecolor=V.GREEN, lw=1.1, zorder=2))
        ax.add_patch(plt.Rectangle((0.05, y + .13), .055, .70, facecolor=V.GREEN,
                                   edgecolor=V.GREEN, lw=1.1, zorder=3))
        ax.text(.0775, y + .48, k, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold", zorder=4)
        ax.text(.145, y + .655, title, ha="left", va="center", fontsize=9,
                fontweight="bold", zorder=4)
        ax.text(.145, y + .335, sub, ha="left", va="center", fontsize=7.4,
                color=V.GREY, zorder=4, linespacing=1.35)
        if i < n - 1:
            ax.annotate("", xy=(.5, y - .04), xytext=(.5, y + .09),
                        arrowprops=dict(arrowstyle="-|>", color=V.GREEN, lw=1.3))
    ax.text(.5, n + .5, "Sentinel-2 raw 9-band acquisition", ha="center",
            fontsize=9, fontweight="bold")
    ax.annotate("", xy=(.5, n - .02), xytext=(.5, n + .32),
                arrowprops=dict(arrowstyle="-|>", color=V.GREEN, lw=1.3))
    ax.annotate("", xy=(.5, -.34), xytext=(.5, -.02),
                arrowprops=dict(arrowstyle="-|>", color=V.VERM, lw=1.3))
    ax.text(.5, -.48, "Field-level stress alarm", ha="center", va="top",
            fontsize=9, fontweight="bold", color=V.VERM)
    ax.text(.5, -.74,
            "Unsupervised end to end. Detectors are fitted on one set of fields and\n"
            "evaluated on a disjoint set; thresholds never see the data they judge.",
            ha="center", va="top", fontsize=7, color=V.GREY, linespacing=1.5)
    V.save(fig, "fig01_pipeline")


def fig_spectral_signature():
    """Why nine bands beat one: the red edge and SWIR carry separate information."""
    import rasterio
    RAW = re.compile(r"^\d{8}\.tif$")
    tile = sorted(os.listdir(RAW_DIR), key=lambda x: int(x) if x.isdigit() else 1 << 30)[1]
    paths = sorted(p for p in glob.glob(os.path.join(RAW_DIR, tile, "*.tif"))
                   if RAW.match(os.path.basename(p)))
    want = {"winter": 1, "peak": 7}
    picked = {}
    for p in paths:
        d = dt.datetime.strptime(os.path.basename(p)[:8], "%Y%m%d")
        for k, mth in want.items():
            if d.month == mth and k not in picked:
                picked[k] = (d, p)
    if len(picked) < 2:
        print("    (skipping spectral signature: no suitable dates)")
        return

    wl = [490, 560, 665, 705, 740, 783, 842, 1610, 2190]
    names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for (k, (d, p)), c, lab in [(list(picked.items())[0], V.ORANGE, "Bare / dormant"),
                                (list(picked.items())[1], V.GREEN, "Peak canopy")]:
        with rasterio.open(p) as s:
            cube = s.read().astype(float)
        prof = [np.nanmedian(cube[i]) for i in range(9)]
        ax.plot(wl, prof, "o-", color=c, lw=1.9, ms=5,
                label=f"{lab} — {d:%d %b %Y}")
    ax.axvspan(700, 800, color=V.GREEN, alpha=.07)
    ax.text(750, ax.get_ylim()[1] * .97, "red edge", ha="center", va="top",
            fontsize=7.5, color=V.GREY)
    ax.axvspan(1500, 2300, color=V.BLUE, alpha=.06)
    ax.text(1900, ax.get_ylim()[1] * .97, "SWIR — water absorption",
            ha="center", va="top", fontsize=7.5, color=V.GREY)
    for x, nm in zip(wl, names):
        ax.annotate(nm, (x, 0), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=6.5, color=V.GREY)
    ax.set_xlabel("Wavelength (nm)", labelpad=12)
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Sentinel-2 spectral response — tile {tile}", loc="left", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=2)
    fig.text(0, -0.06,
             "The derived MSAVI product used previously collapses all nine bands to one number. "
             "The red-edge\nbands track chlorophyll and the SWIR bands track canopy water — "
             "independent stress channels that\na single greenness index cannot represent.",
             fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig02_spectral_signature")


def fig_index_ablation():
    df = load("e1_index_ablation")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.4, 1], hspace=.12))
    V.metric_bars(a1, df, "configuration",
                  highlight="Multispectral (4 indices)")
    a1.set_title("Does multispectral input actually help?", loc="left", pad=20)
    V.legend_above(a1)
    try:
        res = json.load(open(os.path.join(OUT_DIR, "results.json"), encoding="utf-8"))
        g = res.get("e1_gain", {})
        a1.text(.005, .97,
                f"Paired difference {g.get('diff', float('nan')):+.3f} F1 · "
                f"permutation p = {g.get('p', float('nan')):.4f}\n"
                f"LOF, identical features pipeline, {len(EVAL['seeds'])} seeds, "
                "held-out fields.",
                transform=a1.transAxes, va="top", fontsize=7, color=V.GREY)
    except Exception:
        pass
    V.false_alarm_bars(a2, df, "configuration")
    a2.set_xlabel("Feature set")
    V.save(fig, "fig03_index_ablation")


def fig_models():
    df = load("e2_model_comparison")
    order = df.sort_values("f1", ascending=False).reset_index(drop=True)
    best = order.loc[0, "method"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.4, 1], hspace=.12))
    V.metric_bars(a1, order, "method", highlight=best)
    a1.set_title("Learned detectors versus rule-based baselines", loc="left", pad=20)
    V.legend_above(a1)
    a1.text(.005, .97,
            "Identical features, threshold and persistence rule for every method —\n"
            "only the scoring function differs.",
            transform=a1.transAxes, va="top", fontsize=7, color=V.GREY)
    V.false_alarm_bars(a2, order, "method")
    a2.set_xlabel("Method")
    for lbl in a2.get_xticklabels():
        lbl.set_rotation(12)
        lbl.set_ha("right")
    V.save(fig, "fig04_model_comparison")


def fig_neighbors():
    df = load("e3_n_neighbors").sort_values("n_neighbors")
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for m, c in zip(V.METRICS, V.MCOLOR):
        ax.plot(df["n_neighbors"], df[m], "o-", color=c, lw=1.8, ms=4,
                label=V.MLABEL[m])
        if f"{m}_sd" in df:
            ax.fill_between(df["n_neighbors"], df[m] - df[f"{m}_sd"],
                            df[m] + df[f"{m}_sd"], color=c, alpha=.15)
    best = df.loc[df["f1"].idxmax(), "n_neighbors"]
    ax.axvline(best, color=V.GREY, ls="--", lw=1)
    ax.annotate(f"selected k = {int(best)}", (best, .04), fontsize=7.5,
                color=V.GREY, xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("LOF n_neighbors"); ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
    ax.set_title("Neighbourhood size sensitivity", loc="left", pad=20)
    V.legend_above(ax)
    fig.text(0, -.02,
             "The earlier pipeline fixed n_neighbors at 10 without ever sweeping it.",
             fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig05_n_neighbors")


def fig_threshold():
    df = load("e4_threshold")
    per = df[df["threshold"].str.contains("per-field")].copy()
    glo = df[df["threshold"].str.contains("global")].copy()
    per["p"] = per["threshold"].str.extract(r"P(\d+)").astype(int)
    glo["p"] = glo["threshold"].str.extract(r"P(\d+)").astype(int)
    per = per.sort_values("p")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.7, 1], hspace=.14))
    for m, c in zip(V.METRICS, V.MCOLOR):
        a1.plot(per["p"], per[m], "o-", color=c, lw=1.8, ms=3.5, label=V.MLABEL[m])
    if len(glo):
        a1.scatter(glo["p"], glo["f1"], marker="D", s=42, color="black", zorder=5,
                   label="global threshold (F1)")
    chosen = DECIDE["threshold_percentile"]
    a1.axvline(chosen, color=V.GREY, ls="--", lw=1)
    a1.annotate(f"chosen P{chosen}", (chosen, .05), fontsize=7.5, color=V.GREY,
                xytext=(3, 0), textcoords="offset points")
    a1.set_ylabel("Score"); a1.set_ylim(0, 1.02)
    a1.set_title("Choosing the decision threshold", loc="left", pad=20)
    V.legend_above(a1, ncols=4)

    a2.plot(per["p"], per["false_alarms"], "o-", color=V.VERM, lw=1.8, ms=3.5)
    a2.set_yscale("log")
    a2.set_ylabel("False alarms (log)"); a2.set_xlabel("Threshold percentile")
    fig.text(0, -.03,
             "Thresholds are learned on each field's clean calibration window and applied "
             "unchanged to its test\nwindow, so no boundary is ever fitted to the data it "
             "judges.", fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig06_threshold")


def fig_persistence():
    df = load("e5_persistence")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.4, 1], hspace=.12))
    best = df.loc[df["f1"].idxmax(), "persistence"]
    V.metric_bars(a1, df, "persistence", highlight=best)
    a1.set_title("Temporal persistence requirement", loc="left", pad=20)
    V.legend_above(a1)
    a1.text(.005, .97,
            "All rules causal: the decision at each pass uses only that pass and\n"
            "the ones before it. No sliding window peeks at the future.",
            transform=a1.transAxes, va="top", fontsize=7, color=V.GREY)
    V.false_alarm_bars(a2, df, "persistence")
    a2.set_xlabel("Consecutive exceedances required")
    V.save(fig, "fig07_persistence")


def fig_limits():
    df = load("e6_detection_limits")
    # severity arrives either as "40%" or as the fraction 0.4 depending on
    # whether the table has been through the cleanup pass
    sev = df["severity"].astype(str).str.rstrip("%").astype(float)
    df["sev"] = np.where(sev <= 1.0, sev * 100, sev).round().astype(int)
    df["dur"] = (df["duration"].astype(str).str.replace(" obs", "", regex=False)
                 .astype(float).round().astype(int))
    piv = df.pivot(index="dur", columns="sev", values="f1")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    data = np.ma.masked_invalid(piv.values)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#EDEDED")
    vmin, vmax = float(np.nanmin(piv.values)), float(np.nanmax(piv.values))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            rel = (v - vmin) / (vmax - vmin) if vmax > vmin else .5
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white" if rel < .55 else "#101010")
    ax.set_xticks(range(piv.shape[1]), [f"{c}%" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{r} obs" for r in piv.index])
    ax.set_xlabel("Stress severity"); ax.set_ylabel("Duration")
    ax.set_title("Detection limits", loc="left", pad=20)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0], 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.set_label("F1", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    V.save(fig, "fig08_detection_limits")


def fig_protocols():
    """How much of a result is protocol, and how much is model."""
    df = load("e8_protocols")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    V.metric_bars(ax, df, "protocol", highlight=df.loc[0, "protocol"])
    ax.set_title("Evaluation protocol changes the answer", loc="left", pad=20)
    ax.set_xlabel("")
    V.legend_above(ax)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(7.5)
    fig.text(0, -.04,
             "The temporal protocol mirrors deployment: the detector has the field's own "
             "history and is asked\nabout a later season. The cross-field protocol is "
             "stricter than deployment — it withholds that history\nentirely. Reporting "
             "both makes clear how much of any published number comes from the\n"
             "evaluation design rather than the model.",
             fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig14_protocols")


def fig_anomaly_types():
    df = load("e7_anomaly_types")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    V.metric_bars(ax, df, "signature")
    ax.set_title("Robustness to stress signature", loc="left", pad=20)
    ax.set_xlabel("Injected signature")
    V.legend_above(ax)
    ax.text(.005, .97,
            "Each signature is injected coherently across all four indices, in the\n"
            "proportions the underlying biology implies.",
            transform=ax.transAxes, va="top", fontsize=7, color=V.GREY)
    V.save(fig, "fig09_anomaly_types")


# ── figures that need the scored frame ────────────────────────────────
def _scored():
    p = os.path.join(OUT_DIR, "final_scored.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, parse_dates=["date"])


def fig_separation(sc):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    lo, hi = np.percentile(sc["score"], [0.2, 99.8])
    bins = np.linspace(lo, hi, 70)
    ax.hist(sc.loc[sc.stressed == 0, "score"], bins=bins, color=V.BLUE, alpha=.75,
            density=True, label=f"Healthy (n={int((sc.stressed==0).sum()):,})")
    ax.hist(sc.loc[sc.stressed == 1, "score"], bins=bins, color=V.VERM, alpha=.75,
            density=True, label=f"Stressed (n={int((sc.stressed==1).sum()):,})")
    ax.set_xlabel("LOF anomaly score"); ax.set_ylabel("Density")
    ax.set_title("Score separability on held-out fields", loc="left", pad=20)
    ax.legend(loc="upper right")
    V.save(fig, "fig10_score_separation")


def fig_pr_roc(sc):
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    y, s = sc["stressed"].to_numpy(), sc["score"].to_numpy()
    if len(np.unique(y)) < 2:
        return
    pr, rc, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    ap, ra = average_precision_score(y, s), roc_auc_score(y, s)
    from sklearn.metrics import precision_score, recall_score
    op_p = precision_score(y, sc["alarm"], zero_division=0)
    op_r = recall_score(y, sc["alarm"], zero_division=0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    a1.plot(rc, pr, color=V.BLUE, lw=1.8)
    a1.scatter([op_r], [op_p], color=V.VERM, s=55, zorder=5, label="Operating point")
    a1.annotate(f"({op_r:.2f}, {op_p:.2f})", (op_r, op_p), fontsize=7.5,
                color=V.VERM, xytext=(8, -12), textcoords="offset points")
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision")
    a1.set_title(f"A   Precision–recall (AP = {ap:.3f})", loc="left")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.02); a1.legend(loc="lower left", fontsize=7.5)
    a2.plot(fpr, tpr, color=V.BLUE, lw=1.8)
    a2.plot([0, 1], [0, 1], ls=":", color=V.GREY, lw=1)
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_title(f"B   ROC (AUC = {ra:.3f})", loc="left")
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.02)
    fig.text(0, -.04,
             "Held-out fields only. Average precision is the informative summary here; "
             "ROC-AUC is flattered by\nthe large healthy majority.",
             fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig11_pr_roc")


def fig_confusion(sc):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(sc["stressed"], sc["alarm"], labels=[0, 1])
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}\n{row[i,j]*100:.1f}%", ha="center",
                    va="center", fontsize=10, fontweight="bold",
                    color="white" if row[i, j] > .55 else "#101010")
    ax.set_xticks([0, 1], ["Predicted\nhealthy", "Predicted\nstressed"])
    ax.set_yticks([0, 1], ["Actual\nhealthy", "Actual\nstressed"])
    ax.set_title("Confusion matrix, held-out fields", loc="left", pad=20)
    ax.grid(False)
    V.save(fig, "fig12_confusion")


def fig_trajectory(sc):
    """One field, end to end: signal, score, threshold, alarms."""
    cand = (sc[sc.stressed == 1].groupby("tile_id")
            .agg(hit=("alarm", "sum"), n=("alarm", "size")))
    cand = cand[(cand.hit >= 2)]
    if cand.empty:
        print("    (skipping trajectory: no field with >=2 detected observations)")
        return
    tile = int(cand.index[0])
    g = sc[sc.tile_id == tile].sort_values("date")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                 gridspec_kw=dict(hspace=.16))
    a1.plot(g.date, g.ndvi_sm, color=V.GREEN, lw=1.8, marker="o", ms=3,
            mfc="white", label="NDVI (smoothed)")
    if "ndmi_sm" in g:
        a1.plot(g.date, g.ndmi_sm, color=V.SKY, lw=1.5, ls="--", marker="o", ms=2.5,
                mfc="white", label="NDMI (moisture)")
    inj = g.stressed == 1
    a1.scatter(g.date[inj], g.ndvi_sm[inj], color=V.VERM, s=40, zorder=5,
               label="Injected stress")
    lo, hi = float(min(g.ndvi_sm.min(), g.ndmi_sm.min())), float(g.ndvi_sm.max())
    a1.set_ylim(lo - .1 * (hi - lo), hi + .45 * (hi - lo))
    a1.set_ylabel("Index value")
    a1.set_title(f"A   Vegetation and moisture — field {tile}", loc="left")
    a1.legend(loc="upper left", ncols=3)

    a2.plot(g.date, g.score, color=V.BLUE, lw=1.8, marker="o", ms=3, mfc="white",
            label="LOF score")
    raw = g.raw_flag == 1
    al = g.alarm == 1
    a2.scatter(g.date[raw & ~al], g.score[raw & ~al], facecolors="none",
               edgecolors=V.ORANGE, s=70, lw=1.4, zorder=4,
               label="Exceeded, not confirmed")
    a2.scatter(g.date[al], g.score[al], color=V.VERM, s=70, marker="X", zorder=5,
               label=f"Alarm ({DECIDE['min_consecutive']} consecutive)")
    lo2, hi2 = float(g.score.min()), float(g.score.max())
    a2.set_ylim(lo2 - .08 * (hi2 - lo2), hi2 + .45 * (hi2 - lo2))
    a2.set_ylabel("Anomaly score"); a2.set_xlabel("Observation date")
    a2.set_title("B   Score, exceedances and confirmed alarms", loc="left")
    a2.legend(loc="upper left", ncols=3)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.text(0, -.03,
             "This field is in the held-out set — the detector was never fitted on it, "
             "and its threshold was\nlearned from its own earlier, clean observations.",
             fontsize=7, color=V.GREY, va="top")
    V.save(fig, "fig13_trajectory")


def main():
    print(f"Writing figures to {os.path.join(OUT_DIR, 'figures')}\n")
    print("From results tables:")
    fig_pipeline()
    try:
        fig_spectral_signature()
    except Exception as e:
        print(f"    (spectral signature skipped: {e})")
    for fn in (fig_index_ablation, fig_models, fig_neighbors, fig_threshold,
               fig_persistence, fig_limits, fig_anomaly_types, fig_protocols):
        try:
            fn()
        except Exception as e:
            print(f"    ({fn.__name__} skipped: {e})")

    sc = _scored()
    if sc is None:
        print("\n(no final_scored.csv — run run_experiments.py first)")
        return
    print("\nFrom the scored evaluation set:")
    for fn in (fig_separation, fig_pr_roc, fig_confusion, fig_trajectory):
        try:
            fn(sc)
        except Exception as e:
            print(f"    ({fn.__name__} skipped: {e})")
    print("\nDone.")


if __name__ == "__main__":
    main()
