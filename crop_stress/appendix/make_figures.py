"""
make_figures.py — every figure in experiment4/outputs/figures/.

Reads outputs/results.json, so figures can be re-rendered without re-running
the experiments, and the plotted numbers cannot drift from the reported ones.
Each figure is written as a 400 dpi PNG and as a vector PDF.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common as C

# ── House style ──────────────────────────────────────────────────────
BLUE, GREEN, AMBER, RED, GREY = "#2A6FB5", "#2E8B6E", "#D2921F", "#C0453B", "#8A8F98"
LIGHT = "#E7E9EC"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

METRICS = [("precision", "Precision", BLUE),
           ("recall", "Recall", GREEN),
           ("f1", "F1", AMBER)]

NICE = {
    "baseline_p85_postinj": "Study 2 baseline\n(P85, post-injection)",
    "clean_p85": "Clean-calibrated\nP85",
    "clean_pstar": "Clean-calibrated\nP{p}",
}


def titled(ax, title, subtitle=None):
    """
    Title plus a grey explanatory subtitle, spaced so they never collide.

    The subtitle sits just above the axes in axes coordinates, so the title's
    pad has to grow with the number of subtitle lines.
    """
    n = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(title, loc="left", pad=13 + 10.5 * n)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=7.2,
                color=GREY, va="bottom", linespacing=1.35)


def save(fig, name):
    C.ensure_dirs()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(C.FIG_DIR, f"{name}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}.png + .pdf")


def _bars(ax, labels, data, highlight=None):
    """Grouped precision/recall/F1 bars with s.d. whiskers."""
    x = np.arange(len(labels))
    w = 0.26
    for i, (key, lab, col) in enumerate(METRICS):
        vals = [d[key] for d in data]
        errs = [d.get(key + "_sd", 0.0) for d in data]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2.5,
               color=col, label=lab,
               error_kw=dict(lw=0.8, ecolor="#444"))
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.6, color="#333")
    if highlight is not None:
        ax.axvspan(highlight - 0.5, highlight + 0.5, color=AMBER, alpha=0.07,
                   zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="upper left")


def _fa_panel(ax, labels, data):
    x = np.arange(len(labels))
    fa = [d["false_alarms"] for d in data]
    ax.bar(x, fa, 0.5, color=GREY)
    for xi, v in zip(x, fa):
        ax.text(xi, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("False alarms")
    ax.set_ylim(0, max(fa) * 1.25)


# ══════════════════════════════════════════════════════════════════════
def fig1_threshold_selection(R):
    e1 = R["e1"]
    cal = e1["cal_curve"]
    test = e1["test_curve"]
    p = [r["pctl"] for r in cal]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1],
                                                    hspace=0.14))
    ax1.plot(p, [r["f1"] for r in cal], color=AMBER, lw=2,
             label="F1 (calibration split)")
    ax1.plot(p, [r["f1"] for r in test], color=AMBER, lw=1.4, ls="--",
             alpha=0.75, label="F1 (test split)")
    ax1.plot(p, [r["precision"] for r in cal], color=BLUE, lw=1.2, alpha=0.85,
             label="Precision (cal)")
    ax1.plot(p, [r["recall"] for r in cal], color=GREEN, lw=1.2, alpha=0.85,
             label="Recall (cal)")

    ps = e1["p_star"]
    ax1.axvline(ps, color=RED, lw=1.3, ls=":")
    ax1.annotate(f"chosen on calibration split\nP{ps:g}",
                 xy=(ps, 0.06), xytext=(6, 0), textcoords="offset points",
                 fontsize=7.5, color=RED, va="bottom")
    ax1.axvline(C.BASELINE_PCTL, color=GREY, lw=1.1, ls=":")
    ax1.annotate(f"Study 2\nP{C.BASELINE_PCTL}", xy=(C.BASELINE_PCTL, 0.06),
                 xytext=(-34, 0), textcoords="offset points",
                 fontsize=7.5, color=GREY, va="bottom")

    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.02)
    titled(ax1, "Choosing the threshold on clean data",
           "Threshold is each field's own percentile of its CLEAN scores. "
           "The percentile itself is chosen on the\ncalibration split, so the "
           "test curve is shown for reference only — nothing is selected on it.")
    ax1.legend(ncol=2, loc="center left")

    ax2.plot(p, [r["false_alarms"] for r in cal], color=GREY, lw=1.6)
    ax2.axvline(ps, color=RED, lw=1.3, ls=":")
    ax2.axvline(C.BASELINE_PCTL, color=GREY, lw=1.1, ls=":")
    ax2.set_yscale("log")
    ax2.set_ylabel("False alarms\n(calibration)")
    ax2.set_xlabel("Per-field threshold percentile, computed on clean scores")
    save(fig, "fig1_threshold_selection")


def fig2_headline(R):
    e1 = R["e1"]["results"]
    ps = R["e1"]["p_star"]
    keys = ["baseline_p85_postinj", "clean_p85", "clean_pstar"]
    labels = [NICE[k].replace("{p}", f"{ps:g}") for k in keys]
    data = [e1[k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.3, 1],
                                                    hspace=0.12))
    _bars(ax1, labels, data, highlight=2)
    titled(ax1, "Held-out test fields: what the clean-calibrated threshold buys",
           "All three evaluated on the same held-out fields with the same "
           "injections and the same persistence rule.\nOnly the threshold "
           f"differs. Mean ± s.d. over {len(C.SEEDS)} seeds.")
    _fa_panel(ax2, labels, data)
    ax2.set_xlabel("Threshold rule")
    save(fig, "fig2_headline_comparison")


def fig3_point_vs_event(R):
    e1, e2 = R["e1"]["results"], R["e2"]
    ps = R["e1"]["p_star"]
    keys = ["baseline_p85_postinj", "clean_p85", "clean_pstar"]
    labels = [NICE[k].replace("{p}", f"{ps:g}") for k in keys]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(keys))
    w = 0.2
    point_r = [e1[k]["recall"] for k in keys]
    point_p = [e1[k]["precision"] for k in keys]
    rng_r = [e2[f"{k}@alpha0.5"]["range_recall"] for k in keys]
    rng_p = [e2[f"{k}@alpha0.5"]["range_precision"] for k in keys]

    for i, (vals, lab, col, hatch) in enumerate([
            (point_p, "Precision (point-wise)", BLUE, None),
            (rng_p, "Precision (event-level)", BLUE, "///"),
            (point_r, "Recall (point-wise)", GREEN, None),
            (rng_r, "Recall (event-level)", GREEN, "///")]):
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab,
               alpha=1.0 if hatch is None else 0.55,
               hatch=hatch, edgecolor="white", linewidth=0.6)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.3, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(ncol=2, loc="upper left")
    titled(ax, "Observation-wise versus event-level scoring",
           "Identical predictions, two metrics. Event-level scoring (Tatbul et "
           "al., α=0.5) asks whether the event was\ncaught at all rather than "
           "whether every observation of it was flagged. The two nearly "
           "coincide here:\npersistence requires 2 consecutive exceedances and "
           "events last 2 observations, so detection is\nall-or-nothing and "
           "there is almost no partial credit to award.")
    save(fig, "fig3_point_vs_event")


def fig4_peer_feature(R):
    e3 = R["e3"]
    labels = ["16 features\n(Study 2)", "17 features\n(+ same-date peer dev.)"]
    data = [e3["base_16"], e3["peer_17"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.6),
                                   gridspec_kw=dict(width_ratios=[2.1, 1],
                                                    wspace=0.32))
    _bars(ax1, labels, data)
    titled(ax1, "Adding a same-date peer-deviation feature",
             "Deviation from the median of all fields seen on the SAME DATE, "
             "which cancels region-wide haze\nthat persistence cannot filter.")
    _fa_panel(ax2, ["16", "17"], data)
    ax2.set_xlabel("Features")
    ax2.set_title("False alarms", loc="left", pad=16)
    save(fig, "fig4_peer_feature")


def fig5_kneighbors(R):
    e4 = R["e4"]
    ks = [r["k"] for r in e4["test"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 4.8), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1],
                                                    hspace=0.14))
    ax1.plot(ks, [r["f1"] for r in e4["cal"]], "o-", color=AMBER, lw=1.8,
             ms=4, label="F1 (calibration)")
    ax1.plot(ks, [r["f1"] for r in e4["test"]], "s--", color=AMBER, lw=1.4,
             ms=4, alpha=0.75, label="F1 (test)")
    ax1.plot(ks, [r["precision"] for r in e4["test"]], "^-", color=BLUE, lw=1.1,
             ms=3.5, alpha=0.8, label="Precision (test)")
    ax1.plot(ks, [r["recall"] for r in e4["test"]], "v-", color=GREEN, lw=1.1,
             ms=3.5, alpha=0.8, label="Recall (test)")
    ax1.axvline(e4["k_default"], color=GREY, ls=":", lw=1.1)
    ax1.annotate(f"Study 2 default\nk={e4['k_default']}",
                 xy=(e4["k_default"], 0.05), xytext=(6, 0),
                 textcoords="offset points", fontsize=7.5, color=GREY,
                 va="bottom")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.02)
    ax1.legend(ncol=2)
    titled(ax1, "LOF neighbourhood size — a limitation the main study left open",
           "The main study fixed k=10 and never swept it. Performance is nearly "
           "flat, so the choice was not\nload-bearing.")

    ax2.plot(ks, [r["false_alarms"] for r in e4["test"]], "o-", color=GREY,
             lw=1.5, ms=4)
    ax2.set_ylabel("False alarms\n(test)")
    ax2.set_xlabel("LOF n_neighbors")
    save(fig, "fig5_kneighbors")


def fig6_significance(R):
    e5 = R["e5"]
    boot = np.array(e5["boot"])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(boot, bins=55, color=LIGHT, edgecolor=GREY, linewidth=0.5)
    ax.axvline(0, color=RED, lw=1.6, label="no difference")
    ax.axvline(e5["delta_f1"], color=BLUE, lw=1.6,
               label=f"observed ΔF1 = {e5['delta_f1']:+.4f}")
    lo, hi = e5["ci95"]
    ax.axvspan(lo, hi, color=BLUE, alpha=0.12,
               label=f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    ax.set_xlabel("F1(adaptive P85) − F1(global P90)")
    ax.set_ylabel("Bootstrap resamples")
    ax.legend(loc="upper left")
    verdict = ("spans zero — the two rules are not distinguishable"
               if lo <= 0 <= hi else "excludes zero — the difference is real")
    titled(ax, "Is adaptive P85 actually better than global P90?",
            f"Study 2 chose between them on an F1 gap of 0.006 with no test. "
            f"Bootstrap over {e5['n_units']:,} field-seed\nunits, "
            f"{e5['n_boot']:,} resamples: the interval {verdict} "
            f"(permutation p = {e5['p_value']:.3f}).")
    save(fig, "fig6_significance")


def fig7_evt(R):
    e6 = R.get("e6")
    if not e6:
        return
    qs = [r["q"] for r in e6["cal_curve"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                   gridspec_kw=dict(width_ratios=[1.25, 1],
                                                    wspace=0.3))
    ax1.plot(qs, [r["f1"] for r in e6["cal_curve"]], "o-", color=AMBER, lw=1.8,
             ms=4, label="F1")
    ax1.plot(qs, [r["precision"] for r in e6["cal_curve"]], "^-", color=BLUE,
             lw=1.1, ms=3.5, label="Precision")
    ax1.plot(qs, [r["recall"] for r in e6["cal_curve"]], "v-", color=GREEN,
             lw=1.1, ms=3.5, label="Recall")
    ax1.axvline(e6["q_star"], color=RED, ls=":", lw=1.3)
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("Risk q (log scale, stricter →)")
    ax1.set_ylabel("Score (calibration split)")
    ax1.set_ylim(0, 1.02)
    ax1.legend()
    ax1.set_title("EVT threshold: sweeping the risk", loc="left", pad=14)

    names = ["Clean-calibrated\npercentile", "EVT / GPD tail"]
    vals = [R["e1"]["results"]["clean_pstar"], e6["test"]]
    x = np.arange(2)
    for i, (key, lab, col) in enumerate(METRICS):
        ax2.bar(x + (i - 1) * 0.26, [v[key] for v in vals], 0.26, color=col,
                label=lab)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.12)
    ax2.legend(ncol=3, fontsize=7)
    ax2.set_title(f"On test (ξ = {e6['xi']:.3f})", loc="left", pad=14)
    save(fig, "fig7_evt")


def main():
    with open(os.path.join(C.OUT_DIR, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    print("Rendering figures...")
    fig1_threshold_selection(R)
    fig2_headline(R)
    fig3_point_vs_event(R)
    fig4_peer_feature(R)
    fig5_kneighbors(R)
    fig6_significance(R)
    fig7_evt(R)


if __name__ == "__main__":
    main()
