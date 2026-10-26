"""make_figures.py — the paper figures, all from outputs/results.json."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core as C

BLUE, GREEN, AMBER, RED, GREY = "#2A6FB5", "#2E8B6E", "#D2921F", "#C0453B", "#8A8F98"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 400, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
})

METRICS = [("precision", "Precision", BLUE), ("recall", "Recall", GREEN),
           ("f1", "F1", AMBER)]


def titled(ax, title, subtitle=None):
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


def bars(ax, labels, data, highlight=None, note=True):
    x = np.arange(len(labels))
    w = 0.26
    for i, (key, lab, col) in enumerate(METRICS):
        v = [d[key] for d in data]
        e = [d.get(key + "_sd", 0) for d in data]
        ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=2.5, color=col, label=lab,
               error_kw=dict(lw=0.8, ecolor="#444"))
        if note:
            for xi, vv in zip(x + (i - 1) * w, v):
                ax.text(xi, vv + 0.018, f"{vv:.3f}", ha="center", va="bottom",
                        fontsize=6.4, color="#333")
    if highlight is not None:
        ax.axvspan(highlight - .5, highlight + .5, color=AMBER, alpha=.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Score")
    ax.legend(ncol=3, loc="upper left")


def fa_panel(ax, labels, data, log=False):
    x = np.arange(len(labels))
    fa = [d["false_alarms"] for d in data]
    ax.bar(x, fa, 0.5, color=GREY)
    for xi, v in zip(x, fa):
        # round, don't truncate: these are means over seeds, and the tables in
        # run_all.py round. Truncating here made figures and tables disagree
        # (33 vs 34, 206 vs 207), which contradicts the paper's reproducibility
        # claim that plotted and reported numbers cannot diverge.
        ax.text(xi, v, f"{int(round(v)):,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("False alarms")
    if log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, max(fa) * 1.28)


def _wrap(s, n=18):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════
def fig1_operating_point(R):
    a = R["a"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6),
                                   gridspec_kw=dict(wspace=0.28))
    p = [r["pctl"] for r in a["pctl_curve"]]
    ax1.plot(p, [r["f1"] for r in a["pctl_curve"]], color=AMBER, lw=2, label="F1")
    ax1.plot(p, [r["precision"] for r in a["pctl_curve"]], color=BLUE, lw=1.1,
             label="Precision")
    ax1.plot(p, [r["recall"] for r in a["pctl_curve"]], color=GREEN, lw=1.1,
             label="Recall")
    ax1.axvline(a["p_star"], color=RED, ls=":", lw=1.3)
    ax1.axvline(C.LEGACY_PCTL, color=GREY, ls=":", lw=1.1)
    ax1.annotate(f"Study 2\nP{C.LEGACY_PCTL}", xy=(C.LEGACY_PCTL, 0.04),
                 xytext=(3, 0), textcoords="offset points", fontsize=7,
                 color=GREY, va="bottom")
    ax1.annotate(f"P{a['p_star']:g}", xy=(a["p_star"], 0.04), xytext=(-24, 0),
                 textcoords="offset points", fontsize=7, color=RED, va="bottom")
    ax1.set_xlabel("Clean-data percentile")
    ax1.set_ylabel("Score (calibration split)")
    ax1.set_ylim(0, 1.02)
    ax1.legend()
    titled(ax1, "Percentile rule")

    q = [r["q"] for r in a["evt_curve"]]
    ax2.plot(q, [r["f1"] for r in a["evt_curve"]], "o-", color=AMBER, lw=1.8,
             ms=4, label="F1")
    ax2.plot(q, [r["precision"] for r in a["evt_curve"]], "^-", color=BLUE,
             lw=1.1, ms=3.5, label="Precision")
    ax2.plot(q, [r["recall"] for r in a["evt_curve"]], "v-", color=GREEN,
             lw=1.1, ms=3.5, label="Recall")
    ax2.axvline(a["q_star"], color=RED, ls=":", lw=1.3)
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.set_xlabel("EVT risk q (stricter →)")
    ax2.set_ylim(0, 1.02)
    ax2.legend()
    titled(ax2, f"EVT rule (ξ = {a['xi']:.3f})")
    fig.suptitle("Choosing the operating point on clean, held-out calibration data",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.10)
    save(fig, "fig1_operating_point")


def fig2_threshold_rules(R):
    res = R["a"]["results"]
    keys = list(res)
    labels = [_wrap(k.replace("Study 2 rule: ", "Study 2: "), 20) for k in keys]
    data = [res[k] for k in keys]
    best = max(range(len(keys)), key=lambda i: data[i]["f1"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.3, 1],
                                                  hspace=0.12))
    bars(a1, labels, data, highlight=best)
    titled(a1, "Threshold rule decides the operating point",
           "Same detector, same fields, same injections, same persistence rule "
           "— only the threshold differs.\nThe first row is what Study 2 used. "
           f"Mean ± s.d. over {len(C.SEEDS)} seeds on held-out fields.")
    fa_panel(a2, labels, data, log=True)
    a2.set_xlabel("Threshold rule")
    save(fig, "fig2_threshold_rules")


def fig3_detectors(R):
    keys = list(R["b"])
    data = [R["b"][k] for k in keys]
    labels = [_wrap(k, 14) for k in keys]
    best = max(range(len(keys)), key=lambda i: data[i]["f1"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.3, 1],
                                                  hspace=0.12))
    bars(a1, labels, data, highlight=best)
    lof, hyb = R["b"]["LOF"]["f1"], R["b"]["Hybrid (all 3)"]["f1"]
    titled(a1, "Ensembling the detectors makes them worse",
           f"Standalone LOF reaches F1 {lof:.3f}; the three-model hybrid reaches "
           f"{hyb:.3f}, a factor of {lof/max(hyb,1e-9):.1f}.\nScore-level "
           "averaging drags a strong ranking towards the weaker members. Each "
           "detector gets its own\nclean-data threshold, since the scores are "
           "on different scales.")
    fa_panel(a2, labels, data, log=True)
    a2.set_xlabel("Detector")
    save(fig, "fig3_detectors")


def fig4_baselines(R):
    keys = list(R["c"])
    data = [R["c"][k] for k in keys]
    labels = [_wrap(k, 14) for k in keys]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.3, 1],
                                                  hspace=0.12))
    bars(a1, labels, data, highlight=0)
    titled(a1, "The learned detector against rules you could write by hand",
           "Identical data, identical threshold source and persistence rule. "
           "Only the scoring function differs.")
    fa_panel(a2, labels, data, log=True)
    a2.set_xlabel("Scoring method")
    save(fig, "fig4_baselines")


def fig5_limits(R):
    rows = R["e"]
    sev = sorted({r["severity"] for r in rows})
    dur = sorted({r["duration"] for r in rows})
    M = np.full((len(dur), len(sev)), np.nan)
    for r in rows:
        M[dur.index(r["duration"]), sev.index(r["severity"])] = r["f1"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(len(dur)):
        for j in range(len(sev)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if M[i, j] > 0.55 else "#222")
            else:
                ax.text(j, i, "—", ha="center", va="center", color=GREY)
    ax.set_xticks(range(len(sev)))
    ax.set_xticklabels([f"{int(round(s*100))}%" for s in sev])
    ax.set_yticks(range(len(dur)))
    ax.set_yticklabels([f"{d} obs" for d in dur])
    ax.set_xlabel("Stress severity")
    ax.set_ylabel("Duration")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="F1", shrink=0.85)
    titled(ax, "Detection limits",
           "Where the method works and where it stops working. "
           "Grey cells were not evaluated.")
    save(fig, "fig5_limits")


def fig6_persistence(R):
    keys = list(R["d"])
    data = [R["d"][k] for k in keys]
    labels = [_wrap(k, 14) for k in keys]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.3, 1],
                                                  hspace=0.12))
    bars(a1, labels, data)
    titled(a1, "How many consecutive passes before an alarm fires",
           "Persistence trades recall for precision. All four rules are causal "
           "— the sliding window inspects\n[t-2, t-1, t], not the future.")
    fa_panel(a2, labels, data, log=True)
    a2.set_xlabel("Persistence rule")
    save(fig, "fig6_persistence")


def fig7_signatures(R):
    keys = list(R["f"])
    data = [R["f"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars(ax, keys, data)
    worst = min(keys, key=lambda k: R["f"][k]["f1"])
    titled(ax, "Robustness to the shape of the stress event",
           f"{worst} is the weak case — the decline accumulates slowly enough "
           "to be absorbed by the\nfield's own rolling baseline.")
    ax.set_xlabel("Injected signature")
    save(fig, "fig7_signatures")


def fig8_kneighbors(R):
    g = R["g"]
    ks = [r["k"] for r in g["test"]]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(ks, [r["f1"] for r in g["cal"]], "o-", color=AMBER, lw=1.8, ms=4,
            label="F1 (calibration)")
    ax.plot(ks, [r["f1"] for r in g["test"]], "s--", color=AMBER, lw=1.3, ms=4,
            alpha=.75, label="F1 (test)")
    ax.plot(ks, [r["precision"] for r in g["test"]], "^-", color=BLUE, lw=1.1,
            ms=3.5, label="Precision (test)")
    ax.plot(ks, [r["recall"] for r in g["test"]], "v-", color=GREEN, lw=1.1,
            ms=3.5, label="Recall (test)")
    ax.axvline(g["k_default"], color=GREY, ls=":", lw=1.1)
    ax.set_xlabel("LOF n_neighbors")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2)
    f1s = [r["f1"] for r in g["test"]]
    titled(ax, "LOF neighbourhood size is not load-bearing",
           f"Test F1 spans {max(f1s)-min(f1s):.3f} across k ∈ [{min(ks)}, "
           f"{max(ks)}]. The default k={g['k_default']} is marked.")
    save(fig, "fig8_kneighbors")


def fig9_protocol(R):
    keys = list(R["h"])
    data = [R["h"][k] for k in keys]
    labels = [_wrap(k, 22) for k in keys]
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    bars(ax, labels, data)
    d = data[0]["f1"] - data[1]["f1"]
    titled(ax, "What held-out evaluation actually costs",
           f"Fitting on a disjoint half of the fields instead of on all of them "
           f"costs {d:.3f} F1 — not the large\ndrop that comparing across two "
           "codebases suggested. Both arms use Study 2's own threshold rule.")
    ax.set_xlabel("Evaluation protocol")
    save(fig, "fig9_protocol")


def main():
    with open(os.path.join(C.OUT_DIR, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    print("Rendering figures...")
    for fn in (fig1_operating_point, fig2_threshold_rules, fig3_detectors,
               fig4_baselines, fig5_limits, fig6_persistence, fig7_signatures,
               fig8_kneighbors, fig9_protocol):
        fn(R)
    # Figures 10-16 need the illustrative data that extras.py computes.
    if "extras" in R:
        import figures_extra
        figures_extra.render(R)
    else:
        print("  (no 'extras' in results.json — run extras.py for figs 10-16)")


if __name__ == "__main__":
    main()
