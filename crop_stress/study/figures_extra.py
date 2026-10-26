"""
figures_extra.py — figures 10 to 16, the illustrative half of the paper.

The nine figures in make_figures.py prove the results. These show the method
and the data: what the pipeline does, what one field's decision path looks
like including its failure, what the input imagery actually is, and the
threshold-free evidence that the ranking is good rather than the cut lucky.

Driven by the "extras" block that extras.py appends to results.json.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import core as C
from make_figures import BLUE, GREEN, AMBER, RED, GREY, titled, save


# ══════════════════════════════════════════════════════════════════════
def fig10_pipeline(R):
    """The method as a flow, including the three-way split."""
    s = R["setup"]
    total = sum(s["n_tiles"].values())
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.6)
    ax.axis("off")
    ax.grid(False)

    steps = [
        ("Sentinel-2 MSAVI rasters",
         f"{total:,} fields, 2019 season", BLUE),
        ("Screening",
         "reject cloud and fill;\nkeep 25 passes ≥ 10 days apart", BLUE),
        ("Per-field statistics",
         "mask nodata, clip 1st/99th pct,\nsummarise over the field", BLUE),
        ("Temporal features",
         f"rolling stats, differences,\nSavitzky–Golay → {s['n_features']} features",
         BLUE),
        ("Disjoint field split",
         f"fit {s['n_tiles']['fit']:,}  |  cal {s['n_tiles']['cal']:,}  |  "
         f"test {s['n_tiles']['test']:,}", AMBER),
        ("LOF, novelty mode",
         f"fitted on CLEAN fit-split rows only\nk = {s['k_default']}", GREEN),
        ("Threshold calibration",
         "per-field GPD tail on CLEAN scores;\nrisk q chosen on the cal split",
         GREEN),
        ("Persistence",
         f"{s['min_consecutive']} consecutive exceedances\nrequired to confirm",
         GREEN),
        ("Field-level alarm",
         "one decision per field per pass", RED),
    ]
    h, gap = 1.28, 0.42
    y = 15.6 - h
    for i, (title, sub, col) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((0.7, y), 8.6, h,
                                    boxstyle="round,pad=0.06", linewidth=1.3,
                                    edgecolor=col, facecolor=col, alpha=0.11))
        ax.text(5.0, y + h * 0.66, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1a1a1a")
        ax.text(5.0, y + h * 0.26, sub, ha="center", va="center",
                fontsize=7, color="#444", linespacing=1.3)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - gap),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.1, color=GREY))
        y -= (h + gap)
    ax.text(0.7, 15.62, "No label enters at any stage; every threshold is "
                        "calibrated on clean data",
            fontsize=7.4, color=GREY, style="italic", va="bottom")
    ax.set_title("Processing pipeline", loc="left", pad=18)
    save(fig, "fig10_pipeline")


def fig11_worked_example(R):
    """One held-out field, start to finish, including what it misses."""
    ex = R["extras"].get("example")
    if not ex:
        return
    n = len(ex["scores"])
    x = np.arange(n)
    gt = np.array(ex["gt"], dtype=bool)
    conf = np.array(ex["confirmed"], dtype=bool)
    sup = np.array(ex["suppressed"], dtype=bool)
    sc = np.array(ex["scores"], dtype=float)

    # Wide and short on purpose. This is the paper's only full-width float, and
    # in twocolumn LaTeX a tall figure* eats over half a page, which makes it
    # unplaceable and strands it in the bibliography. A 25-point series wants
    # width anyway. The per-field outcome counts live in the caption rather
    # than in a panel subtitle, which is what allowed the height to come down.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 4.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.2],
                                                  hspace=0.22))
    a1.plot(x, ex["clean_p50"], color=GREY, lw=1.2, ls="--", label="clean MSAVI")
    a1.plot(x, ex["injected_p50"], color=BLUE, lw=1.8, marker="o", ms=3,
            label="injected MSAVI")
    for i in np.flatnonzero(gt):
        a1.axvspan(i - .5, i + .5, color=RED, alpha=.10, zorder=0)
    a1.set_ylabel("Field median MSAVI")
    a1.legend(ncol=2, loc="lower left", fontsize=7.2)
    a1.set_title(f"(A)  Field {ex['tile']} across the season — shaded bands are "
                 f"the injected stress events", loc="left", pad=6)

    a2.plot(x, sc, color="#333", lw=1.5, marker="o", ms=3, zorder=2,
            label="LOF score")
    a2.axhline(ex["threshold"], color=RED, ls="--", lw=1.2,
               label=f"this field's threshold ({ex['threshold']:.2f})")
    for i in np.flatnonzero(gt):
        a2.axvspan(i - .5, i + .5, color=RED, alpha=.10, zorder=0)
    if sup.any():
        a2.scatter(x[sup], sc[sup], s=95, facecolors="none", edgecolors=AMBER,
                   linewidths=1.8, zorder=4, label="exceeded, suppressed")
    if conf.any():
        a2.scatter(x[conf], sc[conf], marker="X", s=62, color=GREEN, zorder=5,
                   label="confirmed alarm")
    a2.set_ylabel("LOF anomaly score")
    a2.set_xlabel("Satellite pass")
    # Headroom so the single-row legend clears the tallest marker.
    lo, hi = float(sc.min()), float(sc.max())
    a2.set_ylim(lo - 0.04 * (hi - lo), hi + 0.30 * (hi - lo))
    a2.legend(ncol=4, loc="upper left", fontsize=7.2)
    a2.set_title("(B)  Score, threshold and decision", loc="left", pad=6)
    save(fig, "fig11_worked_example")


def fig12_imagery(R):
    """What the input actually looks like. A data figure, not a results figure."""
    im = R["extras"].get("imagery")
    if not im:
        return
    n = len(im["panels"])
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 4.2),
                             gridspec_kw=dict(height_ratios=[3, 1.5],
                                              hspace=0.35, wspace=0.08))
    for j, (p, d) in enumerate(zip(im["panels"], im["dates"])):
        ax = axes[0, j]
        ax.imshow(np.array(p, dtype=float), cmap="YlGn",
                  vmin=im["vmin"], vmax=im["vmax"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(d[5:], fontsize=7.5, pad=3, fontweight="normal")
    axes[0, 0].set_ylabel("MSAVI", fontsize=8)

    gs = axes[1, 0].get_gridspec()
    for ax in axes[1, :]:
        ax.remove()
    axb = fig.add_subplot(gs[1, :])
    axb.plot(range(n), im["means"], "o-", color=GREEN, lw=1.6, ms=4)
    axb.set_xticks(range(n))
    axb.set_xticklabels([d[5:] for d in im["dates"]], fontsize=7)
    axb.set_ylabel("Field mean")
    axb.set_xlabel("Acquisition date (2019)")

    fig.suptitle(f"The input: field {im['tile']} through the 2019 season",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.06)
    fig.text(0.005, 1.005,
             f"{im['n_available']} cloud-screened passes, {n} shown, on a shared "
             "2nd–98th percentile stretch. The detector consumes the\n"
             "field-level statistics of these rasters, not the pixels. No "
             "detections are drawn: the synthetic stress lives in the\n"
             "feature table, not in this imagery, so marking alarms here would "
             "imply the model saw something it did not.",
             fontsize=7.2, color=GREY, va="top", ha="left", linespacing=1.35)
    save(fig, "fig12_imagery")


def fig13_curves(R):
    c = R["extras"]["curves"]
    op = c["operating_point"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                 gridspec_kw=dict(wspace=0.27))
    a1.plot(c["pr"]["recall"], c["pr"]["precision"], color=BLUE, lw=1.8)
    a1.scatter([op["recall"]], [op["precision"]], s=75, marker="X", color=RED,
               zorder=5, label="deployed operating point")
    a1.set_xlabel("Recall")
    a1.set_ylabel("Precision")
    a1.set_xlim(0, 1.02)
    a1.set_ylim(0, 1.02)
    a1.legend(loc="lower left")
    titled(a1, f"Precision–recall (AP = {c['ap']:.3f})")

    a2.plot(c["roc"]["fpr"], c["roc"]["tpr"], color=GREEN, lw=1.8)
    a2.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    a2.set_xlabel("False positive rate")
    a2.set_ylabel("True positive rate")
    a2.set_xlim(0, 1.02)
    a2.set_ylim(0, 1.02)
    titled(a2, f"ROC (AUC = {c['auc']:.3f})")

    # The three-line note needs real headroom or it lands on the panel titles.
    fig.suptitle("Threshold-free performance, with the deployed point marked",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.32)
    fig.text(0.005, 1.26,
             "The operating point sits below the precision–recall frontier by "
             "construction: the persistence rule discards\nisolated exceedances "
             "that the curve is free to count. Average precision is the more "
             "informative summary\nhere, given the class imbalance.",
             fontsize=7.2, color=GREY, va="top", ha="left", linespacing=1.35)
    save(fig, "fig13_curves")


def fig14_separability(R):
    s = R["extras"]["separability"]
    e = np.array(s["edges"])
    ctr = (e[:-1] + e[1:]) / 2
    w = e[1] - e[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(ctr, s["healthy"], w, color=GREEN, alpha=.65, label="healthy")
    ax.bar(ctr, s["stressed"], w, color=RED, alpha=.65, label="stressed")
    ax.set_yscale("log")
    ax.set_xlabel("LOF anomaly score")
    ax.set_ylabel("Observations (log scale)")
    ax.legend()
    titled(ax, "Are the two populations actually separable?",
           f"Median score {s['median_healthy']:.3f} for healthy observations "
           f"against {s['median_stressed']:.3f} for stressed. The overlap "
           "region is the\nirreducible error — no choice of threshold removes "
           "it, which is why the ranking matters more than the cut.")
    save(fig, "fig14_separability")


def fig15_threshold_spread(R):
    t = R["extras"]["threshold_spread"]
    thr = np.array(t["thresholds"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(thr, bins=55, color=BLUE, alpha=.72)
    ax.axvline(t["global_equivalent"], color=RED, lw=1.6,
               label=f"single global cut ({t['global_equivalent']:.2f})")
    ax.axvline(t["median"], color="#333", ls="--", lw=1.2,
               label=f"median field ({t['median']:.2f})")
    ax.set_xlabel("Per-field threshold")
    ax.set_ylabel("Fields")
    ax.legend()
    titled(ax, "Why the threshold is computed per field",
           f"Thresholds span a {t['ratio']:.1f}-fold range across "
           f"{len(thr):,} held-out fields. One global cut sits far from most of "
           "them —\nit would over-alarm the quiet fields and under-alarm the "
           "volatile ones.")
    save(fig, "fig15_threshold_spread")


def fig16_significance(R):
    sg = R["extras"]["significance"]
    boot = np.array(sg["boot"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(boot, bins=50, color="#E7E9EC", edgecolor=GREY, linewidth=.5)
    ax.axvline(0, color=RED, lw=1.6, label="no difference")
    ax.axvline(sg["delta_f1"], color=BLUE, lw=1.6,
               label=f"observed ΔF1 = {sg['delta_f1']:+.4f}")
    lo, hi = sg["ci95"]
    ax.axvspan(lo, hi, color=BLUE, alpha=.12,
               label=f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    ax.set_xlabel("F1(this study) − F1(Study 2 rule)")
    ax.set_ylabel("Bootstrap resamples")
    ax.legend(loc="upper left")
    titled(ax, "Is the headline improvement real?",
           f"Bootstrap over {sg['n_units']:,} field-seed units, "
           f"{sg['n_boot']:,} resamples. The interval excludes zero by a wide "
           f"margin\n(paired permutation p = {sg['p_value']:.4f}), so the gain "
           "is not seed noise.")
    save(fig, "fig16_significance")


def fig17_study_area(R):
    """Where the tiles are, and how the split falls on the ground."""
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    g = R["extras"]["geometry"]
    X, Y, SP = g["x"], g["y"], g["split"]
    side = g["side_m"]
    ids = sorted(X, key=int)
    x0 = min(X.values()); y0 = min(Y.values())
    COL = {"fit": GREY, "cal": AMBER, "test": BLUE}
    n = {k: sum(1 for v in SP.values() if v == k) for k in COL}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    for t in ids:
        axA.add_patch(Rectangle(((X[t] - side / 2 - x0) / 1000,
                                 (Y[t] - side / 2 - y0) / 1000),
                                side / 1000, side / 1000,
                                facecolor=COL.get(SP[t], GREY),
                                edgecolor="none", alpha=.9))
    ex, ey = g["extent_km"]
    axA.set_xlim(-.6, ex + 1.1); axA.set_ylim(-.6, ey + 1.1)
    axA.set_aspect("equal"); axA.grid(False)
    axA.set_xlabel("easting (km from SW corner)"); axA.set_ylabel("northing (km)")
    titled(axA, f"(A)  {g['n_tiles']:,} tiles, Lombardy 2019",
           f"{side:.0f} m tiles on a 10 m grid, {g['area_km2']:.0f} km$^2$ in "
           f"all. {g['crs']}.")
    axA.legend(handles=[Line2D([], [], marker="s", ls="", ms=7, color=COL[k],
                               label=f"{k}  ({n[k]:,})") for k in
                        ("fit", "cal", "test")],
               loc="upper left", bbox_to_anchor=(1.01, 1.0),
               handletextpad=.6, borderaxespad=0)

    # zoom on a patch near the middle, to make the interleaving legible
    zx, zy, z = ex * .2, ey * .36, side / 1000 * 7
    for t in ids:
        px, py = (X[t] - x0) / 1000, (Y[t] - y0) / 1000
        if zx <= px <= zx + z and zy <= py <= zy + z:
            axB.add_patch(Rectangle((px - side / 2000, py - side / 2000),
                                    side / 1000, side / 1000,
                                    facecolor=COL.get(SP[t], GREY),
                                    edgecolor="white", lw=.7))
    axB.set_xlim(zx, zx + z); axB.set_ylim(zy, zy + z)
    axB.set_aspect("equal"); axB.grid(False)
    axB.set_xlabel("easting (km)")
    titled(axB, "(B)  the split is interleaved",
           f"{g['test_adjacent_to_fit']*100:.0f}% of test tiles are "
           "edge-adjacent to a fit tile.")
    fig.tight_layout()
    save(fig, "fig17_study_area")


def fig18_phenology(R):
    """The seasonal swing the detector is measured against."""
    import datetime as _dt
    p = R["extras"]["phenology"]
    d = [_dt.date.fromisoformat(s) for s in p["date"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.fill_between(d, p["p10"], p["p90"], color=GREEN, alpha=.18, lw=0,
                    label="10th-90th percentile across tiles")
    ax.plot(d, p["median"], color=GREEN, lw=2, label="median tile")
    ax.set_ylabel("MSAVI (tile median)"); ax.set_xlabel("2019")
    titled(ax, "The seasonal signal the detector works against",
           f"Median MSAVI over all {R['setup']['n_tiles']['fit'] + R['setup']['n_tiles']['cal'] + R['setup']['n_tiles']['test']:,} "
           f"tiles at each retained pass. The canopy cycle moves the index\n"
           f"{p['seasonal_swing']:.1f}-fold across the year, more than the 40% "
           "stress injected to test the detector, which is why\nevery threshold "
           "is per tile and computed against a rolling baseline rather than a "
           "fixed value.")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    save(fig, "fig18_phenology")


def fig19_alarm_map(R):
    """Where the false alarms fall, and how concentrated they are."""
    from matplotlib.patches import Rectangle
    g, a = R["extras"]["geometry"], R["extras"]["alarm_map"]
    X, Y, side = g["x"], g["y"], g["side_m"]
    x0 = min(X.values()); y0 = min(Y.values())
    fa = np.array(a["false_alarms"])

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    for t in X:                                    # every tile, faint context
        ax.add_patch(Rectangle(((X[t] - side / 2 - x0) / 1000,
                                (Y[t] - side / 2 - y0) / 1000),
                               side / 1000, side / 1000,
                               facecolor="#E8EAED", edgecolor="none"))
    cmap = plt.get_cmap("YlOrRd")
    vmax = max(1.0, float(np.percentile(fa, 99)))
    for t, v in zip(a["tiles"], fa):
        k = str(t)
        ax.add_patch(Rectangle(((X[k] - side / 2 - x0) / 1000,
                                (Y[k] - side / 2 - y0) / 1000),
                               side / 1000, side / 1000,
                               facecolor=cmap(min(v / vmax, 1.0)),
                               edgecolor="none"))
    ex, ey = g["extent_km"]
    ax.set_xlim(-.6, ex + 1.1); ax.set_ylim(-.6, ey + 1.1)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("easting (km)"); ax.set_ylabel("northing (km)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=.8, pad=.02)
    cb.set_label("false alarms per tile", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    titled(ax, "(A)  False alarms in space",
           f"{len(fa)} held-out tiles in colour; the rest in grey.")

    axh.hist(fa, bins=np.arange(0, fa.max() + .4, .2), color=AMBER,
             edgecolor="none")
    axh.set_xlabel("false alarms per tile (mean of 5 seeds)")
    axh.set_ylabel("tiles")
    titled(axh, "(B)  and concentrated in a minority",
           f"{a['zero_fraction']*100:.0f}% of held-out tiles never false-alarm;\n"
           f"the noisiest tenth carry {a['top_decile_share']*100:.0f}% of them.")
    fig.tight_layout()
    save(fig, "fig19_alarm_map")


def fig20_features(R):
    """Correlation structure of the feature set, and the one dead column."""
    f = R["extras"]["features"]
    M = np.array(f["corr"]); cols = f["columns"]; n = len(cols)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    lbl = [c.replace("_", " ") for c in cols]
    ax.set_xticklabels(lbl, rotation=90, fontsize=6.4)
    ax.set_yticklabels(lbl, fontsize=6.4)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=.82, pad=.02)
    cb.set_label("Pearson r", fontsize=8); cb.ax.tick_params(labelsize=7)
    dead = ", ".join(c.replace("_", " ") for c in f["constant"]) or "none"
    titled(ax, "The feature set is redundant, and one column is inert",
           f"Median |r| is {f['median_abs_r']:.2f} over the informative pairs and "
           f"{f['n_pairs_above_0p9']} exceed 0.9. The blank\nrow and column is "
           f"{dead}: it is constant across every observation, so the\n"
           f"detector sees {f['n_informative']} features rather than {n}.")
    save(fig, "fig20_features")


def render(R):
    for fn in (fig10_pipeline, fig11_worked_example, fig12_imagery,
               fig13_curves, fig14_separability, fig15_threshold_spread,
               fig16_significance, fig17_study_area, fig18_phenology,
               fig19_alarm_map, fig20_features):
        fn(R)
