"""
viz.py — one place for figure style and shared chart primitives.

Every figure is saved as a 400 dpi PNG and a vector PDF, so the same call
serves both a slide deck and a LaTeX build.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from config import FIG_DIR        # noqa: E402

# Okabe-Ito, colourblind-safe
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PURPLE, SKY = "#D55E00", "#CC79A7", "#56B4E9"
GREY, SOFT = "#6E6E6E", "#E4F0E8"

METRICS = ["precision", "recall", "f1"]
MLABEL = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
MCOLOR = [BLUE, ORANGE, GREEN]

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.55,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"    -> figures/{name}.png + .pdf", flush=True)


def note(ax_or_fig, text, y=-0.02):
    """Small grey caption under a figure."""
    target = ax_or_fig if hasattr(ax_or_fig, "text") else ax_or_fig
    target.text(0, y, text, transform=getattr(target, "transFigure", None),
                fontsize=7, color=GREY, va="top")


def metric_bars(ax, df, label_col, highlight=None, ylim=1.10):
    """Grouped precision / recall / F1 bars with seed-to-seed error bars."""
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    w = 0.26
    for k, (m, c) in enumerate(zip(METRICS, MCOLOR)):
        vals = df[m].to_numpy(float)
        errs = (df[f"{m}_sd"].to_numpy(float) if f"{m}_sd" in df
                else np.zeros(len(df)))
        bars = ax.bar(x + (k - 1) * w, vals, w, label=MLABEL[m], color=c,
                      edgecolor="white", linewidth=0.5, yerr=errs, capsize=2.5,
                      error_kw=dict(elinewidth=0.8, ecolor="#333333"))
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, color="#222222")
    if highlight is not None and highlight in labels:
        i = labels.index(highlight)
        ax.axvspan(i - .5, i + .5, color=GREEN, alpha=.07, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim)
    ax.set_ylabel("Score")
    ax.grid(axis="x", visible=False)
    return ax


def legend_above(ax, ncols=3):
    return ax.legend(ncols=ncols, loc="lower right", bbox_to_anchor=(1.0, 1.005))


def false_alarm_bars(ax, df, label_col, color=VERM):
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    v = df["false_alarms"].to_numpy(float)
    ax.bar(x, v, .55, color=color, edgecolor="white", linewidth=.5)
    for xi, vi in zip(x, v):
        ax.text(xi, vi * 1.12, f"{int(vi):,}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("False alarms (log)")
    ax.set_ylim(top=max(v) * 4 if max(v) > 0 else 10)
    ax.grid(axis="x", visible=False)
    return ax


def read_table(name):
    from config import TAB_DIR
    return pd.read_csv(os.path.join(TAB_DIR, f"{name}.csv"))
