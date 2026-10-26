"""
evaluation.py — synthetic stress injection, metrics, and uncertainty.

There are no ground-truth stress labels for this data, so the only honest way
to measure detection is to corrupt clean data in a controlled way and see what
comes back.

Two design decisions matter:

1. **Injection is physically coherent across indices.** A stress event lowers
   the greenness indices *and* canopy moisture together, in the proportions the
   biology implies. Perturbing one channel in isolation would let a
   multispectral model win for the wrong reason.

2. **Ground truth travels as a DataFrame column**, never as a parallel array.
   Feature building regroups rows by tile, so an index-aligned array would
   silently desynchronise. Carrying the label in the frame makes that class of
   bug impossible.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix)

from config import INDICES

# How strongly each index responds to stress, relative to the nominal severity.
# Chlorophyll indices react most; moisture falls too but less sharply; MSAVI is
# soil-adjusted so it moves slightly less than raw NDVI.
STRESS_RESPONSE = {"msavi": 0.90, "ndvi": 1.00, "ndre": 1.10, "ndmi": 0.70}

# A spectral shift is a chlorophyll-specific event: the red edge moves while
# overall greenness and moisture barely do.
SPECTRAL_RESPONSE = {"msavi": 0.25, "ndvi": 0.35, "ndre": 1.30, "ndmi": 0.20}

STAT_SUFFIXES = ["mean", "p10", "p25", "p50", "p75", "p90"]


def _apply(df, rows, idx, factor):
    """Scale every summary statistic of one index, for a set of rows."""
    rows = np.atleast_1d(rows)
    cols = [f"{idx}_{s}" for s in STAT_SUFFIXES if f"{idx}_{s}" in df.columns]
    if len(rows) and cols:
        df.loc[rows, cols] = df.loc[rows, cols].to_numpy() * factor


def inject(obs, ratio, severity, duration, seed, kind="sudden"):
    """
    Corrupt a fraction of the observation table with synthetic stress.

    Returns (modified_copy, n_events). The label lands in a `stressed` column.
    """
    rng = np.random.default_rng(seed)
    df = obs.copy()
    df["stressed"] = 0

    # candidate start positions: need `duration` consecutive rows in one field
    starts = []
    for tile, g in df.groupby("tile_id", sort=True):
        pos = g.index.to_numpy()
        if len(pos) >= duration:
            starts.append(pos[: len(pos) - duration + 1])
    if not starts:
        return df, 0
    starts = np.concatenate(starts)

    n_events = max(1, int(len(df) * ratio) // max(duration, 1))
    n_events = min(n_events, len(starts))
    picked = rng.choice(starts, size=n_events, replace=False)

    response = SPECTRAL_RESPONSE if kind == "spectral" else STRESS_RESPONSE

    # Group affected rows by step so each step's scaling is applied in one
    # vectorised assignment rather than row by row.
    valid = df.index
    for step in range(duration):
        rows = picked + step
        rows = rows[np.isin(rows, valid)]
        if rows.size == 0:
            continue
        # gradual stress ramps in across the event instead of dropping at once
        frac = severity * (step + 1) / duration if kind == "gradual" else severity
        for idx in INDICES:
            _apply(df, rows, idx, 1.0 - frac * response[idx])
        df.loc[rows, "stressed"] = 1

    return df, n_events


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "false_alarms": int(fp),
        "missed": int(fn),
        "tp": int(tp), "tn": int(tn),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    else:
        out["pr_auc"] = np.nan
        out["roc_auc"] = np.nan
    return out


def aggregate(runs):
    """Mean and standard deviation across seeds."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float, np.floating))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = float(np.nanmean(vals))
        out[k + "_sd"] = float(np.nanstd(vals))
    return out


def bootstrap_ci(y_true, y_pred, n=400, seed=0, stat="f1"):
    """
    Percentile bootstrap CI for a single run.

    Resamples observations, so it captures sampling uncertainty in the metric
    itself rather than only seed-to-seed variation in the injection.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    N = len(y_true)
    fn = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[stat]
    vals = np.empty(n)
    for i in range(n):
        s = rng.integers(0, N, N)
        vals[i] = fn(y_true[s], y_pred[s], zero_division=0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_test(a, b, n=10000, seed=0):
    """
    Permutation test on paired per-seed differences.

    Answers "is configuration A actually better than B, or is this noise?" —
    the question the earlier study never asked when picking between two
    configurations 0.006 F1 apart.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return float(obs), p
