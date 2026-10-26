"""
common.py — shared machinery for experiment4.

experiment4 only ever READS from the existing project:
  * hybrid_pipeline_experiments/outputs/features_timeseries.csv   (the 83,200-row feature table)
  * hybrid_pipeline_experiments/config.py, utils/                 (constants + the injector)

Nothing outside experiment4/ is written to or modified.

The one idea that runs through this whole study: **a threshold must never see
an anomaly.** The main study computes its per-field percentile on the batch
that already contains the injections, which quietly lets the cut-off adapt to
the thing it is being measured against. Here every threshold is calibrated on
clean, pre-injection scores, the percentile itself is chosen on a calibration
split, and results are reported on a third, untouched split.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # crop_stress/appendix
PKG = os.path.dirname(BASE_DIR)                         # crop_stress
PROJECT = os.path.dirname(PKG)                          # the repo root
MAIN_PIPELINE = os.path.join(PKG, "engine")
sys.path.insert(0, MAIN_PIPELINE)

from config import TIMESERIES, LOF, FILES              # noqa: E402
from utils.preparation import FEATURE_COLUMNS          # noqa: E402
from utils.evaluation import inject_anomalies          # noqa: E402

CACHE_DIR = os.path.join(PKG, "cache", "appendix")
OUT_DIR = os.path.join(PKG, "outputs", "appendix")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ── Experiment constants (matched to Study 2 unless noted) ───────────
SEEDS = [42, 123, 456, 789, 1024]
INJECTION_RATIO = 0.15
SEVERITY = 0.40
DURATION = 2
MIN_CONSECUTIVE = 2
BASELINE_PCTL = 85            # Study 2's recommended operating point
SPLIT_SEED = 20260822
SPLIT_FRACTIONS = (0.50, 0.25, 0.25)   # fit / calibrate / test

# Peer-median feature: dates seen by fewer fields than this get no adjustment,
# because a "cross-field median" over a handful of fields is noise.
MIN_FIELDS_PER_DATE = 50


# ══════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════
def load_base():
    """The clean feature table, plus the arrays every experiment needs."""
    df = pd.read_csv(FILES["features_timeseries"])
    return {
        "df": df,
        "tiles": df["tile_id"].values,
        "dates": df["date"].values,
        "X": feature_matrix(df),
    }


def feature_matrix(df, columns=None):
    """FEATURE_COLUMNS as a clean float matrix, exactly as get_scores() does."""
    cols = FEATURE_COLUMNS if columns is None else columns
    X = df[cols].values
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def make_splits(tiles):
    """
    Three disjoint tile sets.

      fit   — the detector is fitted here (clean rows only)
      cal   — hyperparameters (threshold percentile, k) are chosen here
      test  — reported numbers come from here, and nothing is chosen on it

    The main study has only the equivalent of `fit`, and reports on it.
    """
    uniq = pd.unique(tiles)
    rng = np.random.RandomState(SPLIT_SEED)
    shuffled = rng.permutation(uniq)
    n = len(shuffled)
    a = int(n * SPLIT_FRACTIONS[0])
    b = a + int(n * SPLIT_FRACTIONS[1])
    fit, cal, test = shuffled[:a], shuffled[a:b], shuffled[b:]
    assert len(set(fit) & set(cal)) == 0
    assert len(set(fit) & set(test)) == 0
    assert len(set(cal) & set(test)) == 0
    return {"fit": fit, "cal": cal, "test": test}


def tile_slices(tiles_subset):
    """
    [(tile_id, row_positions), ...] for one subset, in stable time order.

    Row positions index into the subset array, not the full table.
    """
    order = np.argsort(tiles_subset, kind="stable")
    uniq, starts, counts = np.unique(tiles_subset[order], return_index=True,
                                     return_counts=True)
    return [(t, order[s:s + c]) for t, s, c in zip(uniq, starts, counts)]


# ══════════════════════════════════════════════════════════════════════
#  Injection, cached
# ══════════════════════════════════════════════════════════════════════
def injected(df, seed, tiles):
    """
    Inject once per seed and cache. inject_anomalies() rebuilds the temporal
    features tile by tile in Python, which is the slow step in the whole study;
    every experiment reuses the same cached draw so the injections are also
    identical across experiments.

    Returns (X_modified, ground_truth, p50_modified).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(
        CACHE_DIR,
        f"inj_s{seed}_r{INJECTION_RATIO}_sev{SEVERITY}_d{DURATION}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["gt"], z["p50"]

    df_mod, gt = inject_anomalies(df, INJECTION_RATIO, SEVERITY, DURATION,
                                  seed, TIMESERIES["rolling_window"])
    # inject_anomalies regroups rows by tile; verify the order survived, or the
    # positional ground truth would silently desynchronise from the features.
    assert (df_mod["tile_id"].values == tiles).all(), \
        "injection reordered rows — ground truth would be misaligned"

    X = feature_matrix(df_mod)
    p50 = df_mod["p50"].to_numpy(dtype=float)
    np.savez_compressed(path, X=X, gt=gt, p50=p50)
    return X, gt, p50


# ══════════════════════════════════════════════════════════════════════
#  Detector
# ══════════════════════════════════════════════════════════════════════
def fit_detector(X_train, n_neighbors=None):
    """StandardScaler + LOF in novelty mode, matching the main pipeline."""
    scaler = StandardScaler().fit(X_train)
    lof = LocalOutlierFactor(
        n_neighbors=LOF["n_neighbors"] if n_neighbors is None else n_neighbors,
        contamination=LOF["contamination"],
        novelty=True,
    ).fit(scaler.transform(X_train))
    return scaler, lof


def score(scaler, lof, X):
    """Higher = more anomalous, matching run_experiments.get_scores()."""
    return -lof.score_samples(scaler.transform(X))


# ══════════════════════════════════════════════════════════════════════
#  Decision rule
# ══════════════════════════════════════════════════════════════════════
def strict_consecutive(raw_flags, min_consecutive=MIN_CONSECUTIVE):
    """
    Confirm a flag only when it belongs to a run of >= min_consecutive.
    Identical logic to utils.hybrid.apply_threshold_and_persistence; kept here
    so the threshold can be supplied from outside.
    """
    n = len(raw_flags)
    if min_consecutive <= 1 or n < min_consecutive:
        return raw_flags.copy()
    final = np.zeros(n, dtype=bool)
    run = 0
    for j in range(n):
        if raw_flags[j]:
            run += 1
        else:
            if run >= min_consecutive:
                final[j - run:j] = True
            run = 0
    if run >= min_consecutive:
        final[n - run:n] = True
    return final


def predict(scores, slices, min_consecutive=MIN_CONSECUTIVE,
            pctl=None, frozen=None):
    """
    Per-field threshold + strict persistence.

    pctl   -> threshold is that percentile OF THE SCORES PASSED IN, i.e.
              computed after injection. This is the main study's behaviour.
    frozen -> frozen[i] is field i's threshold, precomputed on clean data.
    """
    assert (pctl is None) != (frozen is None), "pass exactly one of pctl/frozen"
    labels = np.zeros(len(scores), dtype=int)
    for i, (_t, idx) in enumerate(slices):
        s = scores[idx]
        thr = np.percentile(s, pctl) if frozen is None else frozen[i]
        labels[idx[strict_consecutive(s > thr, min_consecutive)]] = 1
    return labels


def clean_thresholds(clean_scores, slices, pctl):
    """Each field's own percentile of its CLEAN scores. Never sees an anomaly."""
    return np.array([np.percentile(clean_scores[idx], pctl) for _t, idx in slices])


# ══════════════════════════════════════════════════════════════════════
#  Point-wise metrics
# ══════════════════════════════════════════════════════════════════════
def point_metrics(gt, pred, scores=None):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 average_precision_score, roc_auc_score,
                                 confusion_matrix)
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "precision": float(precision_score(gt, pred, zero_division=0)),
        "recall": float(recall_score(gt, pred, zero_division=0)),
        "f1": float(f1_score(gt, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_alarms": int(fp),
        "missed": int(fn),
    }
    if scores is not None and len(np.unique(gt)) > 1:
        out["roc_auc"] = float(roc_auc_score(gt, scores))
        out["pr_auc"] = float(average_precision_score(gt, scores))
    else:
        out["roc_auc"] = 0.0
        out["pr_auc"] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════
#  Range-based metrics (Tatbul et al., NeurIPS 2018)
# ══════════════════════════════════════════════════════════════════════
def _runs(binary_seq):
    """Contiguous runs of 1s as (start, end_exclusive)."""
    b = np.asarray(binary_seq).astype(bool)
    if not b.any():
        return []
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _overlap(a, b):
    """Size of the intersection of two [start, end) ranges."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def range_metrics(gt, pred, slices, alpha=0.5):
    """
    Time-series aware precision and recall.

    Point-wise scoring punishes this pipeline unfairly: the persistence rule
    deliberately discards isolated exceedances, so catching one observation of
    a two-observation event scores 0.5 recall even though the event was found.
    For a field advisory the operational question is "was the event caught at
    all", which is what the existence reward captures.

    Settings: flat positional bias (every observation in an event counts the
    same — crop stress has no preferred position), reciprocal cardinality (a
    real event split across several fragmented alarms is penalised), and
    alpha for the existence/overlap trade-off in recall. Precision uses
    alpha = 0, per the paper: there is no existence reward for a prediction.

    Ranges are formed within each field, never across field boundaries.
    """
    rec_scores, prec_scores = [], []
    for _t, idx in slices:
        R = _runs(np.asarray(gt)[idx])
        P = _runs(np.asarray(pred)[idx])

        for r in R:
            hits = [p for p in P if _overlap(r, p) > 0]
            if not hits:
                rec_scores.append(0.0)
                continue
            cardinality = 1.0 if len(hits) == 1 else 1.0 / len(hits)
            overlap = sum(_overlap(r, p) for p in hits) / (r[1] - r[0])
            rec_scores.append(alpha * 1.0 + (1 - alpha) * cardinality * overlap)

        for p in P:
            hits = [r for r in R if _overlap(p, r) > 0]
            if not hits:
                prec_scores.append(0.0)
                continue
            cardinality = 1.0 if len(hits) == 1 else 1.0 / len(hits)
            overlap = sum(_overlap(p, r) for r in hits) / (p[1] - p[0])
            prec_scores.append(cardinality * overlap)

    rec = float(np.mean(rec_scores)) if rec_scores else 0.0
    prec = float(np.mean(prec_scores)) if prec_scores else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"range_precision": prec, "range_recall": rec, "range_f1": f1,
            "n_true_events": len(rec_scores), "n_pred_events": len(prec_scores)}


# ══════════════════════════════════════════════════════════════════════
#  Peer-median feature (experiment 3 of this study)
# ══════════════════════════════════════════════════════════════════════
def peer_deviation(p50, dates):
    """
    Deviation of each field from the median of all other fields observed on the
    SAME DATE, scaled by that date's spread.

    The existing `diff_from_global_median` feature compares against the whole
    dataset, which is a season-wide constant and cannot react to a single hazy
    pass. A weather front or residual haze pushes every field down together on
    one date; that is not crop stress, and persistence cannot filter it because
    the front lasts across passes. Comparing against same-date peers cancels it.

    Dates observed by fewer than MIN_FIELDS_PER_DATE fields get no adjustment,
    since a median over a handful of fields is noise.
    """
    s = pd.Series(p50)
    g = s.groupby(pd.Series(dates))
    med = g.transform("median").to_numpy(dtype=float)
    # median absolute deviation, robust to the injected fields in the batch
    mad = g.transform(lambda x: np.median(np.abs(x - np.median(x))))
    mad = np.asarray(mad, dtype=float)
    counts = g.transform("size").to_numpy(dtype=float)

    scale = np.where(mad > 0, mad * 1.4826, np.nan)
    dev = (p50 - med) / scale
    dev = np.nan_to_num(dev, nan=0.0, posinf=0.0, neginf=0.0)
    dev[counts < MIN_FIELDS_PER_DATE] = 0.0
    return dev


def with_peer_feature(X, p50, dates):
    """Append the peer-deviation column to a feature matrix."""
    return np.column_stack([X, peer_deviation(p50, dates)])


# ══════════════════════════════════════════════════════════════════════
#  Misc
# ══════════════════════════════════════════════════════════════════════
def agg(dicts, keys):
    """mean and sd across seeds for the given metric keys."""
    out = {}
    for k in keys:
        v = np.array([d[k] for d in dicts], dtype=float)
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std())
    return out


def ensure_dirs():
    for d in (CACHE_DIR, OUT_DIR, TABLE_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)
