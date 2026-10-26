"""
core.py — the shared machinery for the unified study.

This folder is the consolidation of three earlier pieces of work into ONE
protocol so that every table in the paper is comparable to every other:

  hybrid_pipeline_experiments2/   Study 2 — which detector wins, and where it breaks
  .../protocol_comparison/        what held-out evaluation actually costs
  experiment4/                    where the operating point should sit

The problem it solves: those three measured different things at different
operating points on different subsets, so no two of their tables could be put
side by side. Here every number comes from the same split, the same detectors,
the same injections and the same decision rule.

READ-ONLY dependencies (nothing outside this folder is written):
  hybrid_pipeline_experiments/outputs/features_timeseries.csv
  hybrid_pipeline_experiments/config.py, utils/

THE PROTOCOL
------------
Fields are split three ways, disjointly:

  fit    50%   detectors are fitted here, on clean rows only
  cal    25%   the threshold rule and any hyperparameter are chosen here
  test   25%   every reported number; nothing is ever selected on it

Every threshold is calibrated on clean, pre-injection scores. Study 2 computed
its percentile on the batch that already contained the injections, which caps
how strict the cut can be — at a 15% injection rate a P85 cut lands exactly on
the boundary of the anomalous group and is dragged up into it.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))     # crop_stress/study
PKG = os.path.dirname(BASE_DIR)                           # crop_stress
PROJECT = os.path.dirname(PKG)                            # the repo root
MAIN = os.path.join(PKG, "engine")
sys.path.insert(0, MAIN)

from config import (TIMESERIES, LOF, ISOLATION_FOREST,      # noqa: E402
                    AUTOENCODER, FILES)
from utils.preparation import FEATURE_COLUMNS               # noqa: E402
from utils.models import CropAutoencoder                    # noqa: E402
from utils.evaluation import (inject_anomalies,             # noqa: E402
                              inject_gradual_anomalies,
                              inject_spectral_shift_anomalies)

# One outputs tree for the whole package, one subfolder per study.
CACHE_DIR = os.path.join(PKG, "cache", "study")
OUT_DIR = os.path.join(PKG, "outputs", "study")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# ── Protocol constants ───────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
INJECTION_RATIO = 0.15
SEVERITY = 0.40
DURATION = 2
MIN_CONSECUTIVE = 2
SPLIT_SEED = 20260822
SPLIT_FRACTIONS = (0.50, 0.25, 0.25)
LEGACY_PCTL = 85          # what Study 2 used, kept for the comparison row
RANDOM_STATE = 42

INJECTORS = {
    "sudden": inject_anomalies,
    "gradual": inject_gradual_anomalies,
    "spectral": inject_spectral_shift_anomalies,
}

TRAJECTORY_FEATURES = ["previous_value", "cumulative_change", "acceleration"]


def ensure_dirs():
    for d in (CACHE_DIR, OUT_DIR, TABLE_DIR, FIG_DIR):
        os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════
def feature_matrix(df, columns=None):
    X = df[FEATURE_COLUMNS if columns is None else columns].values
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def add_trajectory(df):
    """Study 2's rejected extra features, rebuilt so the ablation can be redone."""
    out = []
    for _t, g in df.groupby("tile_id", sort=False):
        g = g.sort_values("date").copy()
        g["previous_value"] = g["smoothed_p50"].shift(1).ffill().bfill()
        g["cumulative_change"] = g["diff"].cumsum()
        g["acceleration"] = g["diff"].diff().ffill().bfill().fillna(0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def load_base():
    df = pd.read_csv(FILES["features_timeseries"])
    return {"df": df, "tiles": df["tile_id"].values,
            "dates": df["date"].values, "X": feature_matrix(df)}


def make_splits(tiles):
    uniq = pd.unique(tiles)
    rng = np.random.RandomState(SPLIT_SEED)
    sh = rng.permutation(uniq)
    a = int(len(sh) * SPLIT_FRACTIONS[0])
    b = a + int(len(sh) * SPLIT_FRACTIONS[1])
    out = {"fit": sh[:a], "cal": sh[a:b], "test": sh[b:]}
    for x in ("fit", "cal", "test"):
        for y in ("fit", "cal", "test"):
            if x < y:
                assert not (set(out[x]) & set(out[y])), "splits overlap"
    return out


def tile_slices(tiles_subset):
    order = np.argsort(tiles_subset, kind="stable")
    uniq, starts, counts = np.unique(tiles_subset[order], return_index=True,
                                     return_counts=True)
    return [(t, order[s:s + c]) for t, s, c in zip(uniq, starts, counts)]


# ══════════════════════════════════════════════════════════════════════
#  Injection, cached on disk
# ══════════════════════════════════════════════════════════════════════
def injected(df, tiles, seed, severity=SEVERITY, duration=DURATION,
             kind="sudden", ratio=INJECTION_RATIO):
    """
    One injected draw. Cached, because rebuilding the temporal features tile by
    tile is by far the slowest step and every ablation reuses the same draws.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"inj_{kind}_s{seed}_r{ratio}_sev{severity}_d{duration}.npz"
    path = os.path.join(CACHE_DIR, tag)
    # experiment4 cached the baseline scenario under an older name; reuse it.
    legacy = os.path.join(CACHE_DIR,
                          f"inj_s{seed}_r{ratio}_sev{severity}_d{duration}.npz")
    if kind == "sudden" and not os.path.exists(path) and os.path.exists(legacy):
        path = legacy
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["gt"], z["p50"]

    df_mod, gt = INJECTORS[kind](df, ratio, severity, duration, seed,
                                 TIMESERIES["rolling_window"])
    assert (df_mod["tile_id"].values == tiles).all(), \
        "injection reordered rows — ground truth would be misaligned"
    X = feature_matrix(df_mod)
    p50 = df_mod["p50"].to_numpy(dtype=float)
    np.savez_compressed(os.path.join(CACHE_DIR, tag), X=X, gt=gt, p50=p50)
    return X, gt, p50


def injected_frame(df, tiles, seed, severity=SEVERITY, duration=DURATION,
                   kind="sudden", ratio=INJECTION_RATIO):
    """The full injected DataFrame — needed by the rule-based baselines."""
    df_mod, gt = INJECTORS[kind](df, ratio, severity, duration, seed,
                                 TIMESERIES["rolling_window"])
    assert (df_mod["tile_id"].values == tiles).all()
    return df_mod, gt


# ══════════════════════════════════════════════════════════════════════
#  Detectors — all fitted on the FIT split, clean rows only
# ══════════════════════════════════════════════════════════════════════
def fit_lof(X, k=None):
    sc = StandardScaler().fit(X)
    m = LocalOutlierFactor(n_neighbors=LOF["n_neighbors"] if k is None else k,
                           contamination=LOF["contamination"],
                           novelty=True).fit(sc.transform(X))
    return sc, m


def fit_iso(X):
    sc = StandardScaler().fit(X)
    m = IsolationForest(n_estimators=ISOLATION_FOREST["n_estimators"],
                        contamination=ISOLATION_FOREST["contamination"],
                        random_state=RANDOM_STATE).fit(sc.transform(X))
    return sc, m


def fit_ae(X, seed=RANDOM_STATE, verbose=False):
    """Same architecture and schedule as the main pipeline, refit on `fit`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    n_tr = int(len(Xs) * AUTOENCODER["train_ratio"])
    perm = np.random.permutation(len(Xs))
    tr = torch.FloatTensor(Xs[perm[:n_tr]])

    model = CropAutoencoder(input_dim=X.shape[1],
                            hidden_1=AUTOENCODER["hidden_dim_1"],
                            latent=AUTOENCODER["latent_dim"],
                            hidden_2=AUTOENCODER["hidden_dim_2"],
                            dropout=AUTOENCODER.get("dropout", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=AUTOENCODER["learning_rate"])
    lossf = torch.nn.MSELoss()
    bs = AUTOENCODER["batch_size"]
    model.train()
    for ep in range(AUTOENCODER["epochs"]):
        idx = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(tr), bs):
            b = tr[idx[i:i + bs]]
            opt.zero_grad()
            loss = lossf(model(b), b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if verbose and (ep + 1) % 50 == 0:
            print(f"      AE epoch {ep+1}/{AUTOENCODER['epochs']} "
                  f"loss={tot/len(tr):.5f}")
    model.eval()
    return sc, model


def score_lof(sc, m, X):
    return -m.score_samples(sc.transform(X))


def score_iso(sc, m, X):
    return -m.decision_function(sc.transform(X))


def score_ae(sc, m, X):
    with torch.no_grad():
        t = torch.FloatTensor(sc.transform(X))
        return torch.mean((t - m(t)) ** 2, dim=1).numpy()


def minmax(x):
    lo, hi = np.min(x), np.max(x)
    return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)


# ══════════════════════════════════════════════════════════════════════
#  Rule-based baselines — identical to Study 2's definitions
# ══════════════════════════════════════════════════════════════════════
def rule_pct_drop(df):
    return np.nan_to_num(-df["percent_change"].to_numpy(dtype=float))


def rule_zscore(df):
    g = df.groupby("tile_id")["smoothed_p50"]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform("std").replace(0, np.nan).to_numpy(dtype=float)
    return np.nan_to_num(-(df["smoothed_p50"].to_numpy(dtype=float) - mu) / sd)


def rule_rolling_dev(df):
    sd = df["rolling_std"].replace(0, np.nan).to_numpy(dtype=float)
    dev = (df["rolling_median"].to_numpy(dtype=float)
           - df["smoothed_p50"].to_numpy(dtype=float)) / sd
    return np.nan_to_num(dev)


RULES = {"Fixed % drop": rule_pct_drop,
         "Per-field z-score": rule_zscore,
         "Rolling-median deviation": rule_rolling_dev}


# ══════════════════════════════════════════════════════════════════════
#  Decision rules
# ══════════════════════════════════════════════════════════════════════
def strict_consecutive(flags, n_required):
    n = len(flags)
    if n_required <= 1 or n < n_required:
        return flags.copy()
    out = np.zeros(n, dtype=bool)
    run = 0
    for j in range(n):
        if flags[j]:
            run += 1
        else:
            if run >= n_required:
                out[j - run:j] = True
            run = 0
    if run >= n_required:
        out[n - run:n] = True
    return out


def sliding_2of3(flags):
    """
    2-of-3 within a BACKWARD window [j-2, j-1, j].

    Note for the write-up: this rule is **causal**. Study 2's audit recorded it
    as non-causal, claiming it inspects j+1; the implementation in
    utils/hybrid.py does not. The rule still loses on F1 — but it loses on
    merit, not because it cheats.
    """
    out = np.zeros(len(flags), dtype=bool)
    for j in range(len(flags)):
        if flags[max(0, j - 2):j + 1].sum() >= 2:
            out[j] = True
    return out


def apply_rule(flags, persistence):
    if persistence == "none":
        return flags.copy()
    if persistence == "sliding_2of3":
        return sliding_2of3(flags)
    return strict_consecutive(flags, int(persistence))


def predict(scores, slices, persistence=MIN_CONSECUTIVE, pctl=None,
            frozen=None, global_pctl=None):
    """
    pctl        per-field percentile of the scores passed in (post-injection)
    frozen      per-field threshold precomputed on clean data
    global_pctl one threshold across all fields, from the scores passed in
    """
    n_given = sum(x is not None for x in (pctl, frozen, global_pctl))
    assert n_given == 1, "pass exactly one threshold source"
    labels = np.zeros(len(scores), dtype=int)
    gthr = np.percentile(scores, global_pctl) if global_pctl is not None else None
    for i, (_t, idx) in enumerate(slices):
        s = scores[idx]
        if frozen is not None:
            thr = frozen[i]
        elif gthr is not None:
            thr = gthr
        else:
            thr = np.percentile(s, pctl)
        labels[idx[apply_rule(s > thr, persistence)]] = 1
    return labels


def clean_thresholds(clean_scores, slices, pctl):
    return np.array([np.percentile(clean_scores[idx], pctl)
                     for _t, idx in slices])


def evt_thresholds(clean_scores, slices, q, u_pctl=80.0):
    """
    Peaks-over-threshold (Siffer et al., KDD 2017).

    25 observations give roughly 5 excesses per field — far too few to fit a
    shape per field — so the shape is fitted once on the pooled, scale-
    normalised excesses and only the scale varies per field. Fitted on clean
    data only, like every other threshold here.
    """
    from scipy import stats
    anchors, exc = [], []
    for _t, idx in slices:
        s = clean_scores[idx]
        u = np.percentile(s, u_pctl)
        anchors.append(u)
        exc.append(s[s > u] - u)
    pool = np.concatenate([e / e.mean() for e in exc
                           if len(e) >= 2 and e.mean() > 0])
    xi, _loc, _b = stats.genpareto.fit(pool, floc=0)

    out = np.empty(len(slices))
    for i, ((_t, idx), u, e) in enumerate(zip(slices, anchors, exc)):
        n, nu = len(clean_scores[idx]), len(e)
        if nu < 2 or e.mean() <= 0:
            out[i] = np.percentile(clean_scores[idx], 99.0)
            continue
        beta = e.mean() * (1 - xi) if xi < 1 else e.mean()
        if beta <= 0:
            out[i] = np.percentile(clean_scores[idx], 99.0)
            continue
        r = q * n / nu
        out[i] = (u - beta * np.log(r) if abs(xi) < 1e-6
                  else u + (beta / xi) * (r ** (-xi) - 1))
    return out, float(xi)


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════
def metrics(gt, pred, scores=None):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 average_precision_score, roc_auc_score,
                                 confusion_matrix)
    gt = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0, 1]).ravel()
    out = {"precision": float(precision_score(gt, pred, zero_division=0)),
           "recall": float(recall_score(gt, pred, zero_division=0)),
           "f1": float(f1_score(gt, pred, zero_division=0)),
           "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
           "false_alarms": int(fp)}
    if scores is not None and len(np.unique(gt)) > 1:
        out["roc_auc"] = float(roc_auc_score(gt, scores))
        out["pr_auc"] = float(average_precision_score(gt, scores))
    else:
        out["roc_auc"] = out["pr_auc"] = 0.0
    return out


def agg(rows, keys=("precision", "recall", "f1", "roc_auc", "pr_auc",
                    "specificity", "false_alarms")):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(v.mean())
        out[k + "_sd"] = float(v.std())
    return out
