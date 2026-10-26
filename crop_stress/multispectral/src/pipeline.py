"""
pipeline.py — one end-to-end evaluation run, with no leakage anywhere.

Protocol
--------
The detector learns what "normal" looks like from a **previous season** and is
then used to monitor the **current** one:

    fit + calibrate   ->  2018 observations   (clean, never injected)
    evaluate          ->  2019 observations   (stress injected here)

This is how the system would actually be deployed: you have a field's history,
and you want to know whether this season is departing from it. It is also clean
by construction — the detector and the decision threshold are both fixed before
a single anomaly exists, so neither can adapt to the events they are meant to
catch.

Two weaker protocols were tried first and are worth recording:

* **Same-season, same-fields** (what the earlier study did) inflates results.
  Fitting LOF in novelty mode on the very rows it later scores memorises the
  manifold, so injected points sit far outside it and separate almost
  perfectly. That is where the earlier ROC-AUC of 0.986 came from.
* **Cross-field within one season** over-corrects. Requiring the detector to
  generalise to fields it has never seen is harder than deployment, because in
  practice you always have the field's own history. Retained here as a
  secondary generalisation test (`prepare_cross_field`).
"""
import numpy as np
import pandas as pd

from config import DECIDE, SPLIT
from src import features as F
from src import decision as D
from src.evaluation import inject, metrics


# ══════════════════════════════════════════════════════════════════════
#  Data preparation
# ══════════════════════════════════════════════════════════════════════
def align_seasons(hist_obs, target_obs):
    """Keep only fields observed well in both seasons."""
    shared = sorted(set(hist_obs.tile_id.unique()) & set(target_obs.tile_id.unique()))
    h = hist_obs[hist_obs.tile_id.isin(shared)].sort_values(
        ["tile_id", "date"]).reset_index(drop=True)
    t = target_obs[target_obs.tile_id.isin(shared)].sort_values(
        ["tile_id", "date"]).reset_index(drop=True)
    return h, t, shared


def prepare_cross_field(obs, fit_fraction=None, seed=None):
    """Secondary protocol: split fields, single season. Used for E8."""
    fit_fraction = SPLIT["fit_fraction"] if fit_fraction is None else fit_fraction
    seed = SPLIT["seed"] if seed is None else seed
    tiles = np.sort(obs.tile_id.unique())
    shuffled = np.random.default_rng(seed).permutation(tiles)
    n = int(len(shuffled) * fit_fraction)
    fit_t, eval_t = set(shuffled[:n].tolist()), set(shuffled[n:].tolist())
    return (obs[obs.tile_id.isin(fit_t)].reset_index(drop=True),
            obs[obs.tile_id.isin(eval_t)].reset_index(drop=True),
            fit_t, eval_t)


# ══════════════════════════════════════════════════════════════════════
#  Caches — feature building and clean scoring are the expensive steps and
#  neither depends on the injection seed.
# ══════════════════════════════════════════════════════════════════════
_FEATS = {}
_CLEAN = {}


def features_of(obs, key):
    if key not in _FEATS:
        _FEATS[key] = F.build(obs.assign(stressed=0))
    return _FEATS[key]


def fit_detector(detector, hist_obs, cols, key="hist"):
    """Fit on the clean history season."""
    detector.fit(F.matrix(features_of(hist_obs, key), cols))
    return detector


def calibration_scores(detector, hist_obs, cols, key, baseline=False):
    """
    Score the clean history season — the source of every decision threshold.

    Cached per detector: independent of the injection seed.
    """
    if key in _CLEAN:
        return _CLEAN[key]
    feats = features_of(hist_obs, "hist")
    s = (detector.score_df(feats) if baseline
         else detector.score(F.matrix(feats, cols)))
    _CLEAN[key] = feats[["tile_id", "date"]].assign(score=s)
    return _CLEAN[key]


def reset_caches():
    _FEATS.clear()
    _CLEAN.clear()


# ══════════════════════════════════════════════════════════════════════
def run_once(detector, hist_obs, target_obs, cols, seed,
             severity, duration, ratio, kind="sudden",
             percentile=None, min_consecutive=None, per_field=True,
             baseline=False, det_key=None, calib=None):
    """
    Inject into the target season, score, decide, measure.

    Returns (metrics_dict, scored_frame).
    """
    percentile = DECIDE["threshold_percentile"] if percentile is None else percentile
    min_consecutive = (DECIDE["min_consecutive"] if min_consecutive is None
                       else min_consecutive)

    if calib is None:
        calib = calibration_scores(detector, hist_obs, cols,
                                   det_key or ("d", len(cols)), baseline)
    thresholds = D.calibrate_thresholds(
        calib["score"].to_numpy(), calib["tile_id"].to_numpy(),
        percentile, per_field=per_field)

    inj, _ = inject(target_obs.reset_index(drop=True), ratio, severity,
                    duration, seed, kind=kind)
    feats = F.build(inj)                      # rebuilt so injection propagates
    s = (detector.score_df(feats) if baseline
         else detector.score(F.matrix(feats, cols)))
    fr = feats.assign(score=s).reset_index(drop=True)

    pred, raw = D.decide(fr["score"].to_numpy(), fr["tile_id"].to_numpy(),
                         fr["date"].to_numpy(), thresholds, min_consecutive)
    fr = fr.assign(alarm=pred, raw_flag=raw.astype(int))

    m = metrics(fr["stressed"].to_numpy(), pred, fr["score"].to_numpy())
    m.update(seed=seed, severity=severity, duration=duration, kind=kind,
             percentile=percentile, min_consecutive=min_consecutive,
             n_test=len(fr), n_stressed=int(fr["stressed"].sum()))
    return m, fr
