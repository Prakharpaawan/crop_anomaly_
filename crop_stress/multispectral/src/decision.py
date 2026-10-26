"""
decision.py — turning a score into an alarm.

Two rules, in order:

1. A threshold. Either per-field (each field judged against its own history) or
   one global cut-off.
2. A persistence requirement. The score must stay above the threshold for N
   consecutive acquisitions before an alarm fires.

Both persistence rules here are **causal**: the decision at observation j uses
only observations up to and including j. The earlier pipeline also offered a
sliding 2-of-3 window that inspected j+1, which cannot be deployed in a
near-real-time system; it is deliberately not reproduced.
"""
import numpy as np


def calibrate_thresholds(scores, tile_ids, percentile, per_field=True):
    """
    Learn the decision boundary from CLEAN data.

    Returned as a dict so the same boundary can be applied unchanged to
    evaluation data. This is the key difference from the earlier pipeline,
    which recomputed the percentile on the batch it was scoring — including
    the injected anomalies.
    """
    if not per_field:
        return {"__global__": float(np.percentile(scores, percentile))}
    out = {}
    for t in np.unique(tile_ids):
        s = scores[tile_ids == t]
        if s.size:
            out[t] = float(np.percentile(s, percentile))
    return out


def apply_thresholds(scores, tile_ids, thresholds, global_fallback=None):
    """Boolean exceedance per observation, using a pre-calibrated boundary."""
    if "__global__" in thresholds:
        return scores > thresholds["__global__"]

    if global_fallback is None:
        vals = list(thresholds.values())
        global_fallback = float(np.median(vals)) if vals else np.inf

    thr = np.fromiter(
        (thresholds.get(t, global_fallback) for t in tile_ids),
        dtype=float, count=len(tile_ids),
    )
    return scores > thr


def persist(raw_flags, tile_ids, dates, min_consecutive):
    """
    Causal N-consecutive persistence.

    An alarm fires at observation j only if j and the N-1 observations
    immediately before it (within the same field, in date order) all exceeded
    the threshold. Nothing after j is consulted.
    """
    out = np.zeros(len(raw_flags), dtype=int)
    if min_consecutive <= 1:
        return raw_flags.astype(int)

    order = np.lexsort((dates, tile_ids))
    tiles_s = np.asarray(tile_ids)[order]
    flags_s = np.asarray(raw_flags)[order]

    run = 0
    prev_tile = None
    fired = np.zeros(len(flags_s), dtype=int)
    for i in range(len(flags_s)):
        if tiles_s[i] != prev_tile:
            run = 0
            prev_tile = tiles_s[i]
        run = run + 1 if flags_s[i] else 0
        if run >= min_consecutive:
            fired[i] = 1
    out[order] = fired
    return out


def decide(scores, tile_ids, dates, thresholds, min_consecutive):
    """Full decision path: threshold, then persistence."""
    raw = apply_thresholds(scores, tile_ids, thresholds)
    return persist(raw, tile_ids, dates, min_consecutive), raw
