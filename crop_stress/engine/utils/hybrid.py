"""
hybrid.py — Step 7: Hybrid Model Scoring
Combines Isolation Forest, LOF, and Autoencoder scores into
a single final anomaly score.

Improvements over previous version:
  - Per-tile thresholding (instead of global)
  - Persistence rule: anomaly confirmed only if consecutive
  - Weights: AE 0.5, ISO 0.3, LOF 0.2
"""

import numpy as np
import pandas as pd


def min_max_normalize(scores):
    """
    Normalizes an array of scores to the [0, 1] range.
    Handles the edge case where all values are identical.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    # Edge case: all scores are the same
    if max_val == min_val:
        return np.zeros_like(scores)

    normalized = (scores - min_val) / (max_val - min_val)
    return normalized


def apply_threshold_and_persistence(scores, tile_ids, config):
    """
    Applies adaptive per-tile thresholding and configurable temporal persistence.
    Returns (labels, total_anomalies).
    """
    percentile = config.get("threshold_percentile", 90)
    use_persistence = config.get("use_persistence", False)
    min_consecutive = config.get("min_consecutive_anomalies", 2 if use_persistence else 1)

    labels = np.zeros(len(scores), dtype=int)
    unique_tiles = np.unique(tile_ids)
    total_anomalies = 0

    for tile in unique_tiles:
        tile_mask = tile_ids == tile
        tile_scores = scores[tile_mask]
        tile_indices = np.where(tile_mask)[0]

        # Adaptive per-tile threshold
        tile_threshold = np.percentile(tile_scores, percentile)
        raw_flags = tile_scores > tile_threshold

        if min_consecutive > 1 and len(tile_scores) >= min_consecutive:
            # Check for N consecutive anomalies
            final_flags = np.zeros(len(tile_scores), dtype=bool)
            
            # Count consecutive hits
            consec_count = 0
            for j in range(len(tile_scores)):
                if raw_flags[j]:
                    consec_count += 1
                else:
                    # If we broke the chain, check if we had enough to flag the past sequence
                    if consec_count >= min_consecutive:
                        final_flags[j - consec_count : j] = True
                    consec_count = 0
                    
            # Handle if the sequence ends on a hit
            if consec_count >= min_consecutive:
                final_flags[len(tile_scores) - consec_count : len(tile_scores)] = True
                
        else:
            final_flags = raw_flags

        # Apply labels back to global array
        for j, idx in enumerate(tile_indices):
            if final_flags[j]:
                labels[idx] = 1

        total_anomalies += int(np.sum(final_flags))

    return labels, total_anomalies

def apply_global_threshold_and_persistence(scores, tile_ids, config):
    """
    Applies a fixed/global threshold (across all farms) and temporal persistence.
    Returns (labels, total_anomalies).
    """
    percentile = config.get("threshold_percentile", 90)
    min_consecutive = config.get("min_consecutive_anomalies", 2 if config.get("use_persistence", False) else 1)

    # Compute one FIXED global threshold for all data
    global_threshold = np.percentile(scores, percentile)
    
    labels = np.zeros(len(scores), dtype=int)
    unique_tiles = np.unique(tile_ids)
    total_anomalies = 0

    for tile in unique_tiles:
        tile_mask = tile_ids == tile
        tile_scores = scores[tile_mask]
        tile_indices = np.where(tile_mask)[0]

        # Use global threshold instead of per-tile
        raw_flags = tile_scores > global_threshold

        if min_consecutive > 1 and len(tile_scores) >= min_consecutive:
            final_flags = np.zeros(len(tile_scores), dtype=bool)
            consec_count = 0
            for j in range(len(tile_scores)):
                if raw_flags[j]:
                    consec_count += 1
                else:
                    if consec_count >= min_consecutive:
                        final_flags[j - consec_count : j] = True
                    consec_count = 0
            if consec_count >= min_consecutive:
                final_flags[len(tile_scores) - consec_count : len(tile_scores)] = True
        else:
            final_flags = raw_flags

        for j, idx in enumerate(tile_indices):
            if final_flags[j]:
                labels[idx] = 1

        total_anomalies += int(np.sum(final_flags))

    return labels, total_anomalies


def run_hybrid(df_ts, iso_scores, ae_errors, lof_scores, hybrid_config):
    """
    Combines Isolation Forest, LOF, and Autoencoder scores into a
    single hybrid anomaly score with per-tile thresholding and
    a persistence rule.

    Args:
        df_ts:        DataFrame with tile_id and date columns
        iso_scores:   numpy array of Isolation Forest decision_function scores
        ae_errors:    numpy array of Autoencoder reconstruction errors
        lof_scores:   numpy array of LOF outlier scores (higher = more anomalous)
        hybrid_config: dict with iso_weight, lof_weight, ae_weight,
                       threshold_percentile, use_persistence

    Returns:
        results_df: DataFrame with columns:
            tile_id, date, iso_score, lof_score, ae_error, final_score, label
    """
    # --- Step 1: Normalize all three scores to [0, 1] ---
    # For Isolation Forest: lower score = more anomalous → invert
    iso_inverted = -iso_scores
    iso_normalized = min_max_normalize(iso_inverted)

    # For LOF: already higher = more anomalous (after negation in run_lof)
    lof_normalized = min_max_normalize(lof_scores)

    # For Autoencoder: higher error = more anomalous (already correct)
    ae_normalized = min_max_normalize(ae_errors)

    print(f"  ISO scores — Min: {iso_normalized.min():.4f}, "
          f"Max: {iso_normalized.max():.4f}, "
          f"Mean: {iso_normalized.mean():.4f}")
    print(f"  LOF scores — Min: {lof_normalized.min():.4f}, "
          f"Max: {lof_normalized.max():.4f}, "
          f"Mean: {lof_normalized.mean():.4f}")
    print(f"  AE  errors — Min: {ae_normalized.min():.4f}, "
          f"Max: {ae_normalized.max():.4f}, "
          f"Mean: {ae_normalized.mean():.4f}")

    # --- Step 2: Weighted combination ---
    iso_w = hybrid_config["iso_weight"]
    lof_w = hybrid_config["lof_weight"]
    ae_w  = hybrid_config["ae_weight"]
    final_score = ae_w * ae_normalized + iso_w * iso_normalized + lof_w * lof_normalized

    print(f"  Weights — AE: {ae_w}, ISO: {iso_w}, LOF: {lof_w}")

    # --- Step 3: Per-tile thresholding + persistence ---
    tile_ids = df_ts["tile_id"].values
    labels, total_anomalies = apply_threshold_and_persistence(final_score, tile_ids, hybrid_config)

    print(f"\n  Total anomalies detected: {total_anomalies} / {len(labels)}")

    # --- Build results DataFrame ---
    results_df = pd.DataFrame({
        "tile_id":     df_ts["tile_id"].values,
        "date":        df_ts["date"].values,
        "iso_score":   iso_normalized,
        "lof_score":   lof_normalized,
        "ae_error":    ae_normalized,
        "final_score": final_score,
        "label":       labels,
    })

    return results_df


def apply_sliding_window_persistence(scores, tile_ids, config):
    """
    Sliding-window persistence: flag a point as anomalous if at least
    2 out of 3 consecutive points exceed the threshold.
    Less aggressive than strict consecutive — handles noisy gaps.
    """
    percentile = config.get("threshold_percentile", 90)
    
    labels = np.zeros(len(scores), dtype=int)
    unique_tiles = np.unique(tile_ids)
    total_anomalies = 0

    for tile in unique_tiles:
        tile_mask = tile_ids == tile
        tile_scores = scores[tile_mask]
        tile_indices = np.where(tile_mask)[0]

        tile_threshold = np.percentile(tile_scores, percentile)
        raw_flags = tile_scores > tile_threshold

        final_flags = np.zeros(len(tile_scores), dtype=bool)
        
        for j in range(len(tile_scores)):
            # Look at window [j-2, j-1, j] (backward-only, no future leak)
            start = max(0, j - 2)
            end = j + 1  # exclusive, so includes j
            window = raw_flags[start:end]
            
            # If 2+ out of the window are flagged, flag this point
            if np.sum(window) >= 2:
                final_flags[j] = True

        for j, idx in enumerate(tile_indices):
            if final_flags[j]:
                labels[idx] = 1

        total_anomalies += int(np.sum(final_flags))

    return labels, total_anomalies
