"""
timeseries.py — Step 3: Time-Series Feature Engineering
Computes rolling statistics and temporal derivatives per tile.

For each tile (sorted by date):
- rolling_mean (window=3)
- rolling_std (window=3)
- diff (current - previous)
- percent_change

Output: features_timeseries.csv
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_timeseries_features(df_tile, window):
    """
    Computes time-series features for a single tile's DataFrame.
    The input must be sorted by date.
    
    Args:
        df_tile: DataFrame for one tile, sorted by date
        window: rolling window size
    
    Returns:
        df_tile with new columns added
    """
    # Make a copy to avoid modifying the original
    df = df_tile.copy()

    # --- Curve Smoothing (Savitzky-Golay) ---
    # Apply to p50 to create a biologically smooth phenological curve
    if len(df) >= 5:
        df["smoothed_p50"] = savgol_filter(df["p50"], window_length=5, polyorder=2)
    else:
        df["smoothed_p50"] = df["p50"]

    # --- Spatial Heterogeneity ---
    df["iqr"] = df["p75"] - df["p25"]
    
    # --- Contextual Global Feature ---
    if "global_p50" in df.columns:
        df["diff_from_global_median"] = df["smoothed_p50"] - df["global_p50"]
    else:
        df["diff_from_global_median"] = 0.0

    # --- Rolling statistics on the median ('p50') column ---
    # Rolling median: median of the last 'window' values
    df["rolling_median"] = (
        df["smoothed_p50"]
        .rolling(window=window, min_periods=1, center=False)
        .median()
    )

    # Rolling std: standard deviation of the last 'window' values
    df["rolling_std"] = (
        df["smoothed_p50"]
        .rolling(window=window, min_periods=1, center=False)
        .std()
    )

    # --- Temporal derivatives ---
    # Diff: how much 'p50' changed from the previous date
    df["diff"] = df["smoothed_p50"].diff()

    # Percent change: relative change from previous date
    # Use replace to handle division by zero safely
    previous_values = df["smoothed_p50"].shift(1)
    # Avoid division by zero
    safe_previous = previous_values.replace(0, np.nan)
    df["percent_change"] = (df["smoothed_p50"] - previous_values) / safe_previous

    # --- Handle first-row NaN values ---
    # Forward fill first, then backward fill as fallback
    df["rolling_std"] = df["rolling_std"].ffill().bfill().fillna(0)
    df["diff"] = df["diff"].ffill().bfill().fillna(0)
    df["percent_change"] = df["percent_change"].ffill().bfill().fillna(0)

    return df


def run_timeseries(df_raw, window):
    """
    Processes all tiles in the raw features DataFrame,
    adding time-series features per tile.
    
    Args:
        df_raw: DataFrame from features_raw.csv
        window: rolling window size from config
    
    Returns:
        df_ts: DataFrame with all original + time-series columns
    """
    all_tiles = []

    # Get unique tile IDs
    tile_ids = df_raw["tile_id"].unique()
    tile_ids.sort()

    # Calculate global medians for context features
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    global_medians = df_raw.groupby("date")["p50"].median().reset_index()
    global_medians.rename(columns={"p50": "global_p50"}, inplace=True)
    df_raw = df_raw.merge(global_medians, on="date", how="left")

    for tile_id in tile_ids:
        # Filter to this tile and sort by date
        df_tile = df_raw[df_raw["tile_id"] == tile_id].copy()
        df_tile = df_tile.sort_values("date").reset_index(drop=True)

        # Compute time-series features
        df_tile = compute_timeseries_features(df_tile, window)

        all_tiles.append(df_tile)
        print(f"  Tile {tile_id}: {len(df_tile)} time steps processed.")

    # Combine all tiles back together
    df_ts = pd.concat(all_tiles, ignore_index=True)

    return df_ts
