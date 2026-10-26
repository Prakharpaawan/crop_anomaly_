"""
cleaning.py — Step 1: Data Cleaning
Filters raw MSAVI satellite images using spatial quality metrics,
selects the best temporally-distributed images per tile,
and produces a cleaning report.

Logic adapted from src/image_quality_filter.py with added:
- try-except for corrupted .tif files
- Per-tile rejected image logging
- Cleaning report CSV output
"""

import os
import shutil
import numpy as np
import pandas as pd
import random
import rasterio


def parse_date_from_filename(filename):
    """
    Extracts the date from a filename like '20190104_MSAVI.tif'.
    Returns a string in YYYY-MM-DD format.
    """
    base = os.path.basename(filename)
    date_str = base.split('_')[0]

    # Validate that the date part is 8 digits
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def compute_image_stats(filepath):
    """
    Reads a single MSAVI .tif and computes spatial quality metrics:
    - std: standard deviation of valid pixels
    - gradient: mean spatial gradient strength (texture measure)
    - unique_ratio: fraction of unique pixel values
    - valid_ratio: fraction of non-nodata pixels
    - mean, min, max: basic statistics
    
    Returns a dict of stats, or None if the file is corrupt.
    """
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
    except Exception as e:
        # Corrupted or unreadable .tif
        print(f"    WARNING: Could not read {filepath}: {e}")
        return None

    # Build valid-pixel mask
    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data)
    else:
        mask = ~np.isnan(data)

    valid_pixels = int(np.sum(mask))
    total_pixels = data.size
    valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

    # Edge case: no valid pixels at all
    if valid_pixels == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "valid_ratio": 0.0,
            "unique_ratio": 0.0,
            "gradient": 0.0,
        }

    values = data[mask]
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    # Unique pixel ratio — how many distinct values exist
    unique_pixels = len(np.unique(values))
    unique_ratio = unique_pixels / total_pixels

    # Spatial gradient strength (texture / edge measure)
    # Fill masked pixels with the mean to avoid gradient artifacts at edges
    data_filled = np.copy(data)
    data_filled[~mask] = mean_val
    gx, gy = np.gradient(data_filled)
    gradient_strength = float(np.mean(np.abs(gx[mask]) + np.abs(gy[mask])))

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "gradient": gradient_strength,
    }


def filter_and_score_images(tile_dir, min_std, min_valid_ratio,
                            min_unique_ratio, min_gradient):
    """
    Scans a tile directory, scores every MSAVI image, and filters out
    low-quality ones using the given thresholds.

    Returns:
        df_all: DataFrame of ALL images with stats (for reporting)
        df_filtered: DataFrame of images that pass quality thresholds
    """
    records = []

    # Iterate over every file in the tile directory
    for filename in os.listdir(tile_dir):
        # Only process MSAVI .tif files
        if not filename.endswith("_MSAVI.tif"):
            continue

        filepath = os.path.join(tile_dir, filename)
        date_str = parse_date_from_filename(filename)
        stats = compute_image_stats(filepath)

        # Skip if the file was unreadable
        if stats is None:
            continue

        stats["date"] = date_str
        stats["filename"] = filename
        stats["filepath"] = filepath
        records.append(stats)

    df_all = pd.DataFrame(records)

    # Edge case: no valid images found
    if df_all.empty:
        return df_all, df_all

    df_all["date"] = pd.to_datetime(df_all["date"])

    # --- Apply quality thresholds ---
    df_all["rejection_reason"] = "None"
    df_all["status"] = "Selected"

    # Identify rejections
    df_all.loc[df_all["std"] < min_std, "rejection_reason"] = "low_std"
    df_all.loc[(df_all["rejection_reason"] == "None") & (df_all["valid_ratio"] < min_valid_ratio), "rejection_reason"] = "low_valid_ratio"
    df_all.loc[(df_all["rejection_reason"] == "None") & (df_all["unique_ratio"] < min_unique_ratio), "rejection_reason"] = "low_unique_ratio"
    df_all.loc[(df_all["rejection_reason"] == "None") & (df_all["gradient"] < min_gradient), "rejection_reason"] = "low_gradient"
    
    df_all.loc[df_all["rejection_reason"] != "None", "status"] = "Rejected"

    df_filtered = df_all[df_all["status"] == "Selected"].copy()

    # Fallback: if filtering removed everything, take top half by std
    if df_filtered.empty:
        count = max(1, len(df_all) // 2)
        df_filtered = df_all.nlargest(count, "std").copy()
        df_all.loc[df_filtered.index, "status"] = "Selected"
        df_all.loc[df_filtered.index, "rejection_reason"] = "None (Fallback)"

    # --- Compute quality score (weighted combination) ---
    std_max = df_filtered["std"].max()
    grad_max = df_filtered["gradient"].max()

    # Normalize to [0, 1] range
    std_norm = df_filtered["std"] / std_max if std_max > 0 else 0
    grad_norm = df_filtered["gradient"] / grad_max if grad_max > 0 else 0

    # Weighted quality score
    df_filtered["quality_score"] = (
        0.4 * std_norm +
        0.4 * grad_norm +
        0.2 * df_filtered["unique_ratio"]
    )

    return df_all, df_filtered


def select_top_distributed(df, target_count, min_days):
    """
    Greedily selects highest-scoring images while enforcing
    minimum temporal spacing of min_days between selections.
    This ensures coverage across the whole growing season.
    """
    # If we already have fewer images than target, return all
    if df.empty or len(df) <= target_count:
        return df.sort_values("date").reset_index(drop=True)

    df_sorted = df.sort_values("quality_score", ascending=False).reset_index(drop=True)
    selected = []

    # First pass: pick highest-score images with spacing constraint
    for idx in range(len(df_sorted)):
        if len(selected) >= target_count:
            break

        candidate_date = df_sorted.loc[idx, "date"]

        # Check if this date is too close to any already-selected date
        too_close = False
        for sel_idx in selected:
            days_apart = abs((candidate_date - df_sorted.loc[sel_idx, "date"]).days)
            if days_apart < min_days:
                too_close = True
                break

        if not too_close:
            selected.append(idx)

    # Second pass: fill remaining slots without spacing constraint
    for idx in range(len(df_sorted)):
        if len(selected) >= target_count:
            break
        if idx not in selected:
            selected.append(idx)

    result = df_sorted.loc[selected].sort_values("date").reset_index(drop=True)
    return result


def run_cleaning(raw_dir, clean_dir, cleaning_config, detailed_report_path=None):
    """
    Main cleaning function. Processes all tiles in raw_dir,
    filters and selects best images, copies them to clean_dir,
    and returns a cleaning report DataFrame.
    
    Args:
        raw_dir: path to archive/lombardia/data2019/
        clean_dir: path to dataset_clean_2/
        cleaning_config: dict with threshold keys from config.py
        detailed_report_path: optional path to save detailed per-image report
    
    Returns:
        report_df: DataFrame with per-tile cleaning statistics
    """
    # Create output directory
    os.makedirs(clean_dir, exist_ok=True)

    # Gather all tile folders
    all_tiles = []
    for name in os.listdir(raw_dir):
        full_path = os.path.join(raw_dir, name)
        if os.path.isdir(full_path):
            all_tiles.append(name)

    all_tiles.sort()
    print(f"  Found {len(all_tiles)} tiles in raw dataset.")

    # Limit to max_tiles if configured (random sample for diversity)
    max_tiles = cleaning_config.get("max_tiles", None)
    if max_tiles is not None and len(all_tiles) > max_tiles:
        random.seed(42)  # Reproducible selection
        all_tiles = sorted(random.sample(all_tiles, max_tiles))
        print(f"  Randomly selected {max_tiles} tiles for processing.")

    print()

    report_rows = []
    detailed_report_rows = []

    for tile_name in all_tiles:
        raw_tile_path = os.path.join(raw_dir, tile_name)
        clean_tile_path = os.path.join(clean_dir, tile_name)
        os.makedirs(clean_tile_path, exist_ok=True)

        print(f"  Cleaning tile: {tile_name}...")

        # Score and filter all images in this tile
        df_all, df_filtered = filter_and_score_images(
            tile_dir=raw_tile_path,
            min_std=cleaning_config["min_std"],
            min_valid_ratio=cleaning_config["min_valid_ratio"],
            min_unique_ratio=cleaning_config["min_unique_ratio"],
            min_gradient=cleaning_config["min_gradient"],
        )

        # Edge case: no images at all
        if df_all.empty:
            print(f"    No MSAVI images found in tile {tile_name}.")
            report_rows.append({
                "tile_id": tile_name,
                "total_images": 0,
                "selected_images": 0,
                "rejected_images": 0,
            })
            continue

        # Select top images with temporal distribution
        df_selected = select_top_distributed(
            df=df_filtered,
            target_count=cleaning_config["target_count"],
            min_days=cleaning_config["min_days_apart"],
        )

        # Copy selected images to the clean directory
        for _, row in df_selected.iterrows():
            dest_path = os.path.join(clean_tile_path, row["filename"])
            shutil.copy2(row["filepath"], dest_path)

        # Update detailed dataframe with temporal rejections
        selected_filenames = set(df_selected["filename"])
        temporal_mask = (df_all["status"] == "Selected") & (~df_all["filename"].isin(selected_filenames))
        df_all.loc[temporal_mask, "rejection_reason"] = "temporal_spacing"
        df_all.loc[temporal_mask, "status"] = "Rejected"

        df_all["tile_id"] = tile_name
        detailed_report_rows.append(df_all)

        total = len(df_all)
        selected = len(df_selected)
        rejected = total - selected

        print(f"    Total: {total} | Selected: {selected} | Rejected: {rejected}")

        report_rows.append({
            "tile_id": tile_name,
            "total_images": total,
            "selected_images": selected,
            "rejected_images": rejected,
        })

    report_df = pd.DataFrame(report_rows)

    if detailed_report_path and detailed_report_rows:
        detailed_df = pd.concat(detailed_report_rows, ignore_index=True)
        detailed_df.to_csv(detailed_report_path, index=False)
        print(f"\n  Detailed cleaning report saved: {detailed_report_path}")

    return report_df
