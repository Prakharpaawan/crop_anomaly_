"""
feature_extraction.py — Step 2: Raw Feature Extraction
Reads each cleaned MSAVI .tif, clips extreme values,
replaces NaN with mean, and computes per-image statistics.

Features extracted per image:
  Existing : mean, std, min, max, valid_pixel_ratio
  New      : p25, p50, p75 (percentiles), slope (linear trend)

Output: features_raw.csv
"""

import os
import numpy as np
import pandas as pd
import rasterio


def parse_date_from_filename(filename):
    """
    Extracts the date from a filename like '20190104_MSAVI.tif'.
    Returns a string in YYYY-MM-DD format.
    """
    base = os.path.basename(filename)
    date_str = base.split('_')[0]

    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def extract_single_image(filepath, clip_lower, clip_upper):
    """
    Reads a single .tif file, handles NaN/nodata, clips outliers,
    and computes summary statistics.
    
    Args:
        filepath: path to the .tif file
        clip_lower: lower percentile for clipping (e.g. 1)
        clip_upper: upper percentile for clipping (e.g. 99)
    
    Returns:
        dict of features, or None if the file is unreadable/empty
    """
    # Try to read the file safely
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
    except Exception as e:
        print(f"    WARNING: Could not read {filepath}: {e}")
        return None

    # Build a mask for valid pixels
    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data)
    else:
        mask = ~np.isnan(data)

    valid_pixels = int(np.sum(mask))
    total_pixels = data.size

    # Edge case: all pixels are invalid
    if valid_pixels == 0:
        return None

    # Extract valid values
    values = data[mask].copy()

    # Clip extreme values at the specified percentiles
    lower_bound = np.percentile(values, clip_lower)
    upper_bound = np.percentile(values, clip_upper)
    values = np.clip(values, lower_bound, upper_bound)

    # Replace any remaining NaN with the mean (safety net)
    mean_val = np.nanmean(values)
    values = np.where(np.isnan(values), mean_val, values)

    # Compute statistics
    valid_pixel_ratio = valid_pixels / total_pixels

    # ── New: percentile features ──────────────────────────────────
    p25 = float(np.percentile(values, 25))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))

    # ── New: Coefficient of Variation (CV = std/mean) ──────────────
    # Measures spatial uniformity of the field. Healthy crops have low CV
    # (uniform canopy), stressed crops often have high CV (patchy damage).
    cv = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0

    return {
        # ── Existing features (unchanged) ──
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "valid_pixel_ratio": float(valid_pixel_ratio),
        # ── New features ──
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "cv": cv,
    }


def run_feature_extraction(clean_dir, clip_lower, clip_upper):
    """
    Iterates over all tiles in the clean dataset directory,
    extracts features from each MSAVI image, and returns
    a combined DataFrame.
    
    Args:
        clean_dir: path to dataset_clean_2/
        clip_lower: lower percentile for outlier clipping
        clip_upper: upper percentile for outlier clipping
    
    Returns:
        df: DataFrame with columns
            [tile_id, date, mean, std, min, max, valid_pixel_ratio, p25, p50, p75, slope]
    """
    all_records = []

    # Get all tile folders
    tile_names = []
    for name in os.listdir(clean_dir):
        full_path = os.path.join(clean_dir, name)
        if os.path.isdir(full_path):
            tile_names.append(name)
    tile_names.sort()

    print(f"  Found {len(tile_names)} tiles in clean dataset.\n")

    for tile_name in tile_names:
        tile_path = os.path.join(clean_dir, tile_name)
        print(f"  Extracting features for tile: {tile_name}...")

        # Get all MSAVI files in this tile
        msavi_files = []
        for filename in os.listdir(tile_path):
            if filename.endswith("_MSAVI.tif"):
                msavi_files.append(filename)
        msavi_files.sort()

        file_count = 0

        for filename in msavi_files:
            filepath = os.path.join(tile_path, filename)
            date_str = parse_date_from_filename(filename)

            # Extract features from this image
            features = extract_single_image(filepath, clip_lower, clip_upper)

            # Skip if extraction failed
            if features is None:
                print(f"    Skipped: {filename} (unreadable or empty)")
                continue

            # Add tile and date identifiers
            features["tile_id"] = tile_name
            features["date"] = date_str

            all_records.append(features)
            file_count += 1

        print(f"    Extracted features from {file_count} images.")

    # Build the final DataFrame
    df = pd.DataFrame(all_records)

    if not df.empty:
        # Reorder columns for clarity
        df["date"] = pd.to_datetime(df["date"])
        df = df[["tile_id", "date",
                  "mean", "std", "min", "max", "valid_pixel_ratio",
                  "p25", "p50", "p75", "cv"]]
        df = df.sort_values(["tile_id", "date"]).reset_index(drop=True)

    return df
