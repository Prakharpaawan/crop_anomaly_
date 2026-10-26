"""
visualization.py — Step 9: Per-Tile Visualization
Generates SEPARATE plot images for each panel per tile:
  1. Raw MSAVI time-series
  2. Rolling Mean with confidence band
  3. Temporal Change (diff bars)
  4. Relative Temporal Variance (percent change)
  5. Hybrid Anomaly Score (ISO + LOF + AE)
  6. Stress Overlay — actual .tif satellite image with detected stress

Each panel is saved as an individual PNG file inside a per-tile folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ═══════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════
COLOR_MAIN      = "#2196F3"   # Blue
COLOR_ROLLING   = "#FF9800"   # Orange
COLOR_DIFF      = "#9C27B0"   # Purple
COLOR_SCORE     = "#4CAF50"   # Green
COLOR_ANOMALY   = "#F44336"   # Red
COLOR_THRESHOLD = "#E91E63"   # Pink
COLOR_PCT       = "#00BCD4"   # Cyan


# ═══════════════════════════════════════════════════════════════════
# INDIVIDUAL PANEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _save_fig(fig, path):
    """Helper to save a figure and close it."""
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_panel_msavi(dates, df_tile, tile_id, output_dir, pop_stats=None):
    """Panel 1: Raw Vegetation Index (Mean MSAVI)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if pop_stats is not None:
        ax.fill_between(pop_stats["date"], pop_stats["pop_p10"], pop_stats["pop_p90"], 
                        color=COLOR_MAIN, alpha=0.15, label="Dataset 10th-90th Pct")
        ax.plot(pop_stats["date"], pop_stats["pop_median"], color=COLOR_MAIN, 
                linewidth=1, linestyle="--", label="Dataset Median")
                
    ax.plot(dates, df_tile["mean"], color="gray", alpha=0.5,
            linewidth=1, label="Raw Mean")
    if "smoothed_p50" in df_tile.columns:
        ax.plot(dates, df_tile["smoothed_p50"], color=COLOR_ROLLING,
                linewidth=1.8, marker="o", markersize=5, label="Smoothed Curve (S-G Filter)")
    else:
        ax.plot(dates, df_tile["mean"], color=COLOR_ROLLING,
                linewidth=1.8, marker="o", markersize=5, label="Tile Mean MSAVI")
    ax.set_ylabel("Mean MSAVI", fontsize=11)
    ax.set_title(f"Tile {tile_id} — Raw Vegetation Index (Mean MSAVI)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{tile_id}_1_msavi.png")
    _save_fig(fig, path)
    return path


def plot_panel_rolling(dates, means, rolling_median, rolling_std,
                       tile_id, output_dir, pop_stats=None):
    """Panel 2: Rolling Median with Confidence Band."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if pop_stats is not None:
        ax.fill_between(pop_stats["date"], pop_stats["pop_p10"], pop_stats["pop_p90"], 
                        color=COLOR_MAIN, alpha=0.15, label="Dataset 10th-90th Pct")
        ax.plot(pop_stats["date"], pop_stats["pop_median"], color=COLOR_MAIN, 
                linewidth=1, linestyle="--", label="Dataset Median")
                
    ax.plot(dates, means, color=COLOR_ROLLING,
            linewidth=1, alpha=0.5, label="Tile Raw Mean")

    if rolling_median is not None:
        ax.plot(dates, rolling_median, color=COLOR_ROLLING,
                linewidth=2, label="Tile Rolling Median")

    if rolling_std is not None and rolling_median is not None:
        ax.fill_between(
            dates,
            rolling_median - rolling_std,
            rolling_median + rolling_std,
            alpha=0.15, color=COLOR_ROLLING, label="±1 Rolling Std"
        )

    ax.set_ylabel("MSAVI", fontsize=11)
    ax.set_title(f"Tile {tile_id} — Rolling Median with Confidence Band",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{tile_id}_2_rolling_mean.png")
    _save_fig(fig, path)
    return path


def plot_panel_diff(dates, diff_values, tile_id, output_dir):
    """Panel 3: Temporal Change (diff bars)."""
    fig, ax = plt.subplots(figsize=(12, 4))

    colors_bar = [COLOR_ANOMALY if v < 0 else COLOR_SCORE for v in diff_values]
    ax.vlines(dates, ymin=0, ymax=diff_values,
              colors=colors_bar, alpha=0.7, linewidth=8)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Diff", fontsize=11)
    ax.set_title(f"Tile {tile_id} — Temporal Change (Current − Previous)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{tile_id}_3_temporal_change.png")
    _save_fig(fig, path)
    return path


def plot_panel_pct_change(dates, pct_values, tile_id, output_dir):
    """Panel 4: Relative Temporal Variance (percent change)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, pct_values * 100, color=COLOR_PCT,
            linewidth=1.5, marker="^", markersize=4, label="Percent Change")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Change (%)", fontsize=11)
    ax.set_title(f"Tile {tile_id} — Relative Temporal Variance",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{tile_id}_4_percent_change.png")
    _save_fig(fig, path)
    return path


def plot_panel_anomaly_score(result_dates, scores, labels,
                             tile_threshold, percentile,
                             tile_id, output_dir, ground_truth=None):
    """Panel 5: Hybrid Anomaly Score with per-tile threshold."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(result_dates, scores, color=COLOR_SCORE,
            linewidth=1.5, marker="s", markersize=4, label="Final Score")

    # Per-tile threshold line
    ax.axhline(y=tile_threshold, color=COLOR_THRESHOLD,
               linewidth=2, linestyle="--",
               label=f"Threshold (P{percentile})")

    # Highlight anomalies with red markers
    anomaly_mask = labels == 1
    if np.any(anomaly_mask):
        ax.scatter(
            result_dates[anomaly_mask],
            scores[anomaly_mask],
            color=COLOR_ANOMALY, s=80, zorder=5,
            edgecolors="black", linewidths=0.8,
            label=f"Anomaly ({int(np.sum(anomaly_mask))})"
        )

    # Highlight ground truth with blue X markers
    if ground_truth is not None:
        gt_mask = ground_truth == 1
        if np.any(gt_mask):
            ax.scatter(
                result_dates[gt_mask],
                scores[gt_mask],
                color="blue", marker="x", s=120, zorder=6,
                linewidths=2,
                label=f"Ground Truth ({int(np.sum(gt_mask))})"
            )

    ax.set_ylabel("Anomaly Score", fontsize=11)
    ax.set_title(f"Tile {tile_id} — Hybrid Anomaly Score (ISO + LOF + AE)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, f"{tile_id}_5_anomaly_score.png")
    _save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════
# STRESS OVERLAY — NEW
# ═══════════════════════════════════════════════════════════════════

def plot_stress_overlay(tile_id, date_str, clean_dir, output_dir):
    """
    For a detected anomaly date, loads the corresponding .tif image,
    normalizes it, creates a stress mask (lowest 20% of pixel values),
    and overlays it in red on the grayscale image.
    """
    try:
        import rasterio
    except ImportError:
        print(f"    WARNING: rasterio not installed, skipping stress overlay.")
        return None

    # Build the expected filename: YYYYMMDD_MSAVI.tif
    date_clean = str(date_str).split(" ")[0].replace("-", "")  # handle Timestamp
    tif_filename = f"{date_clean}_MSAVI.tif"
    tif_path = os.path.join(clean_dir, str(tile_id), tif_filename)

    if not os.path.exists(tif_path):
        return None

    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
    except Exception as e:
        print(f"    WARNING: Could not read {tif_path}: {e}")
        return None

    # Build valid mask
    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data)
    else:
        mask = ~np.isnan(data)

    if np.sum(mask) == 0:
        return None

    # Normalize image to [0, 1]
    valid_vals = data[mask]
    img_min = valid_vals.min()
    img_max = valid_vals.max()

    if img_max == img_min:
        return None

    img_normalized = np.clip((data - img_min) / (img_max - img_min), 0, 1)
    img_normalized[~mask] = 0  # nodata pixels = black

    # Create stress mask: lowest 20% of VALID pixel values = stressed
    stress_threshold = np.percentile(valid_vals, 20)
    stress_mask = (data < stress_threshold) & mask

    # Build RGB overlay
    # Grayscale base
    rgb = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)

    # Overlay stress regions in red with alpha blending
    alpha = 0.4
    rgb[stress_mask, 0] = rgb[stress_mask, 0] * (1 - alpha) + 1.0 * alpha  # Red
    rgb[stress_mask, 1] = rgb[stress_mask, 1] * (1 - alpha) + 0.0 * alpha  # Green
    rgb[stress_mask, 2] = rgb[stress_mask, 2] * (1 - alpha) + 0.0 * alpha  # Blue
    rgb[~mask] = [0.15, 0.15, 0.15]

    # Clean date for display
    date_display = str(date_str).split(" ")[0]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation="bicubic")
    ax.set_title(
        f"Tile {tile_id} — Stress Overlay ({date_display})\n"
        f"Red = low MSAVI (bottom 20%)",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")
    plt.tight_layout()

    date_tag = str(date_str).split(" ")[0].replace("-", "")
    save_path = os.path.join(output_dir, f"{tile_id}_6_stress_{date_tag}.png")
    _save_fig(fig, save_path)
    return save_path


# ═══════════════════════════════════════════════════════════════════
# COMBINED PANEL (still saved alongside individual panels)
# ═══════════════════════════════════════════════════════════════════

def plot_combined_tile(df_tile, results_tile, tile_id, output_dir,
                       threshold_percentile, tile_threshold, pop_stats=None):
    """
    Creates the classic 5-panel combined figure for one tile.
    Saved as {tile_id}_combined.png.
    """
    dates = df_tile["date"].values

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    fig.suptitle(f"Hybrid Crop Stress Detection — Tile {tile_id}",
                 fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: MSAVI
    ax1 = axes[0]
    if pop_stats is not None:
        ax1.fill_between(pop_stats["date"], pop_stats["pop_p10"], pop_stats["pop_p90"], 
                        color=COLOR_MAIN, alpha=0.15, label="Dataset 10th-90th Pct")
        ax1.plot(pop_stats["date"], pop_stats["pop_median"], color=COLOR_MAIN, 
                linewidth=1, linestyle="--", label="Dataset Median")
    ax1.plot(dates, df_tile["mean"], color="gray", alpha=0.5,
             linewidth=1, label="Raw Mean")
    if "smoothed_p50" in df_tile.columns:
        ax1.plot(dates, df_tile["smoothed_p50"], color=COLOR_ROLLING,
                 linewidth=1.5, marker="o", markersize=4, label="Smoothed Curve (S-G Filter)")
    else:
        ax1.plot(dates, df_tile["mean"], color=COLOR_ROLLING,
                 linewidth=1.5, marker="o", markersize=4, label="Tile Mean MSAVI")
    ax1.set_ylabel("Mean MSAVI", fontsize=11)
    ax1.set_title("Raw Vegetation Index (Mean MSAVI)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rolling Mean
    ax2 = axes[1]
    if pop_stats is not None:
        ax2.fill_between(pop_stats["date"], pop_stats["pop_p10"], pop_stats["pop_p90"], 
                        color=COLOR_MAIN, alpha=0.15, label="Dataset 10th-90th Pct")
        ax2.plot(pop_stats["date"], pop_stats["pop_median"], color=COLOR_MAIN, 
                linewidth=1, linestyle="--", label="Dataset Median")
    ax2.plot(dates, df_tile["mean"], color=COLOR_ROLLING,
             linewidth=1, alpha=0.5, label="Tile Raw Mean")
    if "rolling_median" in df_tile.columns:
        ax2.plot(dates, df_tile["rolling_median"], color=COLOR_ROLLING,
                 linewidth=2, label="Tile Rolling Median")
    if "rolling_std" in df_tile.columns:
        rm = df_tile["rolling_median"].values
        rs = df_tile["rolling_std"].values
        ax2.fill_between(dates, rm - rs, rm + rs,
                         alpha=0.15, color=COLOR_ROLLING, label="±1 Tile Rolling Std")
    ax2.set_ylabel("MSAVI", fontsize=11)
    ax2.set_title("Rolling Median with Confidence Band", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Diff
    ax3 = axes[2]
    diff_values = df_tile["diff"].values
    colors_bar = [COLOR_ANOMALY if v < 0 else COLOR_SCORE for v in diff_values]
    ax3.vlines(dates, ymin=0, ymax=diff_values,
               colors=colors_bar, alpha=0.7, linewidth=8)
    ax3.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax3.set_ylabel("Diff", fontsize=11)
    ax3.set_title("Temporal Change (Current − Previous)", fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Percent Change
    ax4 = axes[3]
    if "percent_change" in df_tile.columns:
        pct = df_tile["percent_change"].values * 100
        ax4.plot(dates, pct, color=COLOR_PCT,
                 linewidth=1.5, marker="^", markersize=4, label="Percent Change")
        ax4.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax4.set_ylabel("Change (%)", fontsize=11)
        ax4.set_title("Relative Temporal Variance", fontsize=12)
        ax4.legend(loc="upper right", fontsize=9)
        ax4.grid(True, alpha=0.3)

    # Panel 5: Anomaly Score
    ax5 = axes[4]
    if not results_tile.empty and "final_score" in results_tile.columns:
        rd = results_tile["date"].values
        sc = results_tile["final_score"].values
        lb = results_tile["label"].values

        ax5.plot(rd, sc, color=COLOR_SCORE,
                 linewidth=1.5, marker="s", markersize=4, label="Final Score")
        ax5.axhline(y=tile_threshold, color=COLOR_THRESHOLD,
                     linewidth=2, linestyle="--",
                     label=f"Threshold (P{threshold_percentile})")
        anomaly_mask = lb == 1
        if np.any(anomaly_mask):
            ax5.scatter(rd[anomaly_mask], sc[anomaly_mask],
                        color=COLOR_ANOMALY, s=80, zorder=5,
                        edgecolors="black", linewidths=0.8,
                        label=f"Anomaly ({int(np.sum(anomaly_mask))})")
        if "ground_truth" in results_tile.columns:
            gt_mask = results_tile["ground_truth"].values == 1
            if np.any(gt_mask):
                ax5.scatter(rd[gt_mask], sc[gt_mask],
                            color="blue", marker="x", s=120, zorder=6,
                            linewidths=2,
                            label=f"Ground Truth ({int(np.sum(gt_mask))})")

    ax5.set_ylabel("Anomaly Score", fontsize=11)
    ax5.set_title("Hybrid Anomaly Score (ISO + LOF + AE)", fontsize=12)
    ax5.legend(loc="upper right", fontsize=9)
    ax5.grid(True, alpha=0.3)

    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, f"{tile_id}_combined.png")
    _save_fig(fig, path)
    return path


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def run_visualization(df_ts, results_df, plots_dir,
                      threshold_percentile, clean_dir):
    """
    Generates per-tile plots for all tiles in the dataset.
    Each panel is saved as a separate image, plus one combined view.
    For anomaly dates, a stress overlay of the .tif is also generated.
    """
    import shutil
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Compute population statistics for context background
    df_pop = df_ts.copy()
    df_pop["date"] = pd.to_datetime(df_pop["date"])
    pop_stats = df_pop.groupby("date")["mean"].agg(
        pop_median="median",
        pop_p10=lambda x: np.percentile(x, 10),
        pop_p90=lambda x: np.percentile(x, 90)
    ).reset_index()
    pop_stats = pop_stats.sort_values("date").reset_index(drop=True)

    tile_ids = df_ts["tile_id"].unique()
    tile_ids.sort()

    total_overlays = 0

    for tile_id in tile_ids:
        # Create per-tile subfolder
        tile_dir = os.path.join(plots_dir, str(tile_id))
        os.makedirs(tile_dir, exist_ok=True)

        # Get data for this tile
        df_tile = df_ts[df_ts["tile_id"] == tile_id].copy()
        results_tile = results_df[results_df["tile_id"] == tile_id].copy()

        df_tile["date"] = pd.to_datetime(df_tile["date"])
        df_tile = df_tile.sort_values("date").reset_index(drop=True)

        if not results_tile.empty:
            results_tile["date"] = pd.to_datetime(results_tile["date"])
            results_tile = results_tile.sort_values("date").reset_index(drop=True)

        dates = df_tile["date"].values

        # ── Per-tile threshold ──
        if not results_tile.empty and "final_score" in results_tile.columns:
            tile_threshold = np.percentile(
                results_tile["final_score"].values, threshold_percentile
            )
        else:
            tile_threshold = 0.5

        # ── Panel 1: MSAVI ──
        plot_panel_msavi(dates, df_tile, tile_id, tile_dir, pop_stats)

        # ── Panel 2: Rolling Median ──
        rm = df_tile["rolling_median"].values if "rolling_median" in df_tile.columns else None
        rs = df_tile["rolling_std"].values if "rolling_std" in df_tile.columns else None
        plot_panel_rolling(dates, df_tile["mean"].values, rm, rs, tile_id, tile_dir, pop_stats)

        # ── Panel 3: Diff ──
        plot_panel_diff(dates, df_tile["diff"].values, tile_id, tile_dir)

        # ── Panel 4: Percent Change ──
        if "percent_change" in df_tile.columns:
            plot_panel_pct_change(dates, df_tile["percent_change"].values,
                                 tile_id, tile_dir)

        # ── Panel 5: Anomaly Score ──
        if not results_tile.empty:
            gt = results_tile["ground_truth"].values if "ground_truth" in results_tile.columns else None
            plot_panel_anomaly_score(
                results_tile["date"].values,
                results_tile["final_score"].values,
                results_tile["label"].values,
                tile_threshold, threshold_percentile,
                tile_id, tile_dir, ground_truth=gt
            )

        # ── Combined view ──
        plot_combined_tile(df_tile, results_tile, tile_id, tile_dir,
                           threshold_percentile, tile_threshold, pop_stats)

        # ── Panel 6: Stress Overlays ──
        overlay_paths = []
        if not results_tile.empty:
            anomaly_rows = results_tile[results_tile["label"] == 1]
            for _, row in anomaly_rows.iterrows():
                overlay_path = plot_stress_overlay(
                    tile_id, row["date"], clean_dir, tile_dir
                )
                if overlay_path:
                    overlay_paths.append(overlay_path)
                    total_overlays += 1

        # Combine overlays with the main plot
        combined_path = os.path.join(tile_dir, f"{tile_id}_combined.png")
        if overlay_paths and os.path.exists(combined_path):
            try:
                from PIL import Image
                images = [Image.open(combined_path)]
                for p in overlay_paths:
                    images.append(Image.open(p))
                
                widths, heights = zip(*(i.size for i in images))
                max_width = max(widths)
                total_height = sum(heights)
                
                new_im = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                y_offset = 0
                for im in images:
                    x_offset = (max_width - im.width) // 2
                    new_im.paste(im, (x_offset, y_offset))
                    y_offset += im.height
                new_im.save(combined_path)
            except Exception as e:
                print(f"    WARNING: Could not append images to combined plot: {e}")

        print(f"  Saved plots for tile: {tile_id}")

    print(f"\n  Total stress overlay images generated: {total_overlays}")
