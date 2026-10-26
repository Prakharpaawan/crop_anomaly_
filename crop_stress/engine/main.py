"""
main.py — Pipeline Orchestrator
Runs all 9 steps of the Hybrid Crop Stress Detection Pipeline.

Usage:
    cd hybrid_pipeline
    python main.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Add the hybrid_pipeline directory to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RANDOM_SEED, RAW_DATASET_DIR, CLEAN_DATASET_DIR,
    OUTPUT_DIR, MODELS_DIR, PLOTS_DIR,
    CLEANING, FEATURE_EXTRACTION, TIMESERIES,
    PREPARATION, ISOLATION_FOREST, LOF, AUTOENCODER,
    HYBRID, EVALUATION, FILES,
)


def print_header(step_num, title):
    """Prints a formatted step header."""
    print("\n" + "=" * 60)
    print(f"  STEP {step_num}: {title}")
    print("=" * 60 + "\n")


def main():
    """
    Main entry point for the Hybrid Crop Stress Detection Pipeline.
    Executes all steps sequentially with error handling.
    """
    start_time = time.time()

    print("\n" + "+" + "=" * 58 + "+")
    print("|  HYBRID CROP STRESS DETECTION PIPELINE                    |")
    print("|  Sentinel-2 MSAVI Time Series (2019)                      |")
    print("+" + "=" * 58 + "+\n")

    # Set global random seed
    np.random.seed(RANDOM_SEED)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Verify raw dataset exists
    if not os.path.exists(RAW_DATASET_DIR):
        print(f"ERROR: Raw dataset not found at: {RAW_DATASET_DIR}")
        print("Please ensure the Sentinel-2 archive is in the correct location.")
        return

    # ══════════════════════════════════════════════════════════════
    # STEP 1: DATA CLEANING
    # ══════════════════════════════════════════════════════════════
    print_header(1, "DATA CLEANING")

    from utils.cleaning import run_cleaning

    report_df = run_cleaning(
        raw_dir=RAW_DATASET_DIR,
        clean_dir=CLEAN_DATASET_DIR,
        cleaning_config=CLEANING,
        detailed_report_path=FILES["detailed_cleaning_report"]
    )

    # Save cleaning report
    report_df.to_csv(FILES["cleaning_report"], index=False)
    print(f"\n  Cleaning report saved: {FILES['cleaning_report']}")
    print(f"  Total tiles processed: {len(report_df)}")
    print(f"  Total images selected: {report_df['selected_images'].sum()}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: FEATURE EXTRACTION
    # ══════════════════════════════════════════════════════════════
    print_header(2, "FEATURE EXTRACTION")

    from utils.feature_extraction import run_feature_extraction

    df_raw = run_feature_extraction(
        clean_dir=CLEAN_DATASET_DIR,
        clip_lower=FEATURE_EXTRACTION["clip_lower_percentile"],
        clip_upper=FEATURE_EXTRACTION["clip_upper_percentile"],
    )

    if df_raw.empty:
        print("ERROR: Feature extraction returned empty data. Aborting.")
        return

    # Save raw features
    df_raw.to_csv(FILES["features_raw"], index=False)
    print(f"\n  Raw features saved: {FILES['features_raw']}")
    print(f"  Total rows: {len(df_raw)}")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: TIME-SERIES FEATURES
    # ══════════════════════════════════════════════════════════════
    print_header(3, "TIME-SERIES FEATURES")

    from utils.timeseries import run_timeseries

    df_ts = run_timeseries(
        df_raw=df_raw,
        window=TIMESERIES["rolling_window"],
    )

    # Save time-series features
    df_ts.to_csv(FILES["features_timeseries"], index=False)
    print(f"\n  Time-series features saved: {FILES['features_timeseries']}")
    print(f"  Columns: {list(df_ts.columns)}")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: DATA PREPARATION (Scaling + PCA)
    # ══════════════════════════════════════════════════════════════
    print_header(4, "DATA PREPARATION (Scaling + PCA)")

    from utils.preparation import run_preparation

    df_ts, X_scaled, X_pca = run_preparation(
        df_ts=df_ts,
        pca_variance=PREPARATION["pca_variance_threshold"],
        scaler_path=FILES["scaler_model"],
        pca_path=FILES["pca_model"],
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 5: ISOLATION FOREST
    # ══════════════════════════════════════════════════════════════
    print_header(5, "ISOLATION FOREST")

    from utils.models import run_isolation_forest

    iso_scores, iso_predictions = run_isolation_forest(
        X_scaled=X_scaled,
        iso_config=ISOLATION_FOREST,
        model_path=FILES["isolation_forest_model"],
    )

    # ════════════════════════════════════════════════════════════
    # STEP 5b: LOCAL OUTLIER FACTOR  (NEW)
    # ════════════════════════════════════════════════════════════
    print_header("5b", "LOCAL OUTLIER FACTOR")

    from utils.models import run_lof

    lof_scores, lof_predictions = run_lof(
        X_scaled=X_scaled,
        lof_config=LOF,
        model_path=FILES["lof_model"],
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 6: AUTOENCODER (PyTorch)
    # ══════════════════════════════════════════════════════════════
    print_header(6, "AUTOENCODER (PyTorch)")

    from utils.models import run_autoencoder

    ae_errors, train_idx, test_idx = run_autoencoder(
        X_scaled=X_scaled,
        ae_config=AUTOENCODER,
        model_path=FILES["autoencoder_model"],
        seed=RANDOM_SEED,
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 7: HYBRID MODEL
    # ══════════════════════════════════════════════════════════════
    print_header(7, "HYBRID MODEL (ISO + LOF + AE)")

    from utils.hybrid import run_hybrid

    results_df = run_hybrid(
        df_ts=df_ts,
        iso_scores=iso_scores,
        ae_errors=ae_errors,
        lof_scores=lof_scores,
        hybrid_config=HYBRID,
    )

    # Save final results
    results_df.to_csv(FILES["final_results"], index=False)
    print(f"\n  Final results saved: {FILES['final_results']}")



    # ══════════════════════════════════════════════════════════════
    # STEP 9: VISUALIZATION (Production & Evaluation)
    # ══════════════════════════════════════════════════════════════
    print_header(9, "VISUALIZATION")

    from utils.visualization import run_visualization

    print("  Generating production plots...")
    run_visualization(
        df_ts=df_ts,
        results_df=results_df,
        plots_dir=PLOTS_DIR,
        threshold_percentile=HYBRID["threshold_percentile"],
        clean_dir=CLEAN_DATASET_DIR,
    )


    # ══════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE!                                       |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Time elapsed: {minutes}m {seconds}s")
    print(f"\n  Outputs:")
    print(f"    - Cleaned dataset:      {CLEAN_DATASET_DIR}")
    print(f"    - Cleaning report:      {FILES['cleaning_report']}")
    print(f"    - Raw features:         {FILES['features_raw']}")
    print(f"    - Time-series features: {FILES['features_timeseries']}")
    print(f"    - Final results:        {FILES['final_results']}")

    print(f"    - Models:               {MODELS_DIR}")
    print(f"    - Plots:                {PLOTS_DIR}")
    print()


if __name__ == "__main__":
    main()
