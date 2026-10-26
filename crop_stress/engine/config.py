"""
config.py — Central configuration for the Hybrid Crop Stress Detection Pipeline.
All paths, thresholds, and model parameters are defined here.
No hardcoded values in other files.
"""

import os

# ─── Random Seed (for reproducibility) ────────────────────────────
RANDOM_SEED = 42

# ─── Base Paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_data_root(start):
    """
    Walk up until the folder holding `dataset_clean_2` is found.

    This module used to assume the data sat exactly one level above it. It now
    lives inside `crop_stress/engine/`, two levels down, so the depth is
    discovered rather than assumed — that way the engine keeps working wherever
    it is nested.
    """
    p = start
    while os.path.dirname(p) != p:
        if os.path.isdir(os.path.join(p, "dataset_clean_2")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(start)      # fall back to the old behaviour


PARENT_DIR = _find_data_root(PROJECT_ROOT)

# Input: raw satellite archive
RAW_DATASET_DIR = os.path.join(
    PARENT_DIR, "archive (1)", "lombardia", "data2019"
)

# Output: cleaned dataset
CLEAN_DATASET_DIR = os.path.join(PARENT_DIR, "dataset_clean_2")

# Output: all pipeline outputs (CSVs, reports, plots)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# ─── Step 1: Data Cleaning Thresholds ────────────────────────────
CLEANING = {
    "min_std": 200,             # Minimum standard deviation to keep
    "min_valid_ratio": 0.7,     # Minimum fraction of valid (non-nodata) pixels
    "min_unique_ratio": 0.05,   # Minimum unique pixel ratio
    "min_gradient": 100,        # Minimum spatial gradient strength
    "target_count": 25,         # Number of images to select per tile (was 15)
    "min_days_apart": 10,       # Minimum days between selected images
    "max_tiles": None,          # Use ALL tiles (was 20 — too few for meaningful stats)
}

# ─── Step 2: Feature Extraction ──────────────────────────────────
FEATURE_EXTRACTION = {
    "clip_lower_percentile": 1,   # Clip values below this percentile
    "clip_upper_percentile": 99,  # Clip values above this percentile
}

# ─── Step 3: Time-Series Features ────────────────────────────────
TIMESERIES = {
    "rolling_window": 3,  # Window size for rolling statistics
}

# ─── Step 4: Data Preparation (Scaling + PCA) ────────────────────
# NOTE: PCA is fitted and persisted to models/pca_model.pkl, but NOTHING
# CONSUMES IT. All three detectors are trained on `X_scaled` (see
# utils/models.py) and the scoring path in run_experiments.get_scores()
# only calls scaler.transform(). This setting therefore has no effect on
# any reported result. Kept for reference; do not cite PCA as part of the
# method.
PREPARATION = {
    "pca_variance_threshold": 0.95,  # UNUSED — see note above
}

# NOTE ON `contamination` (both models below): it has no effect on any
# reported result. For LOF in novelty mode and for IsolationForest it only
# shifts `offset_`, which is used by predict()/decision_function(). This
# pipeline scores with -score_samples() and then applies its own percentile
# threshold, and a constant offset cannot change a percentile ranking. The
# method is therefore insensitive to the assumed contamination rate — which
# is a genuine advantage, since that rate is unknown in practice.

# ─── Step 5: Isolation Forest ────────────────────────────────────
ISOLATION_FOREST = {
    "contamination": 0.1,  # inert — see note above
    "random_state": RANDOM_SEED,
    "n_estimators": 100,
}

# ─── Step 5b: Local Outlier Factor (NEW) ─────────────────────────
LOF = {
    "n_neighbors":   10,   # neighbourhood size; never swept — see limitations
    "contamination": 0.1,  # inert — see note above
}

# ─── Step 6: Autoencoder ─────────────────────────────────────────
AUTOENCODER = {
    "hidden_dim_1": 32,   # First hidden layer size (was 16)
    "latent_dim": 4,      # Bottleneck layer size (was 8 — too wide, barely compressed)
    "hidden_dim_2": 32,   # Third hidden layer size (was 16)
    "dropout": 0.2,       # Dropout for regularization (NEW)
    "epochs": 200,        # Training epochs (was 50 — underfit on small data)
    "batch_size": 32,
    "learning_rate": 0.001,
    "train_ratio": 0.8,   # 80% train, 20% test
}

# ─── Step 7: Hybrid Model ────────────────────────────────────────
HYBRID = {
    "iso_weight":           0.4,   # Isolation Forest contribution (pending re-optimization)
    "lof_weight":           0.4,   # LOF contribution (pending re-optimization)
    "ae_weight":            0.2,   # Autoencoder contribution (was 0.0 from bugged grid search)
    "threshold_percentile": 85,    # Threshold percentile (pending re-optimization)
    "use_persistence":      True,  # Require consecutive flags to confirm anomaly
}

# ─── Step 8: Evaluation ──────────────────────────────────────────
EVALUATION = {
    "injection_ratio": 0.10,        # Inject anomalies into 10% of samples
    "degradation_min": 0.30,        # Reduce feature values by 30-50%
    "degradation_max": 0.50,
}

# ─── Output File Names ──────────────────────────────────────────
FILES = {
    "cleaning_report":           os.path.join(OUTPUT_DIR, "cleaning_report.csv"),
    "detailed_cleaning_report":  os.path.join(OUTPUT_DIR, "detailed_cleaning_report.csv"),
    "features_raw":              os.path.join(OUTPUT_DIR, "features_raw.csv"),
    "features_timeseries":       os.path.join(OUTPUT_DIR, "features_timeseries.csv"),
    "final_results":             os.path.join(OUTPUT_DIR, "final_results.csv"),
    "pca_model":                 os.path.join(MODELS_DIR, "pca_model.pkl"),
    "scaler_model":              os.path.join(MODELS_DIR, "scaler_model.pkl"),
    "isolation_forest_model":    os.path.join(MODELS_DIR, "isolation_forest_model.pkl"),
    "lof_model":                 os.path.join(MODELS_DIR, "lof_model.pkl"),   # NEW
    "autoencoder_model":         os.path.join(MODELS_DIR, "autoencoder_model.pt"),
}
