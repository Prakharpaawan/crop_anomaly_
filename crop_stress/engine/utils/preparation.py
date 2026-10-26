"""
preparation.py — Step 4: Data Preparation (Scaling + PCA)
Combines all tile data, applies StandardScaler normalization,
and performs PCA dimensionality reduction.

Saves: scaler_model.pkl, pca_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Define which columns are numeric features for scaling/PCA
# Keep existing features first, then new ones — order matters for PCA interpretability
FEATURE_COLUMNS = [
    # Existing features
    "mean", "std", "min", "max", "valid_pixel_ratio",
    "rolling_median", "rolling_std", "diff", "percent_change",
    # New features (Step 2 improvement)
    "p25", "p50", "p75", "cv", "iqr", "diff_from_global_median", "smoothed_p50"
]


def run_preparation(df_ts, pca_variance, scaler_path, pca_path):
    """
    Normalizes features with StandardScaler and reduces dimensions with PCA.
    
    Args:
        df_ts: DataFrame from features_timeseries.csv
        pca_variance: float, fraction of variance to retain (e.g. 0.95)
        scaler_path: file path to save the fitted StandardScaler
        pca_path: file path to save the fitted PCA model
    
    Returns:
        df_ts: original DataFrame (unchanged, for metadata)
        X_scaled: numpy array of scaled features
        X_pca: numpy array of PCA-transformed features
    """
    # Extract only the numeric feature columns
    X = df_ts[FEATURE_COLUMNS].values.copy()

    # Handle any remaining NaN or Inf values
    # Replace NaN with 0 and Inf with large finite numbers
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # --- Step 1: StandardScaler normalization ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Scaled {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features.")

    # --- Step 2: PCA dimensionality reduction ---
    pca = PCA(n_components=pca_variance, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    explained = sum(pca.explained_variance_ratio_) * 100

    print(f"  PCA: {X_scaled.shape[1]} features -> {n_components} components "
          f"({explained:.1f}% variance explained).")

    # --- Save the fitted models ---
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler: {scaler_path}")

    joblib.dump(pca, pca_path)
    print(f"  Saved PCA model: {pca_path}")

    return df_ts, X_scaled, X_pca
