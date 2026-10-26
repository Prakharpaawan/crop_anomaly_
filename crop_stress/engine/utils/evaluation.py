"""
evaluation.py — Step 8: Evaluation with Synthetic Anomaly Injection (V2)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import torch
import os
from utils.timeseries import compute_timeseries_features
from utils.preparation import FEATURE_COLUMNS
from utils.hybrid import run_hybrid
from utils.models import CropAutoencoder

def inject_anomalies(df_ts, injection_ratio, severity, duration, seed, window):
    np.random.seed(seed)
    
    df_modified = df_ts.copy()
    unique_tiles = df_modified["tile_id"].unique()
    n_samples = len(df_ts)
    
    # We want to inject anomalies across events
    n_inject_total = max(1, int(n_samples * injection_ratio)) 
    n_events = max(1, n_inject_total // duration)
    
    ground_truth = np.zeros(n_samples, dtype=int)
    
    # Find valid start indices (must have enough remaining steps in the same tile)
    valid_starts = []
    for tile in unique_tiles:
        tile_indices = df_modified[df_modified["tile_id"] == tile].index.values
        for i in range(len(tile_indices) - duration + 1):
            valid_starts.append(tile_indices[i])
            
    if len(valid_starts) < n_events:
        n_events = len(valid_starts)
    
    if n_events == 0:
        print("  WARNING: No valid locations to inject anomalies with this duration.")
        return df_modified, ground_truth

    inject_starts = np.random.choice(valid_starts, size=n_events, replace=False)
    
    features_to_degrade = ["mean", "min", "max", "p25", "p50", "p75"]
    
    for start_idx in inject_starts:
        # Degrade consecutive time steps up to 'duration'
        for step in range(duration):
            idx = start_idx + step
            for f in features_to_degrade:
                df_modified.loc[idx, f] = df_modified.loc[idx, f] * (1.0 - severity)
            ground_truth[idx] = 1
            
    # Recompute time-series features per tile
    all_tiles = []
    for tile in unique_tiles:
        df_tile = df_modified[df_modified["tile_id"] == tile].copy()
        df_tile = compute_timeseries_features(df_tile, window)
        all_tiles.append(df_tile)
    df_modified = pd.concat(all_tiles, ignore_index=True)
    
    print(f"  Injected {np.sum(ground_truth)} anomalies (across {n_events} events of duration {duration}, severity {severity*100}%).")
    return df_modified, ground_truth


def inject_gradual_anomalies(df_ts, injection_ratio, severity, duration, seed, window):
    """
    Gradual-onset anomaly injection: stress ramps up over the duration.
    Step 1 gets severity/3, step 2 gets 2*severity/3, step 3 gets full severity.
    More realistic than sudden drops — simulates progressive crop damage.
    """
    np.random.seed(seed)
    
    df_modified = df_ts.copy()
    unique_tiles = df_modified["tile_id"].unique()
    n_samples = len(df_ts)
    
    n_inject_total = max(1, int(n_samples * injection_ratio))
    n_events = max(1, n_inject_total // duration)
    ground_truth = np.zeros(n_samples, dtype=int)
    
    valid_starts = []
    for tile in unique_tiles:
        tile_indices = df_modified[df_modified["tile_id"] == tile].index.values
        for i in range(len(tile_indices) - duration + 1):
            valid_starts.append(tile_indices[i])
    
    if len(valid_starts) < n_events:
        n_events = len(valid_starts)
    if n_events == 0:
        return df_modified, ground_truth

    inject_starts = np.random.choice(valid_starts, size=n_events, replace=False)
    features_to_degrade = ["mean", "min", "max", "p25", "p50", "p75"]
    
    for start_idx in inject_starts:
        for step in range(duration):
            idx = start_idx + step
            # Ramp: severity increases linearly across the duration
            step_severity = severity * (step + 1) / duration
            for f in features_to_degrade:
                df_modified.loc[idx, f] = df_modified.loc[idx, f] * (1.0 - step_severity)
            ground_truth[idx] = 1
    
    # Recompute time-series features
    all_tiles = []
    for tile in unique_tiles:
        df_tile = df_modified[df_modified["tile_id"] == tile].copy()
        df_tile = compute_timeseries_features(df_tile, window)
        all_tiles.append(df_tile)
    df_modified = pd.concat(all_tiles, ignore_index=True)
    
    print(f"  Injected {np.sum(ground_truth)} GRADUAL anomalies ({n_events} events, max severity {severity*100}%).")
    return df_modified, ground_truth


def inject_spectral_shift_anomalies(df_ts, injection_ratio, severity, duration, seed, window):
    """
    Spectral-shift anomaly injection: increases std while decreasing mean.
    Simulates patchy, heterogeneous damage (e.g., localized disease patches)
    rather than uniform canopy degradation.
    """
    np.random.seed(seed)
    
    df_modified = df_ts.copy()
    unique_tiles = df_modified["tile_id"].unique()
    n_samples = len(df_ts)
    
    n_inject_total = max(1, int(n_samples * injection_ratio))
    n_events = max(1, n_inject_total // duration)
    ground_truth = np.zeros(n_samples, dtype=int)
    
    valid_starts = []
    for tile in unique_tiles:
        tile_indices = df_modified[df_modified["tile_id"] == tile].index.values
        for i in range(len(tile_indices) - duration + 1):
            valid_starts.append(tile_indices[i])
    
    if len(valid_starts) < n_events:
        n_events = len(valid_starts)
    if n_events == 0:
        return df_modified, ground_truth

    inject_starts = np.random.choice(valid_starts, size=n_events, replace=False)
    
    for start_idx in inject_starts:
        for step in range(duration):
            idx = start_idx + step
            # Decrease mean and median (canopy loss)
            for f in ["mean", "p50", "p25"]:
                df_modified.loc[idx, f] = df_modified.loc[idx, f] * (1.0 - severity)
            # Increase std and widen IQR (patchy damage = more heterogeneity)
            df_modified.loc[idx, "std"] = df_modified.loc[idx, "std"] * (1.0 + severity)
            # Push p75 up slightly and min down (some areas still healthy, some dead)
            df_modified.loc[idx, "min"] = df_modified.loc[idx, "min"] * (1.0 - severity * 1.5)
            df_modified.loc[idx, "p75"] = df_modified.loc[idx, "p75"] * (1.0 + severity * 0.2)
            ground_truth[idx] = 1
    
    # Recompute time-series features
    all_tiles = []
    for tile in unique_tiles:
        df_tile = df_modified[df_modified["tile_id"] == tile].copy()
        df_tile = compute_timeseries_features(df_tile, window)
        all_tiles.append(df_tile)
    df_modified = pd.concat(all_tiles, ignore_index=True)
    
    print(f"  Injected {np.sum(ground_truth)} SPECTRAL-SHIFT anomalies ({n_events} events, severity {severity*100}%).")
    return df_modified, ground_truth

def evaluate_predictions(ground_truth, predicted_labels):
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT -- Hybrid Crop Stress Detection (V2)")
    lines.append("=" * 60)
    lines.append("")

    n_total = len(ground_truth)
    n_true_anomalies = int(np.sum(ground_truth == 1))
    n_predicted_anomalies = int(np.sum(predicted_labels == 1))

    lines.append(f"Total samples:            {n_total}")
    lines.append(f"True anomalies (injected): {n_true_anomalies}")
    lines.append(f"Predicted anomalies:       {n_predicted_anomalies}")
    lines.append("")

    if n_true_anomalies == 0:
        lines.append("WARNING: No ground truth anomalies.")
        return "\n".join(lines)

    precision = precision_score(ground_truth, predicted_labels, zero_division=0)
    recall = recall_score(ground_truth, predicted_labels, zero_division=0)
    f1 = f1_score(ground_truth, predicted_labels, zero_division=0)

    lines.append(f"Precision:  {precision:.4f}")
    lines.append(f"Recall:     {recall:.4f}")
    lines.append(f"F1-Score:   {f1:.4f}")
    lines.append("")

    lines.append("Confusion Matrix:")
    lines.append("                  Predicted Normal  Predicted Anomaly")
    cm = confusion_matrix(ground_truth, predicted_labels)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        lines.append(f"  Actual Normal:     {tn:>10}        {fp:>10}")
        lines.append(f"  Actual Anomaly:    {fn:>10}        {tp:>10}")
    else:
        lines.append(f"  {cm}")

    lines.append("")
    lines.append("Detailed Classification Report:")
    lines.append("-" * 50)
    report = classification_report(ground_truth, predicted_labels, target_names=["Normal", "Anomaly"], zero_division=0)
    lines.append(report)

    return "\n".join(lines)

def run_evaluation(df_ts, eval_config, seed, iso_config, lof_config, ae_config, hybrid_config, scaler_path, models_dir, window):
    # 1. Inject anomalies (using average degradation if called from original pipeline)
    severity_avg = (eval_config["degradation_min"] + eval_config["degradation_max"]) / 2.0
    df_modified, ground_truth = inject_anomalies(
        df_ts,
        eval_config["injection_ratio"],
        severity=severity_avg,
        duration=2,
        seed=seed,
        window=window
    )
    
    # 2. Scale features
    scaler = joblib.load(scaler_path)
    X = df_modified[FEATURE_COLUMNS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X_scaled = scaler.transform(X)
    
    # 3. Score with models
    # Isolation Forest
    iso_model_path = os.path.join(models_dir, "isolation_forest_model.pkl")
    iso_model = joblib.load(iso_model_path)
    iso_scores = iso_model.decision_function(X_scaled)
    
    # LOF — load saved novelty model and use score_samples (no data leakage)
    lof_model_path = os.path.join(models_dir, "lof_model.pkl")
    lof_model = joblib.load(lof_model_path)
    lof_scores = -lof_model.score_samples(X_scaled)
    
    # Autoencoder
    ae_model_path = os.path.join(models_dir, "autoencoder_model.pt")
    input_dim = X_scaled.shape[1]
    ae_model = CropAutoencoder(
        input_dim=input_dim, hidden_1=ae_config["hidden_dim_1"],
        latent=ae_config["latent_dim"], hidden_2=ae_config["hidden_dim_2"],
        dropout=ae_config.get("dropout", 0.0)
    )
    ae_model.load_state_dict(torch.load(ae_model_path, weights_only=True))
    ae_model.eval()
    
    with torch.no_grad():
        tensor = torch.FloatTensor(X_scaled)
        reconstructed = ae_model(tensor)
        ae_errors = torch.mean((tensor - reconstructed) ** 2, dim=1).numpy()
        
    # 4. Run hybrid logic
    results_df = run_hybrid(df_modified, iso_scores, ae_errors, lof_scores, hybrid_config)
    predicted_labels = results_df["label"].values
    
    # 5. Evaluate
    report_text = evaluate_predictions(ground_truth, predicted_labels)
    
    # Add ground truth and modified df_ts to results for visualization
    results_df["ground_truth"] = ground_truth
    
    return report_text, results_df, df_modified
