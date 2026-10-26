"""
run_experiments.py — Comprehensive Evaluation Experiments (V3)

Major improvements:
  - Multi-seed evaluation with mean ± std reporting
  - ROC-AUC and PR-AUC (Average Precision) metrics
  - Confusion matrices in all experiments
  - Threshold sensitivity analysis
  - Sliding-window persistence (2-of-3) alternative
  - Precision-Recall and ROC curve generation
  - LOF uses novelty=True mode (fixes data leakage)
  - Gradual-onset and spectral-shift anomaly injection types
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

from config import TIMESERIES, LOF, AUTOENCODER, HYBRID, FILES, MODELS_DIR, RANDOM_SEED, OUTPUT_DIR
from utils.preparation import FEATURE_COLUMNS
from utils.evaluation import inject_anomalies, inject_gradual_anomalies, inject_spectral_shift_anomalies
from utils.models import CropAutoencoder
from utils.hybrid import min_max_normalize, apply_threshold_and_persistence, apply_global_threshold_and_persistence, apply_sliding_window_persistence

SEEDS = [42, 123, 456, 789, 1024]
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def load_data_and_models():
    print("Loading data and models...")
    df_ts = pd.read_csv(FILES["features_timeseries"])
    scaler = joblib.load(FILES["scaler_model"])
    iso_model = joblib.load(FILES["isolation_forest_model"])
    
    # LOF in novelty mode — fitted on clean data during pipeline run
    lof_model = joblib.load(FILES["lof_model"])
    
    # AE model
    input_dim = len(FEATURE_COLUMNS)
    ae_model = CropAutoencoder(
        input_dim=input_dim, 
        hidden_1=AUTOENCODER["hidden_dim_1"], 
        latent=AUTOENCODER["latent_dim"], 
        hidden_2=AUTOENCODER["hidden_dim_2"],
        dropout=AUTOENCODER.get("dropout", 0.0)
    )
    ae_model.load_state_dict(torch.load(FILES["autoencoder_model"], weights_only=True))
    ae_model.eval()
    
    return df_ts, scaler, iso_model, lof_model, ae_model


def get_scores(df_modified, scaler, iso_model, lof_model, ae_model):
    X = df_modified[FEATURE_COLUMNS].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X_scaled = scaler.transform(X)
    
    # ISO (invert so higher is anomalous)
    iso_scores = -iso_model.decision_function(X_scaled)
    
    # LOF — use novelty detection (score_samples on new data)
    # score_samples returns negative scores; more negative = more anomalous
    lof_scores = -lof_model.score_samples(X_scaled)
    
    # AE
    with torch.no_grad():
        tensor = torch.FloatTensor(X_scaled)
        reconstructed = ae_model(tensor)
        ae_errors = torch.mean((tensor - reconstructed) ** 2, dim=1).numpy()
        
    return iso_scores, lof_scores, ae_errors


def calculate_hybrid_score(iso_scores, lof_scores, ae_errors, config):
    iso_norm = min_max_normalize(iso_scores)
    lof_norm = min_max_normalize(lof_scores)
    ae_norm = min_max_normalize(ae_errors)
    
    final_score = (config["ae_weight"] * ae_norm + 
                   config["iso_weight"] * iso_norm + 
                   config["lof_weight"] * lof_norm)
    return final_score


def compute_full_metrics(gt, pred, scores=None):
    """Compute all metrics including ROC-AUC and PR-AUC."""
    p = precision_score(gt, pred, zero_division=0)
    r = recall_score(gt, pred, zero_division=0)
    f = f1_score(gt, pred, zero_division=0)
    fa = int(np.sum((gt == 0) & (pred == 1)))
    
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    roc_auc = 0.0
    pr_auc = 0.0
    if scores is not None and len(np.unique(gt)) > 1:
        try:
            roc_auc = roc_auc_score(gt, scores)
            pr_auc = average_precision_score(gt, scores)
        except ValueError:
            pass
    
    return {
        "precision": p, "recall": r, "f1": f, "false_alarms": fa,
        "specificity": specificity, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def format_metrics_row(name, metrics):
    return (f"| {name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['specificity']:.4f} | {metrics['false_alarms']} |")


def multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, config,
                        severity, duration, injection_ratio=0.15,
                        threshold_fn=apply_threshold_and_persistence,
                        inject_fn=inject_anomalies):
    """Run evaluation across multiple seeds, return averaged metrics."""
    all_metrics = []
    
    for seed in SEEDS:
        df_mod, gt = inject_fn(
            df_ts, injection_ratio=injection_ratio, severity=severity,
            duration=duration, seed=seed, window=TIMESERIES["rolling_window"]
        )
        if np.sum(gt) == 0:
            continue
            
        iso_s, lof_s, ae_s = get_scores(df_mod, scaler, iso_model, lof_model, ae_model)
        final_s = calculate_hybrid_score(iso_s, lof_s, ae_s, config)
        
        pred, _ = threshold_fn(final_s, df_mod["tile_id"].values, config)
        metrics = compute_full_metrics(gt, pred, final_s)
        all_metrics.append(metrics)
    
    if not all_metrics:
        return {k: 0.0 for k in ["precision", "recall", "f1", "roc_auc", "pr_auc",
                                   "specificity", "false_alarms", "tp", "fp", "fn", "tn",
                                   "precision_std", "recall_std", "f1_std"]}
    
    avg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg[key] = np.mean(values)
    
    # Also compute std for key metrics
    for key in ["precision", "recall", "f1"]:
        values = [m[key] for m in all_metrics]
        avg[f"{key}_std"] = np.std(values)
    
    return avg


def format_avg_metrics_row(name, avg):
    return (f"| {name} | {avg['precision']:.4f}±{avg.get('precision_std',0):.3f} | "
            f"{avg['recall']:.4f}±{avg.get('recall_std',0):.3f} | "
            f"{avg['f1']:.4f}±{avg.get('f1_std',0):.3f} | "
            f"{avg['roc_auc']:.4f} | {avg['pr_auc']:.4f} | "
            f"{avg['specificity']:.4f} | {int(avg['false_alarms'])} |")


def plot_pr_roc_curves(df_ts, scaler, iso_model, lof_model, ae_model, config, output_dir):
    """Generate Precision-Recall and ROC curves for the hybrid ensemble."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a single representative seed for curves
    df_mod, gt = inject_anomalies(
        df_ts, injection_ratio=0.15, severity=0.40, duration=2,
        seed=42, window=TIMESERIES["rolling_window"]
    )
    
    iso_s, lof_s, ae_s = get_scores(df_mod, scaler, iso_model, lof_model, ae_model)
    final_s = calculate_hybrid_score(iso_s, lof_s, ae_s, config)
    
    if len(np.unique(gt)) < 2:
        print("  WARNING: Cannot plot curves — no positive samples.")
        return
    
    # PR Curve
    precision_vals, recall_vals, _ = precision_recall_curve(gt, final_s)
    pr_auc = average_precision_score(gt, final_s)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, 'b-', linewidth=2, label=f'Hybrid Ensemble (AP={pr_auc:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(gt, final_s)
    roc_auc = roc_auc_score(gt, final_s)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Hybrid Ensemble (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Per-model curves
    models = {
        "Isolation Forest": min_max_normalize(iso_s),
        "LOF": min_max_normalize(lof_s),
        "Autoencoder": min_max_normalize(ae_s),
        "Hybrid Ensemble": final_s
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50']
    
    for (name, scores), color in zip(models.items(), colors):
        if len(np.unique(gt)) < 2:
            continue
        p_vals, r_vals, _ = precision_recall_curve(gt, scores)
        ap = average_precision_score(gt, scores)
        ax1.plot(r_vals, p_vals, color=color, linewidth=2, label=f'{name} (AP={ap:.3f})')
        
        fp_vals, tp_vals, _ = roc_curve(gt, scores)
        auc = roc_auc_score(gt, scores)
        ax2.plot(fp_vals, tp_vals, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    
    ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
    ax1.set_title('PR Curves — Model Comparison'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
    ax2.set_title('ROC Curves — Model Comparison'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_comparison_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Curves saved to {output_dir}")


def run_all_experiments():
    df_ts, scaler, iso_model, lof_model, ae_model = load_data_and_models()
    report = []
    report.append("# Crop Stress Detection — Comprehensive Evaluation Report")
    report.append(f"**Seeds**: {SEEDS} | **Injection Ratio**: 15% | **Metrics averaged over {len(SEEDS)} runs**\n")
    report.append("All metrics reported as mean±std across random seeds for statistical validity.\n")
    
    config = HYBRID.copy()
    config["use_persistence"] = True
    config["min_consecutive_anomalies"] = 2

    # =========================================================================
    # EXPERIMENT 1: SEVERITY & DURATION (MULTI-SEED)
    # =========================================================================
    print("Running Exp 1: Severities (multi-seed)...")
    report.append("## Experiment 1: Anomaly Severity & Duration")
    report.append("Hybrid Ensemble tested against varying vegetation reduction and persistence durations.\n")
    report.append("| Severity | Duration | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|----------|----------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    severities = [0.10, 0.20, 0.30, 0.40, 0.50]
    durations = [1, 2, 3]
    
    for sev in severities:
        for dur in durations:
            avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, config,
                                      severity=sev, duration=dur)
            report.append(format_avg_metrics_row(f"{int(sev*100)}% / {dur}d", avg))
    
    report.append("\n*Recall for 1-day duration expected to be lower due to 2-day persistence rule.*\n")

    # =========================================================================
    # EXPERIMENT 2: INDIVIDUAL VS ENSEMBLE
    # =========================================================================
    print("Running Exp 2: Model Comparison (multi-seed)...")
    report.append("## Experiment 2: Individual Models vs. Hybrid Ensemble")
    report.append("Baseline: 40% severity, 2-day duration. All models use same thresholding.\n")
    report.append("| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|-------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    model_configs = [
        ("Isolation Forest", {"iso_weight": 1.0, "lof_weight": 0.0, "ae_weight": 0.0}),
        ("LOF", {"iso_weight": 0.0, "lof_weight": 1.0, "ae_weight": 0.0}),
        ("Autoencoder", {"iso_weight": 0.0, "lof_weight": 0.0, "ae_weight": 1.0}),
        ("ISO + LOF", {"iso_weight": 0.5, "lof_weight": 0.5, "ae_weight": 0.0}),
        ("Hybrid (all 3)", {"iso_weight": config["iso_weight"], "lof_weight": config["lof_weight"], "ae_weight": config["ae_weight"]}),
    ]
    
    for name, weight_cfg in model_configs:
        cfg = config.copy()
        cfg.update(weight_cfg)
        avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, cfg,
                                  severity=0.40, duration=2)
        report.append(format_avg_metrics_row(name, avg))
    
    report.append("\n")

    # =========================================================================
    # EXPERIMENT 3: THRESHOLD COMPARISON
    # =========================================================================
    print("Running Exp 3: Thresholding...")
    report.append("## Experiment 3: Adaptive vs. Fixed Global Threshold")
    report.append("40% severity, 2-day duration.\n")
    report.append("| Threshold Type | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|----------------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    for name, fn in [("Adaptive (per-tile)", apply_threshold_and_persistence),
                     ("Fixed Global", apply_global_threshold_and_persistence)]:
        avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, config,
                                  severity=0.40, duration=2, threshold_fn=fn)
        report.append(format_avg_metrics_row(name, avg))
    
    report.append("\n")

    # =========================================================================
    # EXPERIMENT 4: PERSISTENCE STRATEGIES
    # =========================================================================
    print("Running Exp 4: Persistence strategies...")
    report.append("## Experiment 4: Temporal Persistence Strategies")
    report.append("40% severity, 3-day duration. Comparing strict consecutive vs. sliding window.\n")
    report.append("| Strategy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|----------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    # No persistence
    cfg_no = config.copy()
    cfg_no["use_persistence"] = False
    cfg_no["min_consecutive_anomalies"] = 1
    avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, cfg_no,
                              severity=0.40, duration=3)
    report.append(format_avg_metrics_row("No Persistence", avg))
    
    # Strict 2 consecutive
    cfg_s2 = config.copy()
    cfg_s2["use_persistence"] = True
    cfg_s2["min_consecutive_anomalies"] = 2
    avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, cfg_s2,
                              severity=0.40, duration=3)
    report.append(format_avg_metrics_row("Strict 2-Consecutive", avg))
    
    # Sliding window 2-of-3
    avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, config,
                              severity=0.40, duration=3,
                              threshold_fn=apply_sliding_window_persistence)
    report.append(format_avg_metrics_row("Sliding 2-of-3", avg))
    
    # Strict 3 consecutive
    cfg_s3 = config.copy()
    cfg_s3["use_persistence"] = True
    cfg_s3["min_consecutive_anomalies"] = 3
    avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, cfg_s3,
                              severity=0.40, duration=3)
    report.append(format_avg_metrics_row("Strict 3-Consecutive", avg))
    
    report.append("\n")

    # =========================================================================
    # EXPERIMENT 5: THRESHOLD SENSITIVITY ANALYSIS
    # =========================================================================
    print("Running Exp 5: Threshold sensitivity...")
    report.append("## Experiment 5: Threshold Sensitivity Analysis")
    report.append("40% severity, 2-day duration. No persistence (to isolate threshold effect).\n")
    report.append("| Percentile | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|------------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    for perc in [75, 80, 85, 90, 95]:
        cfg_perc = config.copy()
        cfg_perc["threshold_percentile"] = perc
        cfg_perc["use_persistence"] = False
        cfg_perc["min_consecutive_anomalies"] = 1
        avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, cfg_perc,
                                  severity=0.40, duration=2)
        report.append(format_avg_metrics_row(f"P{perc}", avg))
    
    report.append("\n")

    # =========================================================================
    # EXPERIMENT 6: ANOMALY INJECTION TYPES
    # =========================================================================
    print("Running Exp 6: Injection types...")
    report.append("## Experiment 6: Anomaly Injection Types")
    report.append("Testing pipeline against different anomaly signatures (40% severity, 2-day duration).\n")
    report.append("| Injection Type | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Specificity | False Alarms |")
    report.append("|----------------|-----------|--------|----------|---------|--------|-------------|--------------|")
    
    injection_types = [
        ("Sudden Drop (baseline)", inject_anomalies),
        ("Gradual Onset", inject_gradual_anomalies),
        ("Spectral Shift", inject_spectral_shift_anomalies),
    ]
    
    for name, inject_fn in injection_types:
        avg = multi_seed_evaluate(df_ts, scaler, iso_model, lof_model, ae_model, config,
                                  severity=0.40, duration=2, inject_fn=inject_fn)
        report.append(format_avg_metrics_row(name, avg))
    
    report.append("\n")

    # =========================================================================
    # GENERATE CURVES
    # =========================================================================
    print("Generating PR and ROC curves...")
    curves_dir = os.path.join(PLOTS_DIR, "evaluation_curves")
    plot_pr_roc_curves(df_ts, scaler, iso_model, lof_model, ae_model, config, curves_dir)

    # =========================================================================
    # SAVE REPORT
    # =========================================================================
    report_path = os.path.join("outputs", "experiments_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\nExperiments complete! Report saved to {report_path}")


if __name__ == "__main__":
    run_all_experiments()
