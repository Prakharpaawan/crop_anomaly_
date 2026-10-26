"""
optimize_weights.py — Grid Search for Optimal Hybrid Ensemble Weights

Searches over model weights (ISO, LOF, AE) and threshold percentiles
to find the configuration that maximizes a precision-weighted objective.

Fixed: indentation bug that caused score calculation to run outside inner loop.
Improved: finer grid (0.1 step), multi-seed averaging, proper structure.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from run_experiments import load_data_and_models, get_scores, calculate_hybrid_score, apply_threshold_and_persistence
from utils.evaluation import inject_anomalies
from config import RANDOM_SEED, HYBRID, TIMESERIES

SEEDS = [42, 123, 456, 789, 1024]

def run_optimization():
    df_ts, scaler, iso_model, lof_model, ae_model = load_data_and_models()

    # Pre-compute scores for each seed (injection changes the data, so scores differ)
    seed_data = []
    for seed in SEEDS:
        df_mod, gt = inject_anomalies(
            df_ts, injection_ratio=0.15, severity=0.40, duration=2,
            seed=seed, window=TIMESERIES["rolling_window"]
        )
        iso_s, lof_s, ae_s = get_scores(df_mod, scaler, iso_model, lof_model, ae_model)
        tile_ids = df_mod["tile_id"].values
        seed_data.append((iso_s, lof_s, ae_s, tile_ids, gt))
        
    best_score = -1
    best_weights = None
    results_log = []

    # Grid search weights (finer 0.1 step)
    for perc in [80, 85, 90, 95]:
        for iso_w in np.arange(0.0, 1.05, 0.1):
            for lof_w in np.arange(0.0, 1.05 - iso_w, 0.1):
                ae_w = round(1.0 - iso_w - lof_w, 2)
                if ae_w < -0.01:  # Skip invalid combinations
                    continue

                cfg = HYBRID.copy()
                cfg["iso_weight"] = round(iso_w, 2)
                cfg["lof_weight"] = round(lof_w, 2)
                cfg["ae_weight"] = max(0, ae_w)
                cfg["use_persistence"] = True
                cfg["min_consecutive_anomalies"] = 2
                cfg["threshold_percentile"] = perc

                # Average metrics across all seeds
                f1s, precs, recs = [], [], []
                for iso_s, lof_s, ae_s, tile_ids, gt in seed_data:
                    final_s = calculate_hybrid_score(iso_s, lof_s, ae_s, cfg)
                    pred, _ = apply_threshold_and_persistence(final_s, tile_ids, cfg)
                    
                    f1s.append(f1_score(gt, pred, zero_division=0))
                    precs.append(precision_score(gt, pred, zero_division=0))
                    recs.append(recall_score(gt, pred, zero_division=0))

                avg_f1 = np.mean(f1s)
                avg_p = np.mean(precs)
                avg_r = np.mean(recs)
                
                # Objective: F1 + 0.5 * Precision (favor precision)
                score = avg_f1 + (avg_p * 0.5)
                
                results_log.append({
                    "iso_w": round(iso_w, 2), "lof_w": round(lof_w, 2),
                    "ae_w": max(0, ae_w), "perc": perc,
                    "avg_p": avg_p, "avg_r": avg_r, "avg_f1": avg_f1,
                    "score": score
                })

                if score > best_score:
                    best_score = score
                    best_weights = (round(iso_w, 2), round(lof_w, 2), max(0, ae_w),
                                    perc, avg_p, avg_r, avg_f1)

    # Save full results log
    df_log = pd.DataFrame(results_log)
    df_log = df_log.sort_values("score", ascending=False)
    df_log.to_csv("outputs/weight_optimization_log.csv", index=False)

    print(f"\nBest Config: ISO={best_weights[0]:.2f}, LOF={best_weights[1]:.2f}, AE={best_weights[2]:.2f}, Threshold P{best_weights[3]}")
    print(f"Averaged over {len(SEEDS)} seeds:")
    print(f"  Precision: {best_weights[4]:.4f}")
    print(f"  Recall:    {best_weights[5]:.4f}")
    print(f"  F1-Score:  {best_weights[6]:.4f}")
    
    # Show top 10 configs
    print(f"\nTop 10 Configurations:")
    print(df_log.head(10).to_string(index=False))


if __name__ == "__main__":
    run_optimization()
