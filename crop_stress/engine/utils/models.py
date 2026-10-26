"""
models.py — Step 5 + 5b + 6: Isolation Forest + LOF + Autoencoder
Contains all anomaly detection models.

Step 5 : Isolation Forest  (scikit-learn)
Step 5b: Local Outlier Factor (scikit-learn) — NEW
Step 6 : Autoencoder (PyTorch — lightweight)
"""

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# ═══════════════════════════════════════════════════════════════════
# STEP 5: ISOLATION FOREST
# ═══════════════════════════════════════════════════════════════════

def run_isolation_forest(X_scaled, iso_config, model_path):
    """
    Trains an Isolation Forest on the scaled features.
    
    Args:
        X_scaled: numpy array of standardized features
        iso_config: dict with contamination, random_state, n_estimators
        model_path: file path to save the trained model
    
    Returns:
        anomaly_scores: numpy array of anomaly scores (lower = more anomalous)
        predictions: numpy array of predictions (-1 = anomaly, 1 = normal)
    """
    # Create the Isolation Forest model
    model = IsolationForest(
        contamination=iso_config["contamination"],
        random_state=iso_config["random_state"],
        n_estimators=iso_config["n_estimators"],
    )

    # Fit and predict
    predictions = model.fit_predict(X_scaled)

    # Get anomaly scores (negative = more anomalous)
    anomaly_scores = model.decision_function(X_scaled)

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"  Saved Isolation Forest model: {model_path}")

    # Count anomalies
    n_anomalies = int(np.sum(predictions == -1))
    print(f"  Isolation Forest detected {n_anomalies} anomalies "
          f"out of {len(predictions)} samples.")

    return anomaly_scores, predictions


# ═══════════════════════════════════════════════════════════════════
# STEP 5b: LOCAL OUTLIER FACTOR  (NEW)
# ═══════════════════════════════════════════════════════════════════

def run_lof(X_scaled, lof_config, model_path):
    """
    Fits a Local Outlier Factor model in NOVELTY mode and returns outlier scores.

    Novelty=True allows the model to be fitted on clean data and then
    used to score new (potentially anomalous) data via score_samples(),
    avoiding data leakage during evaluation.

    Args:
        X_scaled:   numpy array of standardised features
        lof_config: dict with n_neighbors, contamination
        model_path: file path to persist the fitted model (joblib)

    Returns:
        lof_scores:      numpy array of positive outlier factors
                         (shape == X_scaled.shape[0])
        lof_predictions: numpy array of labels (-1 anomaly, 1 normal)
    """
    model = LocalOutlierFactor(
        n_neighbors=lof_config["n_neighbors"],
        contamination=lof_config["contamination"],
        novelty=True,  # Enable score_samples() for evaluation
    )

    # Fit on the data (assumed clean during pipeline training)
    model.fit(X_scaled)
    
    # Predict labels (-1 anomaly, 1 normal)
    lof_predictions = model.predict(X_scaled)

    # score_samples returns negative scores; negate so higher = more anomalous
    lof_scores = -model.score_samples(X_scaled)

    # Save fitted model
    joblib.dump(model, model_path)
    print(f"  Saved LOF model (novelty=True): {model_path}")

    n_anomalies = int(np.sum(lof_predictions == -1))
    print(f"  LOF detected {n_anomalies} anomalies "
          f"out of {len(lof_predictions)} samples.")
    print(f"  LOF scores — Min: {lof_scores.min():.4f}, "
          f"Max: {lof_scores.max():.4f}, "
          f"Mean: {lof_scores.mean():.4f}")

    return lof_scores, lof_predictions


# ═══════════════════════════════════════════════════════════════════
# STEP 6: AUTOENCODER (PyTorch)
# ═══════════════════════════════════════════════════════════════════

class CropAutoencoder(nn.Module):
    """
    Simple symmetric autoencoder for anomaly detection.
    Architecture: Input → hidden_1 → latent → hidden_2 → Output
    With dropout regularization between layers.
    
    Normal samples reconstruct well (low error).
    Anomalies reconstruct poorly (high error).
    """
    def __init__(self, input_dim, hidden_1, latent, hidden_2, dropout=0.0):
        super(CropAutoencoder, self).__init__()

        # Encoder: compress the input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, latent),
            nn.ReLU(),
        )

        # Decoder: reconstruct the input
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def run_autoencoder(X_scaled, ae_config, model_path, seed):
    """
    Trains a PyTorch autoencoder and computes reconstruction errors.
    
    Args:
        X_scaled: numpy array of standardized features
        ae_config: dict with architecture and training params
        model_path: file path to save the trained model (.pt)
        seed: random seed for reproducibility
    
    Returns:
        reconstruction_errors: numpy array of per-sample MSE
        train_indices: indices used for training
        test_indices: indices used for testing
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # --- Train/test split ---
    n_samples = X_scaled.shape[0]
    n_train = int(n_samples * ae_config["train_ratio"])

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train = X_scaled[train_indices]
    X_test = X_scaled[test_indices]

    print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # --- Convert to PyTorch tensors ---
    train_tensor = torch.FloatTensor(X_train).to(device)
    test_tensor = torch.FloatTensor(X_test).to(device)
    all_tensor = torch.FloatTensor(X_scaled).to(device)

    # Create DataLoader for batched training
    train_dataset = TensorDataset(train_tensor, train_tensor)  # input = target
    train_loader = DataLoader(
        train_dataset,
        batch_size=ae_config["batch_size"],
        shuffle=True,
    )

    # --- Build the model ---
    input_dim = X_scaled.shape[1]
    dropout = ae_config.get("dropout", 0.0)
    model = CropAutoencoder(
        input_dim=input_dim,
        hidden_1=ae_config["hidden_dim_1"],
        latent=ae_config["latent_dim"],
        hidden_2=ae_config["hidden_dim_2"],
        dropout=dropout,
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_config["learning_rate"])

    # --- Training loop with early stopping ---
    epochs = ae_config["epochs"]
    patience = 20
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_data, batch_target in train_loader:
            # Forward pass
            output = model(batch_data)
            loss = criterion(output, batch_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_reconstructed = model(test_tensor)
            val_loss = criterion(val_reconstructed, test_tensor).item()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        # Print progress every 25 epochs
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs} — Train: {avg_loss:.6f}, Val: {val_loss:.6f}")
        
        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch + 1} (best val: {best_val_loss:.6f})")
            break

    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Compute reconstruction errors for ALL samples ---
    model.eval()
    with torch.no_grad():
        reconstructed = model(all_tensor)
        # Per-sample MSE
        errors = torch.mean((all_tensor - reconstructed) ** 2, dim=1)
        reconstruction_errors = errors.cpu().numpy()

    print(f"  Reconstruction error — "
          f"Mean: {np.mean(reconstruction_errors):.6f}, "
          f"Max: {np.max(reconstruction_errors):.6f}")

    # --- Save the model ---
    torch.save(model.state_dict(), model_path)
    print(f"  Saved Autoencoder model: {model_path}")

    return reconstruction_errors, train_indices, test_indices
