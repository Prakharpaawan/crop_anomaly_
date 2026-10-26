"""
detectors.py — anomaly scorers.

Every scorer has the same contract:

    fit(X_clean_fit)  ->  self
    score(X)          ->  1-D array, higher = more anomalous

That uniformity is what makes the comparison fair: the learned detectors and
the rule-based baselines all flow through the identical threshold and
persistence path downstream, so the only thing that differs is the score.

`contamination` is deliberately absent. For LOF in novelty mode and for
IsolationForest it only shifts an offset used by predict(); we threshold on raw
scores by percentile, and a constant offset cannot change a percentile ranking.
Carrying it would imply a tuning knob that does nothing.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from config import IFOREST, AUTOENCODER


# ══════════════════════════════════════════════════════════════════════
#  Learned detectors
# ══════════════════════════════════════════════════════════════════════
class LOFDetector:
    """Local Outlier Factor: is this point sparse relative to its neighbours?"""
    name = "LOF"

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                                         novelty=True)
        self.model_.fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        # score_samples is negative and higher-is-normal; negate it
        return -self.model_.score_samples(self.scaler_.transform(X))


class IForestDetector:
    """Isolation Forest: how few random splits isolate this point?"""
    name = "Isolation Forest"

    def fit(self, X):
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = IsolationForest(
            n_estimators=IFOREST["n_estimators"],
            max_samples=IFOREST["max_samples"],
            random_state=IFOREST["random_state"],
        ).fit(self.scaler_.transform(X))
        return self

    def score(self, X):
        return -self.model_.score_samples(self.scaler_.transform(X))


class AEDetector:
    """Autoencoder: reconstruction error after learning the normal manifold."""
    name = "Autoencoder"

    def fit(self, X):
        import torch
        import torch.nn as nn

        cfg = AUTOENCODER
        torch.manual_seed(cfg["seed"])
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)

        n_val = max(1, int(len(Xs) * cfg["val_fraction"]))
        rng = np.random.default_rng(cfg["seed"])
        perm = rng.permutation(len(Xs))
        val, tr = Xs[perm[:n_val]], Xs[perm[n_val:]]

        d = Xs.shape[1]
        self.net_ = nn.Sequential(
            nn.Linear(d, cfg["hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden"], cfg["latent"]), nn.ReLU(),
            nn.Linear(cfg["latent"], cfg["hidden"]), nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden"], d),
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=cfg["lr"])
        lossf = nn.MSELoss()
        tr_t = torch.from_numpy(tr)
        val_t = torch.from_numpy(val)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tr_t, tr_t),
            batch_size=cfg["batch_size"], shuffle=True)

        best, wait, best_state = np.inf, 0, None
        for _ in range(cfg["epochs"]):
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = lossf(self.net_(xb), yb)
                loss.backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = lossf(self.net_(val_t), val_t).item()
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def score(self, X):
        import torch
        Xs = self.scaler_.transform(X).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(Xs)
            err = ((t - self.net_(t)) ** 2).mean(dim=1).numpy()
        return err


# ══════════════════════════════════════════════════════════════════════
#  Rule-based baselines
#
#  These need no fitting. They exist to answer the question a reviewer asks
#  first: does the learned model actually beat a simple rule?
# ══════════════════════════════════════════════════════════════════════
class BaselineDetector:
    """Wraps a column-wise rule so it satisfies the same interface."""

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def fit(self, X):        # nothing to learn
        return self

    def score_df(self, df):
        return np.nan_to_num(self.fn(df).astype(float))


def _rule_pct_drop(df):
    """Flag on the sharpest relative fall in greenness since the last pass."""
    return -df["ndvi_pct"].to_numpy(float)


def _rule_zscore(df):
    """Per-field z-score of the smoothed level, oriented so drops score high."""
    g = df.groupby("tile_id")["ndvi_sm"]
    mu = g.transform("mean").to_numpy(float)
    sd = g.transform("std").replace(0, np.nan).to_numpy(float)
    return -(df["ndvi_sm"].to_numpy(float) - mu) / sd


def _rule_rolling_dev(df):
    """Distance below the field's own rolling median, in local units."""
    sd = df["ndvi_rstd"].replace(0, np.nan).to_numpy(float)
    return df["ndvi_dev"].to_numpy(float) / sd


BASELINES = [
    BaselineDetector("Fixed % drop", _rule_pct_drop),
    BaselineDetector("Per-field z-score", _rule_zscore),
    BaselineDetector("Rolling-median dev.", _rule_rolling_dev),
]
