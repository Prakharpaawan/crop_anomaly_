"""
config.py — every tunable parameter for experiment3, in one place.

Nothing in src/ hardcodes a value that belongs here.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))   # crop_stress/multispectral
PKG = os.path.dirname(ROOT)                         # crop_stress


def _find_data_root(start):
    """
    Walk up to the folder holding the raw archive.

    This used to assume the archive sat one level above; the study now lives
    two levels down inside `crop_stress/`, so the depth is discovered instead.
    """
    p = start
    while os.path.dirname(p) != p:
        if os.path.isdir(os.path.join(p, "archive (1)")):
            return p
        p = os.path.dirname(p)
    return os.path.dirname(start)


PROJECT = _find_data_root(ROOT)

# ── Data ────────────────────────────────────────────────────────────────
LOMBARDIA = os.path.join(PROJECT, "archive (1)", "lombardia")
TARGET_YEAR = 2019      # the season we monitor
HISTORY_YEAR = 2018     # the season the detector learns "normal" from


def raw_dir(year=TARGET_YEAR):
    return os.path.join(LOMBARDIA, f"data{year}")


RAW_DIR = raw_dir(TARGET_YEAR)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
# One outputs tree for the whole package, one subfolder per study.
OUT_DIR = os.path.join(PKG, "outputs", "multispectral")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for _d in (DATA_DIR, MODEL_DIR, OUT_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

# CSV rather than Parquet: no pyarrow/fastparquet in this environment, and a
# plain CSV stays inspectable without extra tooling.
OBSERVATIONS_CSV = os.path.join(DATA_DIR, "observations.csv")


def observations_csv(year=TARGET_YEAR):
    return os.path.join(DATA_DIR, f"observations_{year}.csv")

# ── Sentinel-2 band layout of the raw 9-band rasters ────────────────────
# Verified from a July acquisition: blue/green/red low, red-edge climb
# across bands 4-6, NIR plateau at band 7, SWIR falloff at bands 8-9.
BANDS = {
    "BLUE":  0,   # B2   490 nm
    "GREEN": 1,   # B3   560 nm
    "RED":   2,   # B4   665 nm
    "RE1":   3,   # B5   705 nm
    "RE2":   4,   # B6   740 nm
    "RE3":   5,   # B7   783 nm
    "NIR":   6,   # B8   842 nm
    "SWIR1": 7,   # B11 1610 nm
    "SWIR2": 8,   # B12 2190 nm
}
INDICES = ["msavi", "ndvi", "ndre", "ndmi"]

# ── Stage 1: quality screening ──────────────────────────────────────────
# A cloudy pixel is bright in blue AND bright in SWIR1 while showing little
# vegetation signal. Thresholds are in raw reflectance units (scaled x10000).
SCREEN = {
    "cloud_blue_min":   1800,   # brighter than this in blue is suspicious
    "cloud_swir1_min":  1500,   # ...and bright in SWIR too -> cloud, not canopy
    "cloud_ndvi_max":   0.20,   # ...with no vegetation signal
    "shadow_nir_max":    600,   # very dark NIR -> cloud shadow
    "max_cloud_frac":    0.25,  # reject the acquisition above this cloud fraction
    "min_valid_frac":    0.60,  # reject if too many nodata pixels
    "min_scene_std":      150,  # reject flat scenes (sensor fault / total overcast)
}

# ── Stage 2: date selection ─────────────────────────────────────────────
SELECT = {
    "target_dates":   40,   # per tile (the old pipeline used 25)
    "min_days_apart":  5,   # Sentinel-2 revisit is ~5 days
    "min_dates":      30,   # drop tiles that cannot reach this
}

# ── Stage 3: temporal features ──────────────────────────────────────────
FEATURES = {
    # 5 measured best on held-out fields: a wider window attenuates short
    # events (a 2-observation 40% drop retains only 25% of its depth at
    # window 7, 20% at window 9) while a narrower one leaves cloud jitter in.
    "savgol_window":  5,
    "savgol_order":   2,
    "rolling_window": 3,
    "residual_window": 9,  # wide median for the robust residual (see features.py)
}

# ── Stage 4: train / evaluation split (by TILE, never by row) ───────────
SPLIT = {"fit_fraction": 0.60, "seed": 7}

# ── Stage 5: detectors ──────────────────────────────────────────────────
LOF = {"n_neighbors": 20, "novelty": True}
LOF_SWEEP = [5, 10, 20, 35, 50, 75]

IFOREST = {"n_estimators": 200, "max_samples": "auto", "random_state": 7}

AUTOENCODER = {
    "hidden": 48, "latent": 6, "dropout": 0.15,
    "epochs": 120, "batch_size": 256, "lr": 1e-3,
    "val_fraction": 0.15, "patience": 12, "seed": 7,
}

# ── Stage 6: decision rule ──────────────────────────────────────────────
DECIDE = {
    "threshold_percentile": 90,   # calibrated on CLEAN fit-set scores
    "min_consecutive":       2,   # causal persistence
}
THRESHOLD_SWEEP = list(range(70, 100, 2))

# ── Evaluation ──────────────────────────────────────────────────────────
EVAL = {
    "seeds": [42, 123, 456, 789, 1024],
    "injection_ratio": 0.15,
    "severity": 0.40,
    "duration": 2,
    "bootstrap_n": 400,
}
SEVERITY_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
DURATION_GRID = [1, 2, 3]

# ── Runtime ─────────────────────────────────────────────────────────────
N_WORKERS = 8
