"""
indices.py — band arithmetic and spectral quality screening.

Everything here works on a (9, H, W) float array of raw Sentinel-2 reflectance
plus a boolean valid-pixel mask. No file I/O.
"""
import numpy as np

from config import BANDS, SCREEN


def _safe_ratio(num, den):
    """Normalised-difference denominator guard: 0/0 -> nan, not a crash."""
    den = np.where(den == 0, np.nan, den)
    return num / den


def band(cube, name):
    return cube[BANDS[name]]


def ndvi(cube):
    nir, red = band(cube, "NIR"), band(cube, "RED")
    return _safe_ratio(nir - red, nir + red)


def ndre(cube):
    """Red-edge index. Responds to chlorophyll loss before NDVI does."""
    nir, re1 = band(cube, "NIR"), band(cube, "RE1")
    return _safe_ratio(nir - re1, nir + re1)


def ndmi(cube):
    """Moisture index. Tracks canopy water content — the water-stress channel."""
    nir, swir = band(cube, "NIR"), band(cube, "SWIR1")
    return _safe_ratio(nir - swir, nir + swir)


def msavi(cube):
    """
    Modified Soil Adjusted Vegetation Index.

        MSAVI = (2*N + 1 - sqrt((2*N + 1)^2 - 8*(N - R))) / 2

    Computed on reflectance scaled to [0, 1]; the discriminant is clipped at
    zero because sensor noise can push it slightly negative over bare soil.
    """
    nir = band(cube, "NIR") / 10000.0
    red = band(cube, "RED") / 10000.0
    t = 2.0 * nir + 1.0
    disc = np.clip(t * t - 8.0 * (nir - red), 0.0, None)
    return (t - np.sqrt(disc)) / 2.0


INDEX_FN = {"msavi": msavi, "ndvi": ndvi, "ndre": ndre, "ndmi": ndmi}


def valid_mask(cube, nodata):
    """Pixels that carry real data in every band we rely on."""
    m = np.isfinite(cube).all(axis=0)
    if nodata is not None:
        m &= (cube != nodata).all(axis=0)
    # reflectance must be physically plausible
    m &= (cube >= 0).all(axis=0)
    return m


def cloud_mask(cube, valid):
    """
    Spectral cloud and shadow screen.

    Cloud is bright in blue AND bright in SWIR1 while showing no vegetation
    signal — a combination canopy never produces. Shadow is very dark in NIR.

    This replaces the blur-based heuristic used previously, which could not
    distinguish a cloud from a genuinely smooth field.
    """
    with np.errstate(invalid="ignore"):
        veg = ndvi(cube)
    bright = (band(cube, "BLUE") > SCREEN["cloud_blue_min"]) & \
             (band(cube, "SWIR1") > SCREEN["cloud_swir1_min"])
    cloud = bright & (np.nan_to_num(veg, nan=0.0) < SCREEN["cloud_ndvi_max"])
    shadow = band(cube, "NIR") < SCREEN["shadow_nir_max"]
    return (cloud | shadow) & valid


def screen_scene(cube, nodata):
    """
    Decide whether one acquisition is usable.

    Returns (ok, clean_mask, diagnostics).
    """
    valid = valid_mask(cube, nodata)
    total = valid.size
    valid_frac = float(valid.sum()) / total
    if valid_frac < SCREEN["min_valid_frac"]:
        return False, valid, {"reason": "nodata", "valid_frac": valid_frac,
                              "cloud_frac": np.nan, "scene_std": np.nan}

    cloud = cloud_mask(cube, valid)
    cloud_frac = float(cloud.sum()) / max(int(valid.sum()), 1)
    clean = valid & ~cloud

    if cloud_frac > SCREEN["max_cloud_frac"] or clean.sum() < 0.25 * total:
        return False, clean, {"reason": "cloud", "valid_frac": valid_frac,
                              "cloud_frac": cloud_frac, "scene_std": np.nan}

    scene_std = float(np.std(band(cube, "NIR")[clean]))
    if scene_std < SCREEN["min_scene_std"]:
        return False, clean, {"reason": "flat", "valid_frac": valid_frac,
                              "cloud_frac": cloud_frac, "scene_std": scene_std}

    return True, clean, {"reason": "ok", "valid_frac": valid_frac,
                         "cloud_frac": cloud_frac, "scene_std": scene_std}


def summarise(cube, clean):
    """
    Reduce one screened acquisition to summary statistics per index.

    Percentiles rather than only the mean, because stress often arrives in
    patches: the low tail and the spread move before the average does.
    """
    out = {}
    n = int(clean.sum())
    if n < 32:
        return None
    for name, fn in INDEX_FN.items():
        with np.errstate(invalid="ignore", divide="ignore"):
            v = fn(cube)[clean]
        v = v[np.isfinite(v)]
        if v.size < 32:
            return None
        p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
        out[f"{name}_mean"] = float(v.mean())
        out[f"{name}_std"] = float(v.std())
        out[f"{name}_p10"] = float(p10)
        out[f"{name}_p25"] = float(p25)
        out[f"{name}_p50"] = float(p50)
        out[f"{name}_p75"] = float(p75)
        out[f"{name}_p90"] = float(p90)
        out[f"{name}_iqr"] = float(p75 - p25)
    out["valid_frac"] = float(n) / clean.size
    return out
