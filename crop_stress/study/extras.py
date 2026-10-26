"""
extras.py — the illustrative material the tables cannot carry.

Nine bar charts prove a result but do not let a reader see the mechanism, and a
remote-sensing paper with no imagery in it invites the obvious objection. This
module computes:

  * a worked example on one held-out field — hits, misses and suppressions
  * the raw MSAVI imagery for that field, as a data figure
  * threshold-free curves (PR, ROC) with the deployed operating point marked
  * score separability between healthy and stressed observations
  * the spread of per-field thresholds, which is what justifies calibrating
    per field at all
  * a bootstrap significance test behind the headline claim
  * the study-area geometry, from the rasters' own georeferencing, and how the
    three-way split falls on the ground
  * where the false alarms land in space, and how concentrated they are
  * the season-long index trajectory the detector is measured against
  * the correlation structure of the 16 features

Appends to outputs/results.json under "extras".
"""

import glob
import json
import os

import numpy as np

import core as C
import study as S


DEMO_TILE_CANDIDATES = 40      # search this many test fields for a good example


# ══════════════════════════════════════════════════════════════════════
#  Worked example
# ══════════════════════════════════════════════════════════════════════
def worked_example(b, chosen, seed=42):
    """
    Pick one held-out field and record its whole decision path.

    Chooses a field that shows BOTH a confirmed detection and a suppressed
    exceedance, because a clean success teaches the reader nothing about the
    cost of the persistence rule. If no such field exists, falls back to the
    field with the most injected observations.
    """
    d = b.draw(seed)["test"]
    scores = C.score_lof(*b.lof, d["X"])
    gt = d["gt"]
    fz = S.operating_thresholds(b, "test", chosen)
    slices = b.slices["test"]

    best = None
    for i, (tile, idx) in enumerate(slices[:DEMO_TILE_CANDIDATES * 4]):
        s, g, thr = scores[idx], gt[idx], fz[i]
        raw = s > thr
        conf = C.strict_consecutive(raw, C.MIN_CONSECUTIVE)
        suppressed = raw & ~conf
        hits = int((conf & (g == 1)).sum())
        miss = int(((~conf) & (g == 1)).sum())
        fa = int((conf & (g == 0)).sum())
        score = (suppressed.sum() > 0) * 100 + hits * 10 - fa
        if g.sum() >= 2 and (best is None or score > best["rank"]):
            best = {"rank": int(score), "tile": int(tile), "idx": idx,
                    "scores": s.tolist(), "gt": g.tolist(),
                    "threshold": float(thr),
                    "raw": raw.tolist(), "confirmed": conf.tolist(),
                    "suppressed": suppressed.tolist(),
                    "n_injected": int(g.sum()), "hits": hits,
                    "missed": miss, "false_alarms": fa}
    if best is None:
        return None

    # the field's own MSAVI trajectory, clean and injected, plus its dates
    df = b.base["df"]
    mask = b.mask["test"]
    sub = df[mask].reset_index(drop=True)
    rows = sub.iloc[best["idx"]]
    best["dates"] = rows["date"].tolist()
    best["clean_p50"] = rows["p50"].astype(float).tolist()

    Xm, _gt, p50m = C.injected(df, b.tiles, seed)
    best["injected_p50"] = p50m[mask][best["idx"]].astype(float).tolist()
    best.pop("idx")
    return best


# ══════════════════════════════════════════════════════════════════════
#  Imagery — a data figure, not a results figure
# ══════════════════════════════════════════════════════════════════════
def imagery(tile_id, n_panels=5):
    """
    Raw MSAVI rasters for one field across its season.

    Deliberately NOT annotated with detections. The injected stress lives in
    the feature table, not in these pixels, so drawing alarm markers on real
    imagery would imply the model saw something in the raster that it did not.
    This figure exists to show what the input actually looks like.
    """
    import rasterio
    d = os.path.join(C.PROJECT, "dataset_clean_2", str(tile_id))
    files = sorted(glob.glob(os.path.join(d, "*_MSAVI.tif")))
    if not files:
        return None
    pick = np.linspace(0, len(files) - 1, n_panels).astype(int)
    panels, dates, means = [], [], []
    for i in pick:
        with rasterio.open(files[i]) as src:
            a = src.read(1).astype(float)
            nd = src.nodata
            if nd is not None:
                a[a == nd] = np.nan
            a[a < -1e30] = np.nan
        panels.append(a)
        stem = os.path.basename(files[i])[:8]
        dates.append(f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}")
        means.append(float(np.nanmean(a)))

    stack = np.concatenate([p[~np.isnan(p)].ravel() for p in panels])
    vmin, vmax = np.percentile(stack, [2, 98])
    return {"tile": int(tile_id), "dates": dates,
            "panels": [p.tolist() for p in panels],
            "means": means, "vmin": float(vmin), "vmax": float(vmax),
            "n_available": len(files)}


# ══════════════════════════════════════════════════════════════════════
#  Threshold-free curves
# ══════════════════════════════════════════════════════════════════════
def curves(b, chosen, seed=42):
    """PR and ROC for the LOF score, with the deployed operating point marked."""
    from sklearn.metrics import (precision_recall_curve, roc_curve,
                                 average_precision_score, roc_auc_score)
    d = b.draw(seed)["test"]
    s, g = C.score_lof(*b.lof, d["X"]), d["gt"]
    fz = S.operating_thresholds(b, "test", chosen)
    pred = C.predict(s, b.slices["test"], C.MIN_CONSECUTIVE, frozen=fz)
    op = C.metrics(g, pred, s)

    p, r, _ = precision_recall_curve(g, s)
    fpr, tpr, _ = roc_curve(g, s)
    step = max(1, len(p) // 900)          # keep the JSON small
    return {"pr": {"precision": p[::step].tolist(), "recall": r[::step].tolist()},
            "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
            "ap": float(average_precision_score(g, s)),
            "auc": float(roc_auc_score(g, s)),
            "operating_point": {"precision": op["precision"],
                                "recall": op["recall"]}}


def separability(b, seed=42, bins=70):
    """Score histograms for healthy versus stressed observations."""
    d = b.draw(seed)["test"]
    s, g = C.score_lof(*b.lof, d["X"]), d["gt"]
    lo, hi = float(np.percentile(s, 0.2)), float(np.percentile(s, 99.8))
    edges = np.linspace(lo, hi, bins + 1)
    h0, _ = np.histogram(s[g == 0], bins=edges)
    h1, _ = np.histogram(s[g == 1], bins=edges)
    return {"edges": edges.tolist(), "healthy": h0.tolist(),
            "stressed": h1.tolist(),
            "median_healthy": float(np.median(s[g == 0])),
            "median_stressed": float(np.median(s[g == 1]))}


def threshold_spread(b, chosen):
    """
    The deployed per-field thresholds, against the single global cut.

    This is the empirical case for calibrating per field: if the spread were
    narrow, one global threshold would do and the extra machinery would not be
    earning its place.
    """
    fz = S.operating_thresholds(b, "test", chosen)
    clean = b.clean["test"]
    return {"thresholds": fz.tolist(),
            "global_equivalent": float(np.percentile(clean, 97.25)),
            "ratio": float(np.max(fz) / max(np.min(fz), 1e-9)),
            "median": float(np.median(fz))}


# ══════════════════════════════════════════════════════════════════════
#  Significance behind the headline
# ══════════════════════════════════════════════════════════════════════
def significance(b, chosen, seeds=C.SEEDS, n_boot=2000, rng_seed=11):
    """
    Bootstrap over FIELDS — the unit that actually varies — for the paper's
    central claim: the new operating point beats Study 2's rule.

    Five seeds cannot support a test; resampling fields can.
    """
    fz = S.operating_thresholds(b, "test", chosen)
    slices = b.slices["test"]
    A_counts, B_counts = [], []
    for seed in seeds:
        d = b.draw(seed)["test"]
        s, g = C.score_lof(*b.lof, d["X"]), d["gt"]
        new = C.predict(s, slices, C.MIN_CONSECUTIVE, frozen=fz)
        old = C.predict(s, slices, C.MIN_CONSECUTIVE, pctl=C.LEGACY_PCTL)
        for _t, idx in slices:
            gg = g[idx]
            for pred, sink in ((new, A_counts), (old, B_counts)):
                pp = pred[idx]
                sink.append((int((gg & pp).sum()),
                             int(((1 - gg) & pp).sum()),
                             int((gg & (1 - pp)).sum())))
    A, B = np.array(A_counts), np.array(B_counts)

    def f1(c):
        tp, fp, fn = c.sum(axis=0)
        den = 2 * tp + fp + fn
        return 0.0 if den == 0 else 2 * tp / den

    obs = f1(A) - f1(B)
    rng = np.random.RandomState(rng_seed)
    n = len(A)
    boot = np.array([f1(A[p]) - f1(B[p])
                     for p in (rng.randint(0, n, n) for _ in range(n_boot))])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    perm = np.empty(n_boot)
    for i in range(n_boot):
        sw = rng.rand(n) < 0.5
        perm[i] = f1(np.where(sw[:, None], B, A)) - f1(np.where(sw[:, None], A, B))
    return {"delta_f1": float(obs), "ci95": [float(lo), float(hi)],
            "p_value": float((np.abs(perm) >= abs(obs)).mean()),
            "n_units": int(n), "n_boot": n_boot,
            "boot": boot[::2].tolist()}


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
#  Study-area geometry
# ══════════════════════════════════════════════════════════════════════
GEOM_CACHE = os.path.join(C.CACHE_DIR, "geometry.json")


def geometry(b):
    """
    Where every tile actually sits, read from the rasters' own georeferencing.

    Each tile is a fixed 48x48 pixel square at 10 m -- 480 m on a side -- so the
    study area is a regular grid, not a set of digitised field boundaries. That
    is worth showing rather than asserting.

    Also records how the three-way split falls on the ground. The split is drawn
    at random over tile ids, which leaves held-out tiles interleaved with fitted
    ones; the figure makes that visible and Section VII states the consequence.
    """
    import rasterio
    if os.path.exists(GEOM_CACHE):
        with open(GEOM_CACHE, encoding="utf-8") as f:
            geo = json.load(f)
    else:
        geo = {"crs": None, "tiles": {}}
        root = os.path.join(C.PROJECT, "dataset_clean_2")
        for d in sorted(os.listdir(root), key=lambda x: int(x)):
            files = sorted(glob.glob(os.path.join(root, d, "*_MSAVI.tif")))
            if not files:
                continue
            with rasterio.open(files[0]) as src:
                bb = src.bounds
                if geo["crs"] is None:
                    geo["crs"] = str(src.crs)
                geo["tiles"][d] = [round((bb.left + bb.right) / 2, 1),
                                   round((bb.bottom + bb.top) / 2, 1),
                                   round(bb.right - bb.left, 1)]
        os.makedirs(C.CACHE_DIR, exist_ok=True)
        with open(GEOM_CACHE, "w", encoding="utf-8") as f:
            json.dump(geo, f)

    T = {int(k): v for k, v in geo["tiles"].items()}
    split = {}
    for name in ("fit", "cal", "test"):
        for t in b.splits[name]:
            split[int(t)] = name

    xs = np.array([v[0] for v in T.values()])
    ys = np.array([v[1] for v in T.values()])
    side = float(np.median([v[2] for v in T.values()]))

    # how interleaved is the split? nearest-neighbour agreement, and how far a
    # test tile sits from the nearest tile the detector was fitted on.
    from scipy.spatial import cKDTree
    ids = np.array(sorted(T))
    xy = np.array([[T[t][0], T[t][1]] for t in ids])
    lab = np.array([split.get(int(t), "?") for t in ids])
    tree = cKDTree(xy)
    _d, idx = tree.query(xy, k=2)
    same_nn = float((lab[idx[:, 1]] == lab).mean())
    fit_xy = xy[lab == "fit"]
    dmin, _ = cKDTree(fit_xy).query(xy[lab == "test"], k=1)

    return {
        "crs": geo["crs"], "side_m": side,
        "x": {str(t): T[t][0] for t in T}, "y": {str(t): T[t][1] for t in T},
        "split": {str(t): split.get(t, "?") for t in T},
        "extent_km": [float(np.ptp(xs) / 1000), float(np.ptp(ys) / 1000)],
        "area_km2": float(len(T) * (side / 1000) ** 2),
        "n_tiles": len(T),
        "nn_same_split": same_nn,
        "nn_same_split_expected": float(sum(f * f for f in C.SPLIT_FRACTIONS)),
        "test_to_fit_median_m": float(np.median(dmin)),
        "test_adjacent_to_fit": float((dmin <= side).mean()),
    }


# ══════════════════════════════════════════════════════════════════════
#  False alarms in space
# ══════════════════════════════════════════════════════════════════════
def alarm_map(b, chosen, seeds=C.SEEDS):
    """
    False alarms per held-out tile, averaged over the seeds.

    The absolute count is reported everywhere else as one number; this asks
    whether those alarms are spread evenly or concentrated, because a handful
    of chronically unstable tiles is a different operational problem from a
    uniform background rate.
    """
    fz = S.operating_thresholds(b, "test", chosen)
    sl = b.slices["test"]
    fp = np.zeros(len(sl))
    for seed in seeds:
        d = b.draw(seed)["test"]
        s = C.score_lof(*b.lof, d["X"])
        pred = C.predict(s, sl, persistence=C.MIN_CONSECUTIVE, frozen=fz)
        g = d["gt"].astype(bool)
        p = pred.astype(bool)
        for i, (_t, idx) in enumerate(sl):
            fp[i] += np.sum(p[idx] & ~g[idx])
    fp /= len(seeds)

    order = np.sort(fp)[::-1]
    tot = max(order.sum(), 1e-9)
    return {
        "tiles": [int(t) for t, _i in sl],
        "false_alarms": fp.tolist(),
        "total": float(fp.sum()),
        "zero_fraction": float((fp == 0).mean()),
        "top_decile_share": float(order[:max(1, len(order) // 10)].sum() / tot),
        "max_per_tile": float(fp.max()),
    }


# ══════════════════════════════════════════════════════════════════════
#  Season-long trajectory, and the feature correlation structure
# ══════════════════════════════════════════════════════════════════════
def phenology(b):
    """
    Median index across all tiles at each retained pass, with the spread.

    The seasonal swing is larger than the injected stress, which is the whole
    reason thresholds are per-tile and computed against a rolling baseline.
    """
    df = b.base["df"].copy()
    df["k"] = df.groupby("tile_id").cumcount()
    g = df.groupby("k")["p50"]
    med = g.median()
    dates = df.groupby("k")["date"].agg(lambda s: sorted(s)[len(s) // 2])
    swing = float(med.max() / max(med.min(), 1e-9))
    return {"k": [int(v) for v in med.index],
            "date": [str(d)[:10] for d in dates],
            "median": [float(v) for v in med],
            "p10": [float(v) for v in g.quantile(.10)],
            "p90": [float(v) for v in g.quantile(.90)],
            "seasonal_swing": swing}


def feature_stats(b):
    """
    Correlation among the 16 features, and any that carry no information.

    valid_pixel_ratio is constant: every retained raster is a full square of
    valid pixels, because the screening step rejected anything else before this
    table was built. It is listed as a feature but the detector cannot use it.
    """
    df = b.base["df"]
    cols = list(C.FEATURE_COLUMNS)
    M = df[cols].corr().values
    const = [c for c in cols if float(df[c].std()) == 0.0]
    off = np.abs(M[np.triu_indices(len(cols), 1)])
    off = off[np.isfinite(off)]
    return {"columns": cols, "corr": M.tolist(),
            "constant": const,
            "n_informative": int(len(cols) - len(const)),
            "median_abs_r": float(np.median(off)),
            "n_pairs_above_0p9": int((off > 0.9).sum())}


def compute(b, chosen):
    """Everything in one call, for run_all.py to reuse an existing bench."""
    print("  worked example...")
    ex = worked_example(b, chosen)
    print(f"    field {ex['tile']}: {ex['n_injected']} injected, "
          f"{ex['hits']} confirmed, {ex['missed']} missed, "
          f"{sum(ex['suppressed'])} suppressed, {ex['false_alarms']} false alarms")

    print("  imagery...")
    img = imagery(ex["tile"])
    if img is None:
        print("    no rasters for that field; trying tile 2")
        img = imagery(2)

    print("  PR / ROC curves...")
    cur = curves(b, chosen)
    print(f"    AP={cur['ap']:.4f}  AUC={cur['auc']:.4f}")

    print("  score separability...")
    sep = separability(b)

    print("  threshold spread...")
    spr = threshold_spread(b, chosen)
    print(f"    {spr['ratio']:.1f}x spread across fields")

    print("  significance behind the headline...")
    sig = significance(b, chosen)
    print(f"    dF1={sig['delta_f1']:+.4f} CI[{sig['ci95'][0]:+.4f}, "
          f"{sig['ci95'][1]:+.4f}] p={sig['p_value']:.4f}")

    print("  study-area geometry...")
    geo = geometry(b)
    print(f"    {geo['n_tiles']:,} tiles, {geo['side_m']:.0f} m each, "
          f"{geo['extent_km'][0]:.1f} x {geo['extent_km'][1]:.1f} km, "
          f"{geo['area_km2']:.0f} km2, {geo['crs']}")
    print(f"    {geo['test_adjacent_to_fit']*100:.0f}% of test tiles are "
          f"edge-adjacent to a fit tile")

    print("  false alarms in space...")
    amap = alarm_map(b, chosen)
    print(f"    {amap['total']:.0f} total; {amap['zero_fraction']*100:.0f}% of "
          f"tiles clean; noisiest decile carries "
          f"{amap['top_decile_share']*100:.0f}%")

    print("  seasonal trajectory...")
    phen = phenology(b)
    print(f"    seasonal swing {phen['seasonal_swing']:.1f}x")

    print("  feature correlation...")
    fst = feature_stats(b)
    print(f"    median |r| {fst['median_abs_r']:.2f}, "
          f"{fst['n_pairs_above_0p9']} pairs >0.9, "
          f"constant: {fst['constant'] or 'none'}")

    return {"example": ex, "imagery": img, "curves": cur,
            "separability": sep, "threshold_spread": spr, "significance": sig,
            "geometry": geo, "alarm_map": amap, "phenology": phen,
            "features": fst}


def main():
    """Standalone entry point — rebuilds the bench, then appends to results.json."""
    C.ensure_dirs()
    path = os.path.join(C.OUT_DIR, "results.json")
    with open(path, encoding="utf-8") as f:
        R = json.load(f)

    print("Building bench (this refits the detectors)...")
    b = S.Bench(verbose=True)
    R["extras"] = compute(b, R["a"]["chosen"])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2)
    print(f"\nAppended 'extras' to {path}")


if __name__ == "__main__":
    main()
