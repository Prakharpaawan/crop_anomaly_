"""
ingest.py — raw 9-band GeoTIFFs -> one screened, summarised row per tile-date.

Parallel across tiles. Results cached to Parquet so this runs once.
"""
import os
import re
import glob
import datetime as dt

import numpy as np
import pandas as pd
import rasterio

import config
from config import SELECT, N_WORKERS
from src.indices import screen_scene, summarise

RAW_RE = re.compile(r"^(\d{8})\.tif$")   # excludes the derived *_MSAVI.tif files


def tile_ids(raw_dir=None):
    raw_dir = raw_dir or config.RAW_DIR
    return sorted(
        (d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))),
        key=lambda x: int(x) if x.isdigit() else 1 << 30,
    )


def raw_scenes(tile, raw_dir):
    """(date, path) for every genuine raw acquisition of one tile."""
    out = []
    for p in glob.glob(os.path.join(raw_dir, str(tile), "*.tif")):
        m = RAW_RE.match(os.path.basename(p))
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%d"), p))
    return sorted(out)


def select_dates(rows):
    """
    Keep well-spread acquisitions across the season.

    Greedy by quality, subject to a minimum gap, so we get season-wide coverage
    instead of a cluster of images from one cloud-free fortnight. Falls back to
    filling remaining slots by date order if the gap rule is too restrictive.
    """
    if len(rows) <= SELECT["target_dates"]:
        return sorted(rows, key=lambda r: r["date"])

    ranked = sorted(rows, key=lambda r: (-r["_quality"], r["date"]))
    gap = dt.timedelta(days=SELECT["min_days_apart"])
    chosen = []
    for r in ranked:
        if len(chosen) >= SELECT["target_dates"]:
            break
        if all(abs(r["date"] - c["date"]) >= gap for c in chosen):
            chosen.append(r)
    if len(chosen) < SELECT["target_dates"]:
        taken = {c["date"] for c in chosen}
        for r in ranked:
            if len(chosen) >= SELECT["target_dates"]:
                break
            if r["date"] not in taken:
                chosen.append(r)
    return sorted(chosen, key=lambda r: r["date"])


def process_tile(tile, raw_dir):
    """Screen, summarise and date-select one tile. Returns a list of dict rows."""
    rows, rejected = [], 0
    for date, path in raw_scenes(tile, raw_dir):
        try:
            with rasterio.open(path) as s:
                if s.count < 9:
                    continue
                cube = s.read().astype(np.float64)
                nodata = s.nodata
        except Exception:
            continue

        ok, clean, diag = screen_scene(cube, nodata)
        if not ok:
            rejected += 1
            continue
        stats = summarise(cube, clean)
        if stats is None:
            rejected += 1
            continue

        stats["tile_id"] = int(tile)
        stats["date"] = date
        # quality proxy used only for date selection: prefer clear, complete scenes
        stats["_quality"] = (1.0 - diag["cloud_frac"]) * diag["valid_frac"]
        rows.append(stats)

    if len(rows) < SELECT["min_dates"]:
        return []
    keep = select_dates(rows)
    for r in keep:
        r.pop("_quality", None)
        r["n_rejected"] = rejected
    return keep


def _worker(job):
    """job = (tile, raw_dir) — the directory travels with the work item so
    spawned workers cannot fall back to the default year."""
    tile, raw_dir = job
    try:
        return process_tile(tile, raw_dir)
    except Exception as exc:                      # never let one tile kill the run
        return [{"tile_id": int(tile), "_error": str(exc)}]


def build(limit=None, workers=N_WORKERS, progress_every=200, raw_dir=None):
    """Ingest every tile (or the first `limit`) and return a tidy DataFrame."""
    from concurrent.futures import ProcessPoolExecutor

    raw_dir = raw_dir or config.RAW_DIR
    tiles = tile_ids(raw_dir)
    if limit:
        tiles = tiles[:limit]
    jobs = [(t, raw_dir) for t in tiles]

    all_rows, errors, dropped = [], [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(_worker, jobs, chunksize=8), 1):
            if rows and "_error" in rows[0]:
                errors.append(rows[0])
            elif rows:
                all_rows.extend(rows)
            else:
                dropped += 1
            if progress_every and i % progress_every == 0:
                print(f"    {i}/{len(tiles)} tiles · {len(all_rows):,} observations "
                      f"· {dropped} tiles dropped", flush=True)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["tile_id", "date"]).reset_index(drop=True)
    print(f"    done: {df['tile_id'].nunique() if not df.empty else 0} tiles kept, "
          f"{dropped} dropped (too few usable dates), {len(errors)} errored")
    return df
