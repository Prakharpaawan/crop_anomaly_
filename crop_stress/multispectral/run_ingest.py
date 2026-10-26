"""
run_ingest.py — stage 1-2: raw rasters -> screened observation table.

Slow (reads ~236k rasters). Runs once; the result is cached to Parquet and
every later stage loads that.

    python run_ingest.py            # all tiles
    python run_ingest.py --limit 20 # quick smoke test
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                        # noqa: E402
from config import N_WORKERS                         # noqa: E402
from src.ingest import build                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--year", type=int, default=config.TARGET_YEAR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    config.RAW_DIR = config.raw_dir(args.year)
    out = args.out or config.observations_csv(args.year)

    print(f"Ingesting {'all' if not args.limit else args.limit} tiles "
          f"from {args.year} with {args.workers} workers...")
    t0 = time.time()
    df = build(limit=args.limit, workers=args.workers,
               raw_dir=config.raw_dir(args.year))
    if df.empty:
        print("No observations produced — check the screening thresholds.")
        return

    df.to_csv(out, index=False)
    el = time.time() - t0
    print(f"\nWrote {args.out}")
    print(f"  {len(df):,} observations · {df['tile_id'].nunique():,} tiles "
          f"· {len(df.columns)} columns")
    print(f"  dates per tile: min {df.groupby('tile_id').size().min()} "
          f"max {df.groupby('tile_id').size().max()} "
          f"median {int(df.groupby('tile_id').size().median())}")
    print(f"  elapsed {el/60:.1f} min")


if __name__ == "__main__":
    main()
