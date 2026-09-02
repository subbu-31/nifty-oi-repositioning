#!/usr/bin/env python3
"""
Section 2: data ingestion (spot + options).

Loads NIFTY 50 1-minute spot bars (2025 in-sample, 2026 Jan-Apr out-of-
sample) and the weekly options-chain archives, caching everything to
parquet under checkpoints/. See the top-level README for the exact data
layout this expects under data/.

Depends on: nothing (first script in the pipeline).
Produces (checkpoints/): nifty_spot_1min_2025.parquet,
  nifty_spot_1min_2026_jan_apr_oos.parquet, options_oi_full_year_2025.parquet,
  options_oi_oos_2026.parquet, options_oi_h1window_2025.parquet
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import (
    SPOT_2025_JSON_GLOB, SPOT_2026_JSON_GLOB, OI_WEEKS_2025_GLOB, OI_WEEKS_2026_GLOB,
    ckpt_path,
)
from lib.io_data import load_spot_from_json, load_options_period, parquet_row_count


def main():
    # In-sample: full calendar year 2025 (249 trading days expected)
    spot_2025 = load_spot_from_json(SPOT_2025_JSON_GLOB, "nifty_spot_1min_2025.parquet",
                                     expected_days_note="(expect 249 for full 2025)")

    # Out-of-sample: Jan 6 - Apr 21 2026 (73 usable trading days within options coverage)
    spot_2026 = load_spot_from_json(SPOT_2026_JSON_GLOB, "nifty_spot_1min_2026_jan_apr_oos.parquet",
                                     expected_days_note="(expect ~73-75 for Jan-Apr 2026)")

    # In-sample: 55 weekly expiries covering all of 2025 (~35M strike-minute rows)
    load_options_period(OI_WEEKS_2025_GLOB, "options_oi_full_year_2025.parquet")

    # Out-of-sample: 16 weekly expiries, 2026-01-06 .. 2026-04-21 (~14.8M rows)
    load_options_period(OI_WEEKS_2026_GLOB, "options_oi_oos_2026.parquet")

    # H3's in-sample fitting window is the Jan-Jul 2025 subset (options data available at
    # the time the signal's parameters were locked, before Q3/Q4 2025 data had been
    # collected -- see paper Sec 4.5 / Table 1a). Derived via pyarrow filter pushdown
    # directly from disk -- never materializes the full 35M-row frame in memory.
    h1_window_ckpt = ckpt_path("options_oi_h1window_2025.parquet")
    if os.path.exists(h1_window_ckpt):
        print(f"[cache hit] {h1_window_ckpt} ({parquet_row_count(h1_window_ckpt):,} rows)")
    else:
        h1_expiry_cutoff = pd.Timestamp("2025-07-10").date()
        all_expiries = pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"),
                                        columns=["expiry"])["expiry"].unique()
        h1_expiries = sorted(e for e in all_expiries if e <= h1_expiry_cutoff)
        options_h1window = pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"),
                                            filters=[("expiry", "in", h1_expiries)])
        options_h1window.to_parquet(h1_window_ckpt, index=False)
        print(f"H1-window (Jan-Jul 2025) options subset: {len(options_h1window)} rows, "
              f"{options_h1window['expiry'].nunique()} expiries")
        del options_h1window

    print("\nSection 2 done.")


if __name__ == "__main__":
    main()
