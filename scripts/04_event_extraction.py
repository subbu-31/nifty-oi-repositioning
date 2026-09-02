#!/usr/bin/env python3
"""
Section 4: structural break event extraction.

Extracts events at both study thresholds (0.25% and 0.30%) for each period:
the full 2025 in-sample year, the Jan-Jul 2025 subset used for H3 training
and cross-validation, and the Jan-Apr 2026 out-of-sample window (restricted
to the trading days actually covered by the OOS options data).

Depends on: 02_data_ingestion.py
Produces (checkpoints/): events_fullyear2025_thr{0p3,0p25}.parquet,
  events_h1window_thr{0p3,0p25}.parquet, events_oos2026_thr{0p3,0p25}.parquet
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, BREAK_THRESHOLDS_PCT
from lib.events import extract_events


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    spot_2026 = pd.read_parquet(ckpt_path("nifty_spot_1min_2026_jan_apr_oos.parquet"))

    # Full-year 2025 in-sample events (358 @ 0.30%, 490 @ 0.25% expected)
    events_fullyear = extract_events(spot_2025, BREAK_THRESHOLDS_PCT, "fullyear2025")

    # Jan-Jul 2025 subset (H3 training window / cross-validation window)
    events_h1window = extract_events(
        spot_2025, BREAK_THRESHOLDS_PCT, "h1window",
        date_mask=lambda df: (df["date"].dt.tz_localize(None) >= "2025-01-01")
                            & (df["date"].dt.tz_localize(None) <= "2025-07-03 15:30:00"),
    )

    # Out-of-sample 2026 events, restricted to the window the OOS options data covers.
    # Read only the timestamp column (cheap) rather than holding the full ~14.8M-row
    # options table in memory just to find its min/max.
    _opts_2026_ts = pd.read_parquet(ckpt_path("options_oi_oos_2026.parquet"), columns=["timestamp"])
    opts_2026_tmin = _opts_2026_ts["timestamp"].min()
    opts_2026_tmax = _opts_2026_ts["timestamp"].max()
    del _opts_2026_ts
    print(f"OOS options coverage: {opts_2026_tmin} .. {opts_2026_tmax}")
    events_oos = extract_events(
        spot_2026, BREAK_THRESHOLDS_PCT, "oos2026",
        date_mask=lambda df: (df["date"].dt.tz_localize(None) >= opts_2026_tmin)
                            & (df["date"].dt.tz_localize(None) <= opts_2026_tmax),
    )

    print("\nSection 4 done.")
    return events_fullyear, events_h1window, events_oos


if __name__ == "__main__":
    main()
