#!/usr/bin/env python3
"""
Section 3: Market Structure Engine (BoS/CHoCH detector) -- threshold
selection sanity check.

Runs the engine across a range of candidate thresholds on the full 2025
spot series and prints events-per-day, to show how the 0.25%/0.30% study
thresholds (locked in lib/config.py) were chosen. The engine itself
(lib.market_structure.run_market_structure_engine) is exercised for real by
Section 4's event extraction; this script is a diagnostic, not a
checkpoint-producing step.

Depends on: 02_data_ingestion.py (nifty_spot_1min_2025.parquet)
Produces: nothing new (diagnostic only).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.market_structure import run_market_structure_engine


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    n_days_2025 = spot_2025["date"].dt.date.nunique()

    print(f"{'threshold%':>10} | {'swings':>7} | {'BoS':>6} | {'CHoCH':>6} | {'events/day':>10}")
    for thr_pct in [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        swings_df, ev = run_market_structure_engine(spot_2025.sort_values("date").reset_index(drop=True), thr_pct)
        n_bos = (ev["event_type"] == "BoS").sum() if len(ev) else 0
        n_choch = (ev["event_type"] == "CHoCH").sum() if len(ev) else 0
        print(f"{thr_pct:>10} | {len(swings_df):>7} | {n_bos:>6} | {n_choch:>6} | "
              f"{len(ev)/n_days_2025:>10.2f}")

    print("\nSection 3 done.")


if __name__ == "__main__":
    main()
