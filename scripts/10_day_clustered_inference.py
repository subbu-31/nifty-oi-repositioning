#!/usr/bin/env python3
"""
Section 10: Day-clustered inference & block bootstrap (paper §5.7, Table 4).

Minute-level trades within the H3 backtest are not independent -- the
15-minute holding period means adjacent-minute signals overlap, and all
trades on a given day share that day's regime. A plain one-sample t-test
across pooled trades understates the true uncertainty. This aggregates P&L
to one observation per trading day and runs both a day-clustered t-test and
a block bootstrap (resample days with replacement) for a distribution-free
95% CI, for both the IS and OOS backtests.

Depends on: 07_h3_signal.py (h3_is_trades.csv, h3_oos_trades.csv)
Produces (checkpoints/): h3_daylevel_inference.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.stats_utils import day_level_inference


def main():
    is_trades = pd.read_csv(ckpt_path("h3_is_trades.csv"))
    oos_trades = pd.read_csv(ckpt_path("h3_oos_trades.csv"))

    h3_daylevel_inference = pd.DataFrame([
        day_level_inference(is_trades, "IN-SAMPLE (Jan-Jul 2025)"),
        day_level_inference(oos_trades, "OUT-OF-SAMPLE (Jan-Apr 2026)"),
    ])
    h3_daylevel_inference.to_csv(ckpt_path("h3_daylevel_inference.csv"), index=False)
    print(h3_daylevel_inference.to_string(index=False))

    print("\nSection 10 done.")


if __name__ == "__main__":
    main()
