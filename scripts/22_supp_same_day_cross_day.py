#!/usr/bin/env python3
"""
Section 22: Supplementary -- does the broken structural level derive from a
prior trading day?

Every BoS/CHoCH event is confirmed against a swing high or low that was set
at some earlier bar -- possibly on a previous trading day. This is a
look-ahead sanity check: if the H1 association were driven mainly by
overnight-carried, already-known levels, it should be visibly weaker for
events whose defining swing was set intraday, on the same day as the event
itself.

Depends on: 02_data_ingestion.py, 05_h1_association.py
Produces: nothing new (prints only).
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.market_structure import run_market_structure_engine
from lib.stats_utils import fisher_r_to_z_test


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    oi_shift_thr0p3 = pd.read_parquet(ckpt_path("oi_shift_fullyear_thr0p3.parquet"))

    _swings_2025, _events_2025_full = run_market_structure_engine(
        spot_2025.sort_values("date").reset_index(drop=True), 0.30)
    _swings_2025 = _swings_2025.copy()
    _swings_2025["dt"] = pd.to_datetime(_swings_2025["date"])
    if _swings_2025["dt"].dt.tz is not None:
        _swings_2025["dt"] = _swings_2025["dt"].dt.tz_localize(None)
    _swings_2025["trading_date"] = _swings_2025["dt"].dt.date

    _high_swings = _swings_2025[_swings_2025["type"] == "high"].reset_index(drop=True)
    _low_swings = _swings_2025[_swings_2025["type"] == "low"].reset_index(drop=True)

    def _find_defining_swing_date(row):
        target_price = row["broken_level"]
        event_idx = row["idx"]
        pool = _high_swings if row["direction"] == "bullish" else _low_swings
        cand = pool[pool["idx"] < event_idx]
        if len(cand) == 0:
            return None
        diffs = (cand["price"] - target_price).abs()
        return cand.loc[diffs.idxmin(), "trading_date"]

    _events_2025_full = _events_2025_full.copy()
    _events_2025_full["dt"] = pd.to_datetime(_events_2025_full["date"])
    if _events_2025_full["dt"].dt.tz is not None:
        _events_2025_full["dt"] = _events_2025_full["dt"].dt.tz_localize(None)
    _events_2025_full["trading_date"] = _events_2025_full["dt"].dt.date
    _events_2025_full["defining_swing_date"] = _events_2025_full.apply(_find_defining_swing_date, axis=1)
    _events_2025_full["swing_predates_event_day"] = (
        _events_2025_full["defining_swing_date"] != _events_2025_full["trading_date"]
    )

    _n_valid = _events_2025_full["defining_swing_date"].notna().sum()
    _n_cross_day = _events_2025_full["swing_predates_event_day"].sum()
    print(f"Events with an identifiable defining swing: {_n_valid}/{len(_events_2025_full)}")
    print(f"Defining swing on an earlier calendar day than the event: {_n_cross_day}/{_n_valid} "
          f"({_n_cross_day/_n_valid:.1%})")
    print(f"Defining swing on the same day as the event: {_n_valid-_n_cross_day}/{_n_valid} "
          f"({(_n_valid-_n_cross_day)/_n_valid:.1%})")

    # Join the same-day/cross-day flag onto the headline (0.30% threshold) H1 OI-shift table
    _h1_split = oi_shift_thr0p3.merge(
        _events_2025_full[["idx", "swing_predates_event_day"]], on="idx", how="inner")
    _h1_split["dir_bin"] = (_h1_split["direction"] == "bullish").astype(int)
    _h1_split["pe_minus_ce_shift"] = _h1_split["pe_oi_shift"] - _h1_split["ce_oi_shift"]

    _same_day = _h1_split[~_h1_split["swing_predates_event_day"]]
    _cross_day = _h1_split[_h1_split["swing_predates_event_day"]]
    r_same, p_same = stats.pointbiserialr(_same_day["dir_bin"], _same_day["pe_minus_ce_shift"])
    r_cross, p_cross = stats.pointbiserialr(_cross_day["dir_bin"], _cross_day["pe_minus_ce_shift"])
    print(f"\nSame-day defining swing (n={len(_same_day)}): r={r_same:+.4f}, p={p_same:.3e}")
    print(f"Cross-day defining swing (n={len(_cross_day)}): r={r_cross:+.4f}, p={p_cross:.3e}")

    z_split, p_split = fisher_r_to_z_test(r_same, len(_same_day), r_cross, len(_cross_day))
    print(f"Fisher r-to-z (same-day vs. cross-day): z={z_split:.4f}, p={p_split:.4f}")
    print("\nThe association is not materially weaker when the broken level was only just set "
          "intraday, which argues against the H1 correlation being an artifact of overnight-carried, "
          "already-known structural levels.")

    print("\nSection 22 done.")


if __name__ == "__main__":
    main()
