"""
Section 4 of the original notebook: structural break event extraction.

Extracts events at both study thresholds (0.25% and 0.30%) for whatever
spot period/date-mask you pass in, caching each (period, threshold) pair to
its own parquet checkpoint.
"""
import os

import pandas as pd

from lib.config import ckpt_path
from lib.market_structure import run_market_structure_engine


def extract_events(spot_df, thr_list, out_prefix, date_mask=None):
    spot_sorted = spot_df.sort_values("date").reset_index(drop=True)
    if date_mask is not None:
        spot_sorted = spot_sorted[date_mask(spot_sorted)].reset_index(drop=True)
    print(f"[{out_prefix}] spot bars: {len(spot_sorted)}, days: {spot_sorted['date'].dt.date.nunique()}")

    out = {}
    for thr in thr_list:
        tag = str(thr).replace(".", "p")
        ckpt = ckpt_path(f"events_{out_prefix}_thr{tag}.parquet")
        if os.path.exists(ckpt):
            events_df = pd.read_parquet(ckpt)
        else:
            _, events_df = run_market_structure_engine(spot_sorted, thr)
            events_df["date"] = pd.to_datetime(events_df["date"])
            events_df["trading_date"] = events_df["date"].dt.date
            events_df["hour"] = events_df["date"].dt.hour
            events_df.to_parquet(ckpt, index=False)
        n_bos = (events_df["event_type"] == "BoS").sum()
        n_choch = (events_df["event_type"] == "CHoCH").sum()
        print(f"  thr={thr}%: {len(events_df)} events (BoS={n_bos}, CHoCH={n_choch}), "
              f"bullish={(events_df['direction']=='bullish').sum()}, "
              f"bearish={(events_df['direction']=='bearish').sum()}")
        out[tag] = events_df
    return out
