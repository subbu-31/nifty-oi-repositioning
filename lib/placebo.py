"""
Shared loader for the Section 21 placebo pool.

The checkpoint (placebo_pool_thr0p3.parquet) caches the pool BEFORE the
exactly-zero-preceding-move filter, so the cheap filter step can be redone
without recomputing the expensive per-event OI lookups. Every section that
consumes the pool (21, 25, 26, 27) uses the filtered version, so they all
go through this one function to stay consistent.
"""
import pandas as pd

from lib.config import ckpt_path


def load_filtered_placebo_pool():
    pool_result = pd.read_parquet(ckpt_path("placebo_pool_thr0p3.parquet"))
    n_before = len(pool_result)
    pool_result = pool_result[pool_result["preceding_chg"] != 0].reset_index(drop=True)
    print(f"Placebo pool: {len(pool_result)} candidates (excluded {n_before - len(pool_result)} "
          f"with an exactly-zero preceding 5-minute move)")
    # dir_bin is a deterministic derived column (Section 21 adds it in-kernel); every
    # consumer of the filtered pool needs it, so it's added once here.
    pool_result["dir_bin"] = (pool_result["pseudo_direction"] == "bullish").astype(int)
    return pool_result
