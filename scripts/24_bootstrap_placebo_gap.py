#!/usr/bin/env python3
"""
Section 24: Day-clustered block bootstrap on the placebo-test gap.

OI reactions from the same trading day are not independent draws (shared
market regime, shared expiry-week liquidity). This re-estimates the
confidence interval on the real-vs.-magnitude-matched-placebo gap (Section
21) using a block bootstrap that resamples whole trading days rather than
individual events, both for the real/matched point estimates and for the
gap between them.

Depends on: 21_placebo_test.py (real_sorted.parquet, matched.parquet -- built
  together as a 1:1 pairing, so row order must be preserved on load)
Produces: nothing new (prints only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.stats_utils import pointbiserial_fast

N_BOOT_DAY = 10000


def main():
    real_sorted = pd.read_parquet(ckpt_path("real_sorted.parquet"))
    matched = pd.read_parquet(ckpt_path("matched.parquet"))

    _day_rng = np.random.default_rng(42)

    _pair_real = real_sorted.reset_index(drop=True).copy()
    _pair_real["trading_date"] = pd.to_datetime(_pair_real["date"]).dt.date
    _pair_matched = matched.reset_index(drop=True).copy()
    assert len(_pair_real) == len(_pair_matched), "real/matched pairing must be 1:1 (built together in Section 21)"

    real_shift = _pair_real["pe_minus_ce_shift"].to_numpy()
    real_dirbin = _pair_real["naive_dir_bin"].to_numpy(dtype=float)
    matched_shift = _pair_matched["pe_minus_ce_shift"].to_numpy()
    matched_dirbin = _pair_matched["dir_bin"].to_numpy(dtype=float)
    pair_day = _pair_real["trading_date"].to_numpy()
    unique_days = np.unique(pair_day)
    day_to_positions = {d: np.where(pair_day == d)[0] for d in unique_days}
    n_days = len(unique_days)

    boot_gap = np.empty(N_BOOT_DAY)
    boot_r_real = np.empty(N_BOOT_DAY)
    boot_r_matched = np.empty(N_BOOT_DAY)
    for b in range(N_BOOT_DAY):
        sampled_days = _day_rng.choice(unique_days, size=n_days, replace=True)
        idx = np.concatenate([day_to_positions[d] for d in sampled_days])
        rr = pointbiserial_fast(real_dirbin[idx], real_shift[idx])
        rm = pointbiserial_fast(matched_dirbin[idx], matched_shift[idx])
        boot_r_real[b] = rr; boot_r_matched[b] = rm; boot_gap[b] = rr - rm

    print(f"Day-clustered block bootstrap ({N_BOOT_DAY} resamples, {n_days} unique real-event trading days):")
    print(f"real(naive) r: mean={boot_r_real.mean():.4f}, "
          f"95% CI=[{np.percentile(boot_r_real,2.5):.4f}, {np.percentile(boot_r_real,97.5):.4f}]")
    print(f"matched-placebo r: mean={boot_r_matched.mean():.4f}, "
          f"95% CI=[{np.percentile(boot_r_matched,2.5):.4f}, {np.percentile(boot_r_matched,97.5):.4f}]")
    print(f"gap (real - matched): mean={boot_gap.mean():.4f}, "
          f"95% CI=[{np.percentile(boot_gap,2.5):.4f}, {np.percentile(boot_gap,97.5):.4f}]")
    print(f"Fraction of {N_BOOT_DAY} draws with gap <= 0: {(boot_gap <= 0).mean():.4%}")

    print("\nSection 24 done.")


if __name__ == "__main__":
    main()
