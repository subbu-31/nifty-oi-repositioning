#!/usr/bin/env python3
"""
Section 27: Supplementary -- placebo pool's own OI zero-fill rate.

Section 12 checked the zero-fill rate on the real-event side. This is the
same diagnostic run on the placebo pool's strike-leg OI lookups, as a
supplementary check that the placebo comparison is not being distorted by
a systematically different zero-fill rate than the real events. This
section is not quoted directly in the paper text; it is included here for
completeness of the public codebase.

Depends on: 21_placebo_test.py (placebo_pool_thr0p3.parquet)
Produces: nothing new (prints only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.market_structure import strikes_near
from lib.placebo import load_filtered_placebo_pool


def main():
    pool_result = load_filtered_placebo_pool()

    _zf_total_lookups = 0
    _zf_zero_filled = 0
    _zf_zero_traded_later = 0

    _pool_zf = pool_result.copy()
    _pool_zf["trading_date"] = pd.to_datetime(_pool_zf["date_naive"]).dt.date
    _expiries_zf = sorted(pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"), columns=["expiry"])["expiry"].unique())

    def _active_expiry_zf(trading_date):
        for e in _expiries_zf:
            if e >= trading_date:
                return e
        return None
    _pool_zf["active_expiry"] = _pool_zf["trading_date"].apply(_active_expiry_zf)

    for expiry, batch_ev in _pool_zf.groupby("active_expiry", sort=False):
        if expiry is None:
            continue
        batch = pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"), filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
        del batch
        for _, row in batch_ev.iterrows():
            strikes = strikes_near(row["close"])
            before_t = row["date_naive"]
            for k in strikes:
                for opt in ("CE", "PE"):
                    key = (k, opt)
                    _zf_total_lookups += 1
                    if key not in groups:
                        _zf_zero_filled += 1
                        continue
                    tarr, oarr = groups[key]
                    idx = np.searchsorted(tarr, np.datetime64(before_t), side="right") - 1
                    if idx < 0:
                        _zf_zero_filled += 1
                        if len(tarr) > 0:
                            _zf_zero_traded_later += 1
        del groups

    print(f"Placebo pool: total strike-leg lookups (before-timestamp only): {_zf_total_lookups}")
    print(f"Zero-filled (no trade at/before lookup time): {_zf_zero_filled} ({_zf_zero_filled/_zf_total_lookups:.4%})")
    print(f"Of those, traded LATER in the same archive week: {_zf_zero_traded_later} "
          f"({_zf_zero_traded_later/max(_zf_zero_filled,1):.2%} of zero-fills)")

    print("\nSection 27 done.")


if __name__ == "__main__":
    main()
