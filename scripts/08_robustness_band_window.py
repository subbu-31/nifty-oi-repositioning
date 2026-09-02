#!/usr/bin/env python3
"""
Section 8: Robustness -- strike-band & OI-window sensitivity
(paper §5.4-5.5, Tables 6-7).

Reports only the PE-CE differential point-biserial r/p per configuration
(the one H1 feature construction that carries the signal), at the primary
0.30% threshold, to keep the sweep cheap. Same memory-safe per-expiry
pattern as Section 5.

Depends on: 04_event_extraction.py
Produces (checkpoints/): robustness_band.csv, robustness_window.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, STRIKE_BAND, H1_AFTER_WINDOW_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import asof_oi_from_groups


def run_h1_config(events_df, options_path, band, after_min):
    expiries = sorted(pd.read_parquet(options_path, columns=["expiry"])["expiry"].unique())

    def active_expiry_for(trading_date):
        for e in expiries:
            if e >= trading_date:
                return e
        return None

    ev = events_df.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev["trading_date_computed"] = ev["date"].dt.date
    ev["active_expiry"] = ev["trading_date_computed"].apply(active_expiry_for)

    results = []
    for expiry, ev_batch in ev.groupby("active_expiry", sort=False):
        if expiry is None:
            continue
        batch = pd.read_parquet(options_path, filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
        del batch
        for _, row in ev_batch.iterrows():
            strikes = strikes_near(row["close"], band)
            before_t, after_t = row["date"], row["date"] + pd.Timedelta(minutes=after_min)
            ce_before = pe_before = ce_after = pe_after = 0.0
            for k in strikes:
                for opt in ("CE", "PE"):
                    b = asof_oi_from_groups(groups, k, opt, before_t)
                    a = asof_oi_from_groups(groups, k, opt, after_t)
                    if opt == "CE":
                        ce_before += b; ce_after += a
                    else:
                        pe_before += b; pe_after += a
            results.append({"direction": row["direction"],
                             "pe_minus_ce_shift": (pe_after - pe_before) - (ce_after - ce_before)})
        del groups

    df = pd.DataFrame(results)
    df["dir_bin"] = (df["direction"] == "bullish").astype(int)
    r, p = stats.pointbiserialr(df["dir_bin"], df["pe_minus_ce_shift"])
    return len(df), r, p


def main():
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    opts_fullyear_path = ckpt_path("options_oi_full_year_2025.parquet")

    band_ckpt = ckpt_path("robustness_band.csv")
    if os.path.exists(band_ckpt):
        robustness_band = pd.read_csv(band_ckpt)
    else:
        rows = []
        for band in [100, 150, 200, 300]:
            n, r, p = run_h1_config(events_fullyear_0p3, opts_fullyear_path, band, H1_AFTER_WINDOW_MIN)
            rows.append({"band": band, "n": n, "r": r, "p": p})
            print(f"  band=+/-{band}: n={n}, r={r:.4f}, p={p:.3e}", flush=True)
        robustness_band = pd.DataFrame(rows)
        robustness_band.to_csv(band_ckpt, index=False)

    window_ckpt = ckpt_path("robustness_window.csv")
    if os.path.exists(window_ckpt):
        robustness_window = pd.read_csv(window_ckpt)
    else:
        rows = []
        for win in [1, 2, 5, 10, 15]:
            n, r, p = run_h1_config(events_fullyear_0p3, opts_fullyear_path, STRIKE_BAND, win)
            rows.append({"window_min": win, "n": n, "r": r, "p": p})
            print(f"  window={win}min: n={n}, r={r:.4f}, p={p:.3e}", flush=True)
        robustness_window = pd.DataFrame(rows)
        robustness_window.to_csv(window_ckpt, index=False)

    print("\nTable 6 -- strike-band sensitivity:")
    print(robustness_band.to_string(index=False))
    print("\nTable 7 -- OI-window sensitivity:")
    print(robustness_window.to_string(index=False))

    print("\nSection 8 done.")


if __name__ == "__main__":
    main()
