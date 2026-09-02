#!/usr/bin/env python3
"""
Section 12: Data-quality diagnostic -- does the OI zero-fill convention
mask real open interest?

The as-of OI convention assigns 0 to a strike/leg that hasn't traded yet
within the active contract, rather than treating it as missing. This could
in principle understate OI for a strike that carries open interest but
simply hasn't retraded yet in the current archive. This check quantifies,
for every (strike, option_type) leg lookup in the H1 pre-break snapshots,
whether it (a) had a prior trade, (b) never traded at all that week
(genuinely negligible OI, most likely), or (c) -- the scenario that would
actually be concerning -- traded later in the week but not yet by the
lookup timestamp.

Depends on: 04_event_extraction.py
Produces (checkpoints/): oi_zero_diagnostic.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.market_structure import strikes_near
from lib.oi_lookup import classify_oi_lookup


def oi_zero_diagnostic(events_df, options_path, tag):
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

    counts = {"has_prior_trade": 0, "traded_later_this_week_not_yet": 0, "never_traded_in_archive": 0}
    total = 0
    for expiry, ev_batch in ev.groupby("active_expiry", sort=False):
        if expiry is None:
            continue
        batch = pd.read_parquet(options_path, filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
        del batch
        for _, row in ev_batch.iterrows():
            for k in strikes_near(row["close"]):
                for opt in ("CE", "PE"):
                    counts[classify_oi_lookup(groups, k, opt, row["date"])] += 1
                    total += 1
        del groups

    print(f"===== {tag}: {total} strike-leg lookups =====")
    for k, v in counts.items():
        print(f"  {k}: {v} ({100*v/total:.2f}%)")
    return {"tag": tag, "total_legs": total, **counts}


def main():
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    events_fullyear_0p25 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p25.parquet"))
    opts_fullyear_path = ckpt_path("options_oi_full_year_2025.parquet")

    oi_zero_diagnostic_result = pd.DataFrame([
        oi_zero_diagnostic(events_fullyear_0p3, opts_fullyear_path, "0.30% threshold"),
        oi_zero_diagnostic(events_fullyear_0p25, opts_fullyear_path, "0.25% threshold"),
    ])
    oi_zero_diagnostic_result.to_csv(ckpt_path("oi_zero_diagnostic.csv"), index=False)
    print(oi_zero_diagnostic_result.to_string(index=False))

    print("\nSection 12 done.")


if __name__ == "__main__":
    main()
