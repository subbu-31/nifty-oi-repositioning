"""
Section 5 core builder: for every structural break event, aggregate options
open interest at strikes within `band` of the underlying, separately for
calls (CE) and puts (PE), measured immediately before the break and
`after_min` minutes after.

Uses a memory-safe, per-expiry batch pattern: the naive "load the whole
35M-row options table, build one big `groups` dict" approach OOM-kills on a
full year of data. Processing one active-expiry at a time via pyarrow filter
pushdown keeps each batch to a few hundred thousand rows. Reused as-is by
the Section 19 vol-adaptive-threshold robustness check on a different event
set.
"""
import os

import pandas as pd

from lib.config import ckpt_path, STRIKE_BAND, H1_AFTER_WINDOW_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import asof_oi_from_groups


def build_h1_oi_shift(events_df, options_path, band=STRIKE_BAND, after_min=H1_AFTER_WINDOW_MIN):
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
            spot = row["close"]
            strikes = strikes_near(spot, band)
            before_t, after_t = row["date"], row["date"] + pd.Timedelta(minutes=after_min)
            ce_before = pe_before = ce_after = pe_after = 0.0
            n_series = 0
            for k in strikes:
                for opt in ("CE", "PE"):
                    b = asof_oi_from_groups(groups, k, opt, before_t)
                    a = asof_oi_from_groups(groups, k, opt, after_t)
                    if (k, opt) in groups:
                        n_series += 1
                    if opt == "CE":
                        ce_before += b; ce_after += a
                    else:
                        pe_before += b; pe_after += a
            total_before, total_after = ce_before + pe_before, ce_after + pe_after
            results.append({
                **{k2: v2 for k2, v2 in row.to_dict().items()
                   if k2 not in ("trading_date_computed", "active_expiry")},
                "expiry": expiry, "spot_at_break": spot, "n_strikes": len(strikes),
                "n_series_found": n_series,
                "ce_oi_before": ce_before, "ce_oi_after": ce_after, "ce_oi_shift": ce_after - ce_before,
                "pe_oi_before": pe_before, "pe_oi_after": pe_after, "pe_oi_shift": pe_after - pe_before,
                "total_oi_before": total_before, "total_oi_after": total_after,
                "total_oi_shift": total_after - total_before,
            })
        del groups
        print(f"  expiry {expiry}: {len(ev_batch)} events done", flush=True)
    return pd.DataFrame(results)


def get_h1_oi_shift(tag, events_df, options_path):
    ckpt = ckpt_path(f"oi_shift_fullyear_{tag}.parquet")
    if os.path.exists(ckpt):
        return pd.read_parquet(ckpt)
    out = build_h1_oi_shift(events_df, options_path)
    out.to_parquet(ckpt, index=False)
    print(f"{tag}: {len(out)} events processed, saved to {ckpt}")
    return out
