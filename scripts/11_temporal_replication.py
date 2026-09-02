#!/usr/bin/env python3
"""
Section 11: Temporal replication of earlier project analysis
(paper §4.4, Table 5).

Earlier in this project, before the fully specified pipeline reproduced
here existed, an earlier set of analysis scripts reported a null result on
a non-overlapping Oct 2025-Apr 2026 dataset. Those two specific tests are
reimplemented here and rerun against this project's own Jan-Jul 2025 data,
which they had not previously used -- a temporal replication across two
different data periods within this same project.

Test A -- continuation-vs-fade: after a confirmed break, does price keep
moving in the break's direction over 5-30 minute horizons (continuation),
or give it back (fade)?

Test B -- OI-reaction-magnitude tercile filter: does the size of the
post-break OI reaction at the single ATM strike predict how far forward
returns run, direction-aware?

Depends on: 02_data_ingestion.py, 04_event_extraction.py
Produces (checkpoints/): continuation_vs_fade_replication.csv,
  reaction_tercile_replication.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.oi_lookup import asof_oi_expiry

HORIZONS = [5, 10, 15, 20, 30]
STRIKE_STEP = 50


def build_fwd_returns(spot_df):
    spot = spot_df.sort_values("date").reset_index(drop=True).copy()
    spot["date_naive"] = spot["date"].dt.tz_localize(None)
    spot["trading_date"] = spot["date_naive"].dt.date
    for h in HORIZONS:
        spot[f"fwd_ret_{h}"] = spot.groupby("trading_date")["close"].transform(lambda s: s.shift(-h) / s - 1)
    spot_ts = spot["date_naive"].to_numpy()

    def fwd_returns_at(times_naive, h):
        idx = np.searchsorted(spot_ts, times_naive, side="right") - 1
        col = spot[f"fwd_ret_{h}"].to_numpy()
        return col[np.clip(idx, 0, len(col) - 1)]
    return fwd_returns_at


def continuation_vs_fade(events_df, tag, fwd_returns_at_2025):
    rows = []
    ev = events_df.copy()
    ev["date_naive"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    for event_type in ["BoS", "CHoCH"]:
        for direction in ["bullish", "bearish"]:
            sub = ev[(ev["event_type"] == event_type) & (ev["direction"] == direction)]
            if len(sub) < 10:
                continue
            sign = 1 if direction == "bullish" else -1
            times = sub["date_naive"].to_numpy()
            for h in HORIZONS:
                vals = fwd_returns_at_2025(times, h) * sign
                vals = vals[~np.isnan(vals)]
                if len(vals) < 10:
                    continue
                t, p = stats.ttest_1samp(vals, 0)
                rows.append({"threshold": tag, "event": f"{event_type.lower()}_{direction}", "horizon_min": h,
                             "n": len(vals), "avg_signed_return_pct": float(np.mean(vals) * 100),
                             "win_rate_continuation": float((vals > 0).mean()), "t_stat": float(t), "p_value": float(p)})
    return pd.DataFrame(rows)


def reaction_tercile_test(events_df, options_h1_path, tag, fwd_returns_at_2025):
    oi = pd.read_parquet(options_h1_path).sort_values("timestamp")
    groups = {}
    for key, g in oi.groupby(["expiry", "strike", "option_type"], sort=False):
        groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
    expiries = sorted(oi["expiry"].unique())

    def active_expiry_for(trading_date):
        for e in expiries:
            if e >= trading_date:
                return e
        return None

    ev = events_df.copy()
    ev["date_naive"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    rows = []
    for _, row in ev.iterrows():
        trading_date = row["date_naive"].date()
        expiry = active_expiry_for(trading_date)
        if expiry is None:
            continue
        atm = round(row["close"] / STRIKE_STEP) * STRIKE_STEP
        opt_type = "CE" if row["direction"] == "bearish" else "PE"
        before_t, after_t = row["date_naive"], row["date_naive"] + pd.Timedelta(minutes=5)
        b = asof_oi_expiry(groups, expiry, int(atm), opt_type, before_t)
        a = asof_oi_expiry(groups, expiry, int(atm), opt_type, after_t)
        rows.append({"date_naive": row["date_naive"], "direction": row["direction"], "atm_oi_chg": a - b})
    reactions = pd.DataFrame(rows)
    if len(reactions) < 30:
        return pd.DataFrame(), reactions
    reactions["tercile"] = pd.qcut(reactions["atm_oi_chg"], 3, labels=["weak", "medium", "strong"], duplicates="drop")

    out_rows = []
    for h in HORIZONS:
        sign = np.where(reactions["direction"] == "bullish", 1, -1)
        rets = fwd_returns_at_2025(reactions["date_naive"].to_numpy(), h) * sign
        reactions[f"signed_ret_{h}"] = rets
        for tier in ["weak", "medium", "strong"]:
            sel = reactions.loc[reactions["tercile"] == tier, f"signed_ret_{h}"].dropna()
            if len(sel) < 5:
                continue
            out_rows.append({"threshold": tag, "horizon_min": h, "reaction_tercile": tier, "n": len(sel),
                              "avg_signed_return_pct": float(sel.mean() * 100),
                              "win_rate_continuation": float((sel > 0).mean())})
    return pd.DataFrame(out_rows), reactions


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    events_h1window_0p3 = pd.read_parquet(ckpt_path("events_h1window_thr0p3.parquet"))
    events_h1window_0p25 = pd.read_parquet(ckpt_path("events_h1window_thr0p25.parquet"))

    fwd_returns_at_2025 = build_fwd_returns(spot_2025)

    test_a = pd.concat([
        continuation_vs_fade(events_h1window_0p3, "0.30%", fwd_returns_at_2025),
        continuation_vs_fade(events_h1window_0p25, "0.25%", fwd_returns_at_2025),
    ], ignore_index=True)
    test_a.to_csv(ckpt_path("continuation_vs_fade_replication.csv"), index=False)
    n_a = len(test_a)
    alpha_a = 0.05 / n_a
    print(f"Test A: {n_a} tests, Bonferroni alpha={alpha_a:.5f}, survivors={(test_a['p_value'] < alpha_a).sum()}")

    h1window_options_path = ckpt_path("options_oi_h1window_2025.parquet")
    test_b_030, react_030 = reaction_tercile_test(events_h1window_0p3, h1window_options_path, "0.30%", fwd_returns_at_2025)
    test_b_025, react_025 = reaction_tercile_test(events_h1window_0p25, h1window_options_path, "0.25%", fwd_returns_at_2025)
    test_b = pd.concat([test_b_030, test_b_025], ignore_index=True)
    test_b.to_csv(ckpt_path("reaction_tercile_replication.csv"), index=False)

    sig_rows = []
    for tag, react in [("0.30%", react_030), ("0.25%", react_025)]:
        for h in HORIZONS:
            col = f"signed_ret_{h}"
            strong, weak = react.loc[react["tercile"] == "strong", col].dropna(), react.loc[react["tercile"] == "weak", col].dropna()
            t, p = stats.ttest_ind(strong, weak, equal_var=False)
            t1, p1 = stats.ttest_1samp(strong, 0)
            sig_rows.append({"threshold": tag, "horizon_min": h, "strong_vs_weak_p": p, "strong_vs_zero_p": p1})
    sig_df = pd.DataFrame(sig_rows)
    n_sig = len(sig_df) * 2
    alpha_b = 0.05 / n_sig
    print(f"\nTest B: Bonferroni alpha across {n_sig} tests = {alpha_b:.5f}")
    print(f"  strong-vs-weak survivors: {(sig_df['strong_vs_weak_p'] < alpha_b).sum()} / {len(sig_df)}")
    print(f"  strong-vs-zero survivors: {(sig_df['strong_vs_zero_p'] < alpha_b).sum()} / {len(sig_df)}")

    print("\nSection 11 done.")


if __name__ == "__main__":
    main()
