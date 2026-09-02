#!/usr/bin/env python3
"""
Section 17: Extension -- put-call VOLUME ratio (Pan & Poteshman 2006).

Pan & Poteshman (2006) is cited in this paper's own literature review, but
its actual construct -- put-call trading VOLUME ratio -- has never been
tested anywhere in this pipeline, which has used OI exclusively. Unlike
`oi` (a cumulative running total, safe to point-sample as-of a timestamp),
`volume` in the raw data is a per-bar flow quantity (it jumps up and down
bar to bar rather than accumulating), so it must be summed over a time
window rather than snapshotted. Windows here are symmetric around the
break: [break-5min, break) vs [break, break+5min), matching the existing
5-minute convention.

This is a reactive (post-break), not predictive (pre-break), test of
volume -- consistent with the paper's own reactive-not-predictive framing
throughout -- so it is not a like-for-like replication of Pan & Poteshman's
original informed-trading claim, just the same construct applied to this
paper's own event set.

Depends on: 04_event_extraction.py
Produces (checkpoints/): pcr_volume_thr0p3.parquet, pcr_volume_thr0p25.parquet
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, H1_AFTER_WINDOW_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import windowed_volume_sum

PCR_FEATURES = ["ce_vol_shift", "pe_vol_shift", "pe_minus_ce_vol_shift", "log_pcr_shift"]


def build_pcr_volume(events_df, options_path, tag):
    ckpt = ckpt_path(f"pcr_volume_{tag}.parquet")
    if os.path.exists(ckpt):
        return pd.read_parquet(ckpt)

    expiries_local = sorted(pd.read_parquet(options_path, columns=["expiry"])["expiry"].unique())
    def active_expiry_for(trading_date):
        for e in expiries_local:
            if e >= trading_date:
                return e
        return None

    ev = events_df.copy()
    ev["date_naive"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    ev["trading_date_computed"] = ev["date_naive"].dt.date
    ev["active_expiry"] = ev["trading_date_computed"].apply(active_expiry_for)

    results = []
    for expiry, ev_batch in ev.groupby("active_expiry", sort=False):
        if expiry is None:
            continue
        batch = pd.read_parquet(options_path, filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["volume"].to_numpy())
        del batch
        for _, row in ev_batch.iterrows():
            strikes = strikes_near(row["close"])
            break_t = row["date_naive"]
            before_t0, before_t1 = break_t - pd.Timedelta(minutes=H1_AFTER_WINDOW_MIN), break_t
            after_t0, after_t1 = break_t, break_t + pd.Timedelta(minutes=H1_AFTER_WINDOW_MIN)
            ce_vb = pe_vb = ce_va = pe_va = 0.0
            for k in strikes:
                ce_vb += windowed_volume_sum(groups, k, "CE", before_t0, before_t1)
                ce_va += windowed_volume_sum(groups, k, "CE", after_t0, after_t1)
                pe_vb += windowed_volume_sum(groups, k, "PE", before_t0, before_t1)
                pe_va += windowed_volume_sum(groups, k, "PE", after_t0, after_t1)
            results.append({
                **{k2: v2 for k2, v2 in row.to_dict().items() if k2 not in ("trading_date_computed", "active_expiry")},
                "ce_vol_shift": ce_va - ce_vb, "pe_vol_shift": pe_va - pe_vb,
                "pe_minus_ce_vol_shift": (pe_va - pe_vb) - (ce_va - ce_vb),
                "log_pcr_shift": (np.log(pe_va + 1) - np.log(ce_va + 1)) - (np.log(pe_vb + 1) - np.log(ce_vb + 1)),
            })
        del groups
    out = pd.DataFrame(results)
    out.to_parquet(ckpt, index=False)
    return out


def run_pcr_battery(df, tag):
    df = df.copy()
    df["dir_bin"] = (df["direction"] == "bullish").astype(int)
    rows = []
    for feat in PCR_FEATURES:
        x = df[feat].to_numpy()
        bull, bear = df.loc[df["direction"] == "bullish", feat].to_numpy(), df.loc[df["direction"] == "bearish", feat].to_numpy()
        r, p_r = stats.pointbiserialr(df["dir_bin"], x)
        rows.append({"threshold": tag, "feature": feat, "test": "point-biserial r (vs direction)", "n": len(x), "statistic": r, "p_value": p_r})
        t_w, p_w = stats.ttest_ind(bull, bear, equal_var=False)
        rows.append({"threshold": tag, "feature": feat, "test": "Welch t (bullish vs bearish)", "n": len(x), "statistic": t_w, "p_value": p_w})
        t_1, p_1 = stats.ttest_1samp(x, 0)
        rows.append({"threshold": tag, "feature": feat, "test": "one-sample t (vs 0)", "n": len(x), "statistic": t_1, "p_value": p_1})
    return pd.DataFrame(rows)


def main():
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    events_fullyear_0p25 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p25.parquet"))
    opts_fullyear_path = ckpt_path("options_oi_full_year_2025.parquet")

    pcr_thr0p3 = build_pcr_volume(events_fullyear_0p3, opts_fullyear_path, "thr0p3")
    pcr_thr0p25 = build_pcr_volume(events_fullyear_0p25, opts_fullyear_path, "thr0p25")

    pcr_battery = pd.concat([run_pcr_battery(pcr_thr0p3, "0.30%"), run_pcr_battery(pcr_thr0p25, "0.25%")], ignore_index=True)
    _alpha_pcr = 0.05 / len(pcr_battery)
    pcr_battery["bonferroni_significant"] = pcr_battery["p_value"] < _alpha_pcr
    _pcr_n_sig = pcr_battery["bonferroni_significant"].sum()
    print(f"PCR-volume battery: {len(pcr_battery)} tests, Bonferroni alpha={_alpha_pcr:.5f}, survivors={_pcr_n_sig}")
    print(pcr_battery.to_string(index=False, formatters={"statistic": lambda x: f"{x:+.4f}", "p_value": lambda x: f"{x:.4g}"}))
    print("\nNote the sign: pe_minus_ce_vol_shift is NEGATIVE-correlated with bullish direction "
          "(more CALL volume, relatively, during bullish breaks) -- the opposite sign from the OI-based "
          "pe_minus_ce_OI_shift, which is positive-correlated. Volume (trading activity) and OI "
          "(net positioning) point in different directions here, which is economically sensible: "
          "fresh put writing during a bullish move (positioning, confirmed in Section 16) can "
          "coexist with heavier call-side trading activity (execution, e.g. profit-taking / short "
          "covering generating turnover) without contradiction.")

    print("\nSection 17 done.")


if __name__ == "__main__":
    main()
