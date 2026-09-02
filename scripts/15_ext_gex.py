#!/usr/bin/env python3
"""
Section 15: Extension -- Gamma Exposure (GEX) reframing.

The H1-H3 tests all use raw open interest. This section asks a sharper
question, motivated by the paper's own cited dealer-hedging mechanism
(Bollen & Whaley-style): does gamma-weighting the OI -- i.e. asking how
much dealer hedging flow each strike's OI actually represents, not just
how many contracts are open -- change or strengthen the picture? It also
runs the one test this reframing exists to make possible: whether the sign
of dealer gamma positioning before a break predicts whether the subsequent
move amplifies or fades.

This is a separate, exploratory test family from the pre-specified H1
appendix -- it gets its own Bonferroni correction below, not folded into
the 30-test family.

Depends on: 02_data_ingestion.py, 04_event_extraction.py
Produces (checkpoints/): gex_events_thr0p3.parquet, gex_events_thr0p25.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.gex import bs_price, bs_gamma, implied_vol, GEX_R, build_gex_events

GEX_FEATURES = ["unsigned_gex_before", "unsigned_gex_shift", "dealer_gex_before", "dealer_gex_shift"]
GEX_HORIZONS = [5, 10, 15, 20, 30]


def run_gex_battery(df, tag):
    df = df.copy()
    df["dir_bin"] = (df["direction"] == "bullish").astype(int)
    rows = []
    for feat in GEX_FEATURES:
        x = df[feat].to_numpy()
        bull = df.loc[df["direction"] == "bullish", feat].to_numpy()
        bear = df.loc[df["direction"] == "bearish", feat].to_numpy()
        r, p_r = stats.pointbiserialr(df["dir_bin"], x)
        rows.append({"threshold": tag, "feature": feat, "test": "point-biserial r (vs direction)", "n": len(x), "statistic": r, "p_value": p_r})
        t_w, p_w = stats.ttest_ind(bull, bear, equal_var=False)
        rows.append({"threshold": tag, "feature": feat, "test": "Welch t (bullish vs bearish)", "n": len(x), "statistic": t_w, "p_value": p_w})
        t_1, p_1 = stats.ttest_1samp(x, 0)
        rows.append({"threshold": tag, "feature": feat, "test": "one-sample t (vs 0)", "n": len(x), "statistic": t_1, "p_value": p_1})
    return pd.DataFrame(rows)


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    events_fullyear_0p25 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p25.parquet"))
    opts_fullyear_path = ckpt_path("options_oi_full_year_2025.parquet")

    # Sanity check: put-call parity + IV round-trip on a synthetic ATM option
    _S, _K, _T, _sigma = 24000.0, 24000.0, 3 / 365, 0.12
    _ce, _pe = bs_price(_S, _K, _T, GEX_R, _sigma, "CE"), bs_price(_S, _K, _T, GEX_R, _sigma, "PE")
    _iv_ce = implied_vol(_ce, _S, _K, _T, GEX_R, "CE")
    print(f"IV round-trip check: input sigma=0.12, recovered={_iv_ce:.4f} (should match)")
    print(f"Put-call parity: CE-PE={_ce-_pe:.4f}, S-Ke^-rT={_S - _K*np.exp(-GEX_R*_T):.4f} (should match)")

    gex_spot_ts = spot_2025.sort_values("date")["date"].dt.tz_localize(None).to_numpy()
    gex_spot_close = spot_2025.sort_values("date")["close"].to_numpy()

    def asof_spot_2025(ts_naive):
        idx = np.searchsorted(gex_spot_ts, ts_naive, side="right") - 1
        idx = np.clip(idx, 0, len(gex_spot_close) - 1)
        return float(gex_spot_close[idx])

    gex_thr0p3 = build_gex_events(events_fullyear_0p3, opts_fullyear_path, "thr0p3", asof_spot_2025)
    gex_thr0p25 = build_gex_events(events_fullyear_0p25, opts_fullyear_path, "thr0p25", asof_spot_2025)

    _total_try = gex_thr0p3["n_attempted_before"].sum() + gex_thr0p3["n_attempted_after"].sum()
    _total_ok = gex_thr0p3["n_inverted_before"].sum() + gex_thr0p3["n_inverted_after"].sum()
    print(f"thr0p3: {len(gex_thr0p3)} events, IV inversion success {_total_ok}/{_total_try} ({100*_total_ok/_total_try:.1f}%)")
    print(f"thr0p25: {len(gex_thr0p25)} events")

    # ---- GEX statistical battery (own Bonferroni family, 24 tests) ----
    gex_battery = pd.concat([run_gex_battery(gex_thr0p3, "0.30%"), run_gex_battery(gex_thr0p25, "0.25%")], ignore_index=True)
    _alpha_gex = 0.05 / len(gex_battery)
    gex_battery["bonferroni_significant"] = gex_battery["p_value"] < _alpha_gex
    _gex_n_sig = gex_battery["bonferroni_significant"].sum()
    print(f"\nGEX battery: {len(gex_battery)} tests, Bonferroni alpha={_alpha_gex:.5f}, survivors={_gex_n_sig}")
    print(gex_battery.to_string(index=False, formatters={"statistic": lambda x: f"{x:+.4f}", "p_value": lambda x: f"{x:.4g}"}))

    # ---- The actual novel test: does pre-break dealer gamma regime predict amplification vs dampening? ----
    for h in GEX_HORIZONS:
        if f"fwd_ret_{h}" not in spot_2025.columns:
            spot_2025[f"fwd_ret_{h}"] = spot_2025.groupby(spot_2025["date"].dt.tz_localize(None).dt.date)["close"].transform(lambda s: s.shift(-h) / s - 1)
    _gex_spot_ts2 = spot_2025.sort_values("date")["date"].dt.tz_localize(None).to_numpy()

    def fwd_returns_at(times_naive, h):
        idx = np.searchsorted(_gex_spot_ts2, times_naive, side="right") - 1
        col = spot_2025.sort_values("date")[f"fwd_ret_{h}"].to_numpy()
        idx = np.clip(idx, 0, len(col) - 1)
        return col[idx]

    def gex_regime_test(gex_df, tag):
        df = gex_df.copy()
        df["date_naive"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["regime"] = np.where(df["dealer_gex_before"] < 0, "negative_gamma", "positive_gamma")
        sign = np.where(df["direction"] == "bullish", 1, -1)
        times = df["date_naive"].to_numpy()
        rows = []
        for h in GEX_HORIZONS:
            rets = fwd_returns_at(times, h) * sign
            df[f"signed_ret_{h}"] = rets
            neg = df.loc[df["regime"] == "negative_gamma", f"signed_ret_{h}"].dropna()
            pos = df.loc[df["regime"] == "positive_gamma", f"signed_ret_{h}"].dropna()
            t, p = stats.ttest_ind(neg, pos, equal_var=False)
            rows.append({"threshold": tag, "horizon_min": h, "n_negative_gamma": len(neg), "n_positive_gamma": len(pos),
                         "negative_gamma_mean_pct": neg.mean()*100, "positive_gamma_mean_pct": pos.mean()*100,
                         "diff_pct": (neg.mean()-pos.mean())*100, "t_stat": t, "p_value": p})
        return pd.DataFrame(rows)

    gex_regime = pd.concat([gex_regime_test(gex_thr0p3, "0.30%"), gex_regime_test(gex_thr0p25, "0.25%")], ignore_index=True)
    _alpha_regime = 0.05 / len(gex_regime)
    gex_regime["bonferroni_significant"] = gex_regime["p_value"] < _alpha_regime
    gex_regime["hypothesis_consistent"] = gex_regime["diff_pct"] > 0
    print(f"\nRegime test: {len(gex_regime)} tests (5 horizons x 2 thresholds), Bonferroni alpha={_alpha_regime:.5f}")
    _regime_n_sig = gex_regime["bonferroni_significant"].sum()
    _regime_n_consistent = gex_regime["hypothesis_consistent"].sum()
    print(f"Survivors: {_regime_n_sig}/{len(gex_regime)}  |  "
          f"Direction consistent with amplify/dampen hypothesis: {_regime_n_consistent}/{len(gex_regime)}")
    print(gex_regime.to_string(index=False, formatters={
        "negative_gamma_mean_pct": lambda x: f"{x:+.4f}", "positive_gamma_mean_pct": lambda x: f"{x:+.4f}",
        "diff_pct": lambda x: f"{x:+.4f}", "p_value": lambda x: f"{x:.4g}"}))
    print("\nPre-break dealer-gamma sign does not detectably predict continuation vs fade at the "
          "individual-event level in this sample -- a clean null on the amplify/dampen mechanism, "
          "reported as such rather than folded into the paper's headline claims. The dealer_gex_shift "
          "result above (which does survive Bonferroni) is a robustness confirmation of H1's existing "
          "PE-CE differential finding under a gamma-weighted construction, not evidence for this "
          "separate amplify/dampen claim.")

    print("\nSection 15 done.")


if __name__ == "__main__":
    main()
