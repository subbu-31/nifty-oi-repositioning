#!/usr/bin/env python3
"""
Section 25: Day-clustered bootstrap and cluster-robust regressions for the
magnitude bins.

Two follow-ups on the magnitude-matching analysis, both accounting for
within-day clustering: (1) a day-block bootstrap 95% CI on the real/placebo
median-ratio within each preceding-move-size bin (Table 10), and (2)
cluster-robust (by trading day) re-estimation of the three Appendix B
regressions (linear, rank-based, log-log) of the signed OI shift on move
size interacted with real-vs.-placebo status.

Depends on: 21_placebo_test.py (ev30_full.parquet, placebo_pool_thr0p3.parquet)
Produces: nothing new (prints only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.placebo import load_filtered_placebo_pool


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def main():
    ev30_full = pd.read_parquet(ckpt_path("ev30_full.parquet"))
    pool_result = load_filtered_placebo_pool()

    _real_bins = ev30_full.dropna(subset=["preceding_chg"]).copy()
    _real_bins = _real_bins[_real_bins["preceding_chg"] != 0].copy()
    _real_bins["is_break"] = 1
    _real_bins["trading_date"] = pd.to_datetime(_real_bins["date"]).dt.date
    _real_bins["abs_move_pct"] = _real_bins["preceding_chg"].abs() * 100
    _real_bins["signed_shift"] = _real_bins["pe_minus_ce_shift"] * np.sign(_real_bins["preceding_chg"])

    _pool_bins = pool_result.copy()
    _pool_bins["is_break"] = 0
    _pool_bins["trading_date"] = pd.to_datetime(_pool_bins["date_naive"]).dt.date
    _pool_bins["abs_move_pct"] = _pool_bins["preceding_chg"].abs() * 100
    _pool_bins["signed_shift"] = _pool_bins["pe_minus_ce_shift"] * np.sign(_pool_bins["preceding_chg"])

    combined = pd.concat([
        _real_bins[["preceding_chg", "pe_minus_ce_shift", "is_break", "abs_move_pct", "signed_shift", "trading_date"]],
        _pool_bins[["preceding_chg", "pe_minus_ce_shift", "is_break", "abs_move_pct", "signed_shift", "trading_date"]],
    ], ignore_index=True)
    combined["day_id"] = pd.factorize(combined["trading_date"])[0]

    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 100]
    labels = ["0-0.05%", "0.05-0.10%", "0.10-0.15%", "0.15-0.20%", "0.20-0.30%", ">0.30%"]
    combined["bin"] = pd.cut(combined["abs_move_pct"], bins=bins, labels=labels)

    N_BOOT_BINS = 5000
    _bins_rng = np.random.default_rng(7)

    print("Table 10 bins: day-block bootstrap 95% CI on the real/placebo signed-shift median ratio")
    for b in labels:
        sub = combined[combined["bin"] == b]
        r_sub = sub[sub["is_break"] == 1]
        p_sub = sub[sub["is_break"] == 0]
        if len(r_sub) < 3 or len(p_sub) < 3:
            print(f"{b:>12}: too few obs, skipped")
            continue
        r_days = r_sub["trading_date"].unique()
        p_days = p_sub["trading_date"].unique()
        r_by_day = {d: r_sub[r_sub["trading_date"] == d]["signed_shift"].to_numpy() for d in r_days}
        p_by_day = {d: p_sub[p_sub["trading_date"] == d]["signed_shift"].to_numpy() for d in p_days}
        ratios = np.empty(N_BOOT_BINS)
        for i in range(N_BOOT_BINS):
            rd = _bins_rng.choice(r_days, size=len(r_days), replace=True)
            pdd = _bins_rng.choice(p_days, size=len(p_days), replace=True)
            rvals = np.concatenate([r_by_day[d] for d in rd])
            pvals = np.concatenate([p_by_day[d] for d in pdd])
            med_p = np.median(pvals)
            ratios[i] = np.median(rvals) / med_p if med_p != 0 else np.nan
        ratios = ratios[~np.isnan(ratios)]
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        frac_below_1 = (ratios <= 1.0).mean()
        print(f"{b:>12}: n_real={len(r_sub):>4}({len(r_days)}d) n_placebo={len(p_sub):>5}({len(p_days)}d) | "
              f"point ratio={r_sub['signed_shift'].median()/p_sub['signed_shift'].median():.2f}x | "
              f"day-boot 95% CI=[{lo:.2f}x, {hi:.2f}x] | P(ratio<=1)={frac_below_1:.4f}")

    print("\nAppendix B regressions: cluster-robust (by trading day) standard errors")
    combined["preceding_chg_pct"] = combined["preceding_chg"] * 100
    lin = smf.ols("pe_minus_ce_shift ~ preceding_chg_pct * is_break", data=combined).fit(
        cov_type="cluster", cov_kwds={"groups": combined["day_id"]})
    print("\n-- Linear OLS, cluster-robust by day --")
    print(lin.summary().tables[1])

    combined["rank_shift"] = combined["pe_minus_ce_shift"].rank()
    combined["rank_move"] = combined["abs_move_pct"].rank()
    rank_m = smf.ols("rank_shift ~ rank_move * is_break", data=combined).fit(
        cov_type="cluster", cov_kwds={"groups": combined["day_id"]})
    print("\n-- Rank-based OLS, cluster-robust by day --")
    print(rank_m.summary().tables[1])

    combined["log_abs_move"] = np.log1p(combined["abs_move_pct"] * 1000)
    combined["log_signed_shift"] = _signed_log1p(combined["signed_shift"])
    loglog = smf.ols("log_signed_shift ~ log_abs_move * is_break", data=combined).fit(
        cov_type="cluster", cov_kwds={"groups": combined["day_id"]})
    print("\n-- Log-log OLS, cluster-robust by day --")
    print(loglog.summary().tables[1])
    print(f"\nn_days total = {combined['day_id'].nunique()}")

    print("\nSection 25 done.")


if __name__ == "__main__":
    main()
