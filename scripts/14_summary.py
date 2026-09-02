#!/usr/bin/env python3
"""
Section 14: Summary -- headline numbers.

A final sanity-check table reprinting every number that appears in the
paper's abstract, so a run of this pipeline can be diff'ed against the
published draft in one glance.

Depends on: 05_h1_association.py, 06_h2_event_time.py, 07_h3_signal.py,
  08_robustness_band_window.py, 09_robustness_quiet_threshold.py,
  10_day_clustered_inference.py
Produces: nothing new (prints only).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path


def main():
    appendix_30tests = pd.read_csv(ckpt_path("appendix_30tests.csv"))
    oi_shift_thr0p3 = pd.read_parquet(ckpt_path("oi_shift_fullyear_thr0p3.parquet"))
    table3_thr0p3 = pd.read_csv(ckpt_path("table3_thr0p3.csv"))
    is_trades = pd.read_csv(ckpt_path("h3_is_trades.csv"))
    oos_trades = pd.read_csv(ckpt_path("h3_oos_trades.csv"))
    with open(ckpt_path("h3_auc_summary.json")) as f:
        auc_summary = json.load(f)
    is_auc, oos_auc = auc_summary["is"], auc_summary["oos"]
    h3_daylevel_inference = pd.read_csv(ckpt_path("h3_daylevel_inference.csv"))
    robustness_band = pd.read_csv(ckpt_path("robustness_band.csv"))
    robustness_window = pd.read_csv(ckpt_path("robustness_window.csv"))
    h3_quiet_sensitivity = pd.read_csv(ckpt_path("h3_quiet_sensitivity.csv"))

    _pece_r = appendix_30tests[(appendix_30tests["feature"] == "PE-CE differential shift") &
                               (appendix_30tests["test"] == "point-biserial r (vs direction)") &
                               (appendix_30tests["threshold"] == "0.30%")]["statistic"].values[0]

    summary = f"""
H1 (association):
  PE-CE differential r = {_pece_r:.3f} (0.30%, n={len(oi_shift_thr0p3)})
  Bonferroni survivors: {appendix_30tests['significant_bonferroni'].sum()} / {len(appendix_30tests)}

H2 (event-time pattern):
  r at t-10min: {table3_thr0p3.iloc[0]['r']:.3f}
  r at t+5min (peak): {table3_thr0p3[table3_thr0p3['offset_min']==5]['r'].values[0]:.3f}
  r at t+10min: {table3_thr0p3.iloc[-1]['r']:.3f}

H3 (tradeable edge):
  IS AUC (up/down): {is_auc['up']:.3f} / {is_auc['down']:.3f}
  OOS AUC (up/down): {oos_auc['up']:.3f} / {oos_auc['down']:.3f}
  IS backtest P&L: {is_trades['pnl'].sum():+.1f} pts (n={len(is_trades)})
  OOS backtest P&L: {oos_trades['pnl'].sum():+.1f} pts (n={len(oos_trades)})
  IS day-clustered p: {h3_daylevel_inference.iloc[0]['p_value']:.4g}
  OOS day-clustered p: {h3_daylevel_inference.iloc[1]['p_value']:.4g}

Robustness:
  Strike band r range: {robustness_band['r'].min():.3f} - {robustness_band['r'].max():.3f}
  OI window r range: {robustness_window['r'].min():.3f} - {robustness_window['r'].max():.3f} (peaks at 5min)
  Quiet-threshold OOS p range: {h3_quiet_sensitivity['oos_pnl_p'].min():.3f} - {h3_quiet_sensitivity['oos_pnl_p'].max():.3f}
"""
    print(summary)
    print("Section 14 done.")


if __name__ == "__main__":
    main()
