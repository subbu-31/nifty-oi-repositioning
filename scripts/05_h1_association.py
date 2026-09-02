#!/usr/bin/env python3
"""
Section 5: H1 -- statistical association (paper §4.1, Table 2, Appendix A).

For every identified structural break, aggregate options open interest at
strikes within STRIKE_BAND of the underlying, separately for calls (CE) and
puts (PE), measured immediately before the break and H1_AFTER_WINDOW_MIN
minutes after. All OI lookups use an as-of convention.

The 30-test battery: five feature constructions (raw total OI shift, total
OI shift as % of pre-break OI, CE-only shift, PE-only shift, PE-minus-CE
differential) x three test types (point-biserial correlation vs. direction,
Welch's t-test bullish-vs-bearish, one-sample t-test vs. zero) x two
thresholds = 30 tests. Because 30 tests are run, Bonferroni correction is
reported alongside raw p-values (family-wise alpha=0.05 -> per-test
threshold alpha=0.05/30=0.00167). One event (2025-12-24 expiry, only 20
traded strikes vs. a typical 80-120) produces a divide-by-zero in the
%-shift feature and is filtered out for that feature only
(total_oi_before > 0).

Depends on: 04_event_extraction.py
Produces (checkpoints/): oi_shift_fullyear_thr0p3.parquet,
  oi_shift_fullyear_thr0p25.parquet, appendix_30tests.csv
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.h1 import get_h1_oi_shift


def run_h1_battery(oi_shift_df, thr_tag):
    df = oi_shift_df.copy()
    df["dir_bin"] = (df["direction"] == "bullish").astype(int)
    df_pct = df[df["total_oi_before"] > 0].copy()
    df_pct["oi_shift_pct"] = df_pct["total_oi_shift"] / df_pct["total_oi_before"] * 100
    df["pe_minus_ce_shift"] = df["pe_oi_shift"] - df["ce_oi_shift"]

    rows = []
    features = [
        ("total_oi_shift", "Raw total OI shift", df),
        ("oi_shift_pct", "Total OI shift, % of pre-break OI", df_pct),
        ("ce_oi_shift", "CE OI shift", df),
        ("pe_oi_shift", "PE OI shift", df),
        ("pe_minus_ce_shift", "PE-CE differential shift", df),
    ]
    for col, label, frame in features:
        up = frame.loc[frame["dir_bin"] == 1, col]
        down = frame.loc[frame["dir_bin"] == 0, col]
        r, p_r = stats.pointbiserialr(frame["dir_bin"], frame[col])
        t, p_t = stats.ttest_ind(up, down, equal_var=False)
        t0, p0 = stats.ttest_1samp(frame[col], 0)
        rows.append({"threshold": thr_tag, "feature": label, "test": "point-biserial r (vs direction)",
                     "n": len(frame), "statistic": r, "raw_p": p_r})
        rows.append({"threshold": thr_tag, "feature": label, "test": "Welch t (up vs down)",
                     "n": len(frame), "statistic": t, "raw_p": p_t})
        rows.append({"threshold": thr_tag, "feature": label, "test": "one-sample t (!=0)",
                     "n": len(frame), "statistic": t0, "raw_p": p0})
    return rows


def main():
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    events_fullyear_0p25 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p25.parquet"))

    oi_shift_thr0p3 = get_h1_oi_shift("thr0p3", events_fullyear_0p3, ckpt_path("options_oi_full_year_2025.parquet"))
    oi_shift_thr0p25 = get_h1_oi_shift("thr0p25", events_fullyear_0p25, ckpt_path("options_oi_full_year_2025.parquet"))

    all_rows = run_h1_battery(oi_shift_thr0p3, "0.30%") + run_h1_battery(oi_shift_thr0p25, "0.25%")
    appendix_30tests = pd.DataFrame(all_rows)
    n_tests = len(appendix_30tests)
    appendix_30tests["bonferroni_p"] = (appendix_30tests["raw_p"] * n_tests).clip(upper=1.0)
    appendix_30tests["significant_bonferroni"] = appendix_30tests["bonferroni_p"] < 0.05
    appendix_30tests = appendix_30tests.sort_values("raw_p").reset_index(drop=True)
    appendix_30tests.to_csv(ckpt_path("appendix_30tests.csv"), index=False)

    print(f"Total tests: {n_tests}")
    print(f"Survivors (Bonferroni-adjusted p < 0.05, family-wise alpha=0.05, per-test alpha=0.00167): "
          f"{appendix_30tests['significant_bonferroni'].sum()} / {n_tests}")
    print(appendix_30tests.head(10).to_string(index=False))

    print("\nSection 5 done.")


if __name__ == "__main__":
    main()
