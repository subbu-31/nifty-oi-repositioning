#!/usr/bin/env python3
"""
Section 6: H2 -- event-time lead-lag analysis (paper §4.2, Table 3).

For each break event, measure the PE-minus-CE OI differential at a
baseline point H2_BASELINE_MIN before the break, and again at nine offsets
relative to confirmation (H2_OFFSETS_MIN). At each offset, correlate the
shift from baseline (point-biserial) against break direction. This is a
descriptive, event-time analysis, not a family of independent hypothesis
tests -- the nine offsets characterize one trajectory of effect sizes, so
no Bonferroni correction is applied across them (unlike the 30 independent
H1 tests).

Depends on: 05_h1_association.py (oi_shift_fullyear_thr{0p3,0p25}.parquet)
Produces (checkpoints/): h2_event_time_fullyear_thr{0p3,0p25}.parquet,
  table3_thr0p3.csv, table3_thr0p25.csv (Table 3 -- new checkpoint not
  present in the original notebook, added so Section 13's figures and
  Section 14's summary can run as independent scripts)
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, H2_BASELINE_MIN, H2_OFFSETS_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import pc_diff_at


def build_h2_event_time(oi_shift_df, options_path, tag):
    import os
    ckpt = ckpt_path(f"h2_event_time_fullyear_{tag}.parquet")
    if os.path.exists(ckpt):
        return pd.read_parquet(ckpt)

    ev = oi_shift_df.copy()  # already has 'expiry' and 'date' from the H1 build
    ev["date"] = pd.to_datetime(ev["date"])
    ev["dir_bin"] = (ev["direction"] == "bullish").astype(int)

    rows = []
    for expiry, ev_batch in ev.groupby("expiry", sort=False):
        batch = pd.read_parquet(options_path, filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
        del batch

        for _, row in ev_batch.iterrows():
            spot = row["close"]
            strikes = strikes_near(spot)
            break_t = row["date"]
            baseline_t = break_t - pd.Timedelta(minutes=H2_BASELINE_MIN)
            baseline_pc = pc_diff_at(groups, strikes, baseline_t)
            rec = {"direction": row["direction"], "dir_bin": row["dir_bin"], "event_type": row["event_type"]}
            for off in H2_OFFSETS_MIN:
                t = break_t + pd.Timedelta(minutes=off)
                rec[f"pc_shift_{off:+d}"] = pc_diff_at(groups, strikes, t) - baseline_pc
            rows.append(rec)
        del groups
        print(f"  expiry {expiry}: {len(ev_batch)} events done", flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(ckpt, index=False)
    return out


def h2_correlation_table(h2_df, tag):
    print(f"===== H2 event-time lead-lag: {tag} (n={len(h2_df)}) =====")
    print(f"{'offset(min)':>12} | {'r':>8} | {'p':>12}")
    rows = []
    for off in H2_OFFSETS_MIN:
        col = f"pc_shift_{off:+d}"
        r, p = stats.pointbiserialr(h2_df["dir_bin"], h2_df[col])
        print(f"{off:>12} | {r:>8.4f} | {p:>12.3g}")
        rows.append({"offset_min": off, "r": r, "p": p})
    return pd.DataFrame(rows)


def main():
    oi_shift_thr0p3 = pd.read_parquet(ckpt_path("oi_shift_fullyear_thr0p3.parquet"))
    oi_shift_thr0p25 = pd.read_parquet(ckpt_path("oi_shift_fullyear_thr0p25.parquet"))
    options_path = ckpt_path("options_oi_full_year_2025.parquet")

    h2_thr0p3 = build_h2_event_time(oi_shift_thr0p3, options_path, "thr0p3")
    h2_thr0p25 = build_h2_event_time(oi_shift_thr0p25, options_path, "thr0p25")

    table3_thr0p3 = h2_correlation_table(h2_thr0p3, "0.30% threshold")
    table3_thr0p25 = h2_correlation_table(h2_thr0p25, "0.25% threshold")
    table3_thr0p3.to_csv(ckpt_path("table3_thr0p3.csv"), index=False)
    table3_thr0p25.to_csv(ckpt_path("table3_thr0p25.csv"), index=False)

    print("\nSection 6 done.")


if __name__ == "__main__":
    main()
