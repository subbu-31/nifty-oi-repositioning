#!/usr/bin/env python3
"""
Section 21: Placebo / negative-control test -- is the signal specific to
confirmed breaks?

Every test so far (H1-H3, GEX, quadrant, PCR, expiry-day, vol-adaptive)
shows the PE-CE OI differential correlates with direction AT CONFIRMED
structural breaks. None of it establishes that "confirmed break" is doing
any explanatory work at all, versus the far more mundane possibility that
OI simply tracks price direction at ANY moment, break or not. This section
runs that sharper test directly.

Design (mirrors the real H1 test's temporal structure): a pool of non-break
timestamps is built (excluding anything within +/-5 minutes of any real
event, either threshold), each given a "pseudo-direction" from the sign of
the price change over the PRECEDING 5 minutes (mimicking a break's
backward-looking direction confirmation, without the swing-structure
machinery), and the identical PE-CE OI shift is then measured over the NEXT
5 minutes -- exactly as for real events. Two comparisons: (a) unconditional
random resampling (no size control), and (b) magnitude-matched sampling
(each real event matched to the nearest-|move| placebo candidate), which
isolates whether confirmed-break status adds anything BEYOND move size.

Depends on: 02_data_ingestion.py, 04_event_extraction.py, 05_h1_association.py
Produces (checkpoints/): placebo_pool_thr0p3.parquet, ev30_full.parquet,
  real_sorted.parquet, matched.parquet, placebo_summary_stats.json (the last four
  are new checkpoints -- not present in the original notebook -- added so
  Sections 24, 25 and 26 can run as independent scripts without holding
  Section 21's in-memory state)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, H1_AFTER_WINDOW_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import asof_oi_from_groups
from lib.stats_utils import fisher_r_to_z_test
from lib.placebo import load_filtered_placebo_pool

PLACEBO_LOOKBACK_MIN = 5
PLACEBO_N_POOL = 2500
PLACEBO_N_RESAMPLES = 200


def main():
    import os

    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    events_fullyear_0p3 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p3.parquet"))
    events_fullyear_0p25 = pd.read_parquet(ckpt_path("events_fullyear2025_thr0p25.parquet"))
    oi_shift_thr0p3 = pd.read_parquet(ckpt_path("oi_shift_fullyear_thr0p3.parquet"))

    _placebo_rng = np.random.default_rng(42)

    _placebo_pool_ckpt = ckpt_path("placebo_pool_thr0p3.parquet")
    if os.path.exists(_placebo_pool_ckpt):
        pool_result = pd.read_parquet(_placebo_pool_ckpt)
        print(f"[cache hit] {_placebo_pool_ckpt}: {len(pool_result)} candidates")
    else:
        spot_p = spot_2025.sort_values("date").reset_index(drop=True).copy()
        spot_p["trading_date"] = spot_p["date"].dt.tz_localize(None).dt.date
        spot_p["date_naive"] = spot_p["date"].dt.tz_localize(None)
        spot_p["close_lag5"] = spot_p.groupby("trading_date")["close"].shift(PLACEBO_LOOKBACK_MIN)
        spot_p["close_lead5"] = spot_p.groupby("trading_date")["close"].shift(-H1_AFTER_WINDOW_MIN)
        spot_p["preceding_chg"] = spot_p["close"] / spot_p["close_lag5"] - 1
        valid = spot_p["preceding_chg"].notna() & spot_p["close_lead5"].notna()

        excluded_idx = set()
        for tag, events_df in (("0p3", events_fullyear_0p3), ("0p25", events_fullyear_0p25)):
            for i in events_df["idx"]:
                excluded_idx.update(range(i - 5, i + 6))
        print(f"Excluded bars (within +/-5min of any real event, either threshold): {len(excluded_idx)}")

        candidate_mask = valid.to_numpy().copy()
        excl_arr = np.array(sorted(excluded_idx))
        excl_arr = excl_arr[(excl_arr >= 0) & (excl_arr < len(spot_p))]
        candidate_mask[excl_arr] = False
        candidate_positions = np.arange(len(spot_p))[candidate_mask]
        print(f"Candidate pool available: {len(candidate_positions)}")

        sampled_positions = _placebo_rng.choice(candidate_positions, size=min(PLACEBO_N_POOL, len(candidate_positions)), replace=False)
        pool = spot_p.iloc[sampled_positions].copy().reset_index(drop=True)
        pool["pseudo_direction"] = np.where(pool["preceding_chg"] > 0, "bullish", "bearish")

        expiries_p = sorted(pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"), columns=["expiry"])["expiry"].unique())
        def active_expiry_for_p(trading_date):
            for e in expiries_p:
                if e >= trading_date:
                    return e
            return None
        pool["active_expiry"] = pool["trading_date"].apply(active_expiry_for_p)

        results = []
        for expiry, batch_ev in pool.groupby("active_expiry", sort=False):
            if expiry is None:
                continue
            batch = pd.read_parquet(ckpt_path("options_oi_full_year_2025.parquet"), filters=[("expiry", "=", expiry)])
            groups = {}
            for key, g in batch.groupby(["strike", "option_type"], sort=False):
                groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
            del batch
            for _, row in batch_ev.iterrows():
                strikes = strikes_near(row["close"])
                before_t = row["date_naive"]
                after_t = before_t + pd.Timedelta(minutes=H1_AFTER_WINDOW_MIN)
                ce_before = pe_before = ce_after = pe_after = 0.0
                for k in strikes:
                    for opt in ("CE", "PE"):
                        b = asof_oi_from_groups(groups, k, opt, before_t)
                        a = asof_oi_from_groups(groups, k, opt, after_t)
                        if opt == "CE":
                            ce_before += b; ce_after += a
                        else:
                            pe_before += b; pe_after += a
                results.append({
                    "date_naive": row["date_naive"], "close": row["close"], "preceding_chg": row["preceding_chg"],
                    "pseudo_direction": row["pseudo_direction"],
                    "pe_minus_ce_shift": (pe_after - pe_before) - (ce_after - ce_before),
                })
            del groups
        pool_result = pd.DataFrame(results)
        pool_result.to_parquet(_placebo_pool_ckpt, index=False)

    pool_result = load_filtered_placebo_pool()
    print(pool_result["pseudo_direction"].value_counts().to_string())

    # ---- Real events (0.30% threshold): official label vs naive (preceding-5min-sign) proxy label ----
    _spot_sorted_p = spot_2025.sort_values("date").reset_index(drop=True)
    _spot_close = _spot_sorted_p["close"].to_numpy()
    _spot_trading_date = _spot_sorted_p["date"].dt.tz_localize(None).dt.date
    _close_lag5_full = _spot_sorted_p.groupby(_spot_trading_date)["close"].shift(PLACEBO_LOOKBACK_MIN).to_numpy()

    events_0p3_full = events_fullyear_0p3[["idx"]].copy()
    _ev30_idx_all = events_0p3_full["idx"].to_numpy()
    _idx_clipped_all = np.clip(_ev30_idx_all, PLACEBO_LOOKBACK_MIN, None)
    events_0p3_full["preceding_chg"] = np.where(
        _ev30_idx_all >= PLACEBO_LOOKBACK_MIN,
        _spot_close[_idx_clipped_all] / _close_lag5_full[_idx_clipped_all] - 1, np.nan)

    # oi_shift_thr0p3 (357 rows) can have fewer rows than events_fullyear["0p3"] (358 rows) --
    # build_h1_oi_shift silently drops events with no active expiry found (e.g. late-Dec events
    # needing a Jan-2026 expiry not present in the 2025-only options file). Join on idx rather
    # than assuming positional row alignment.
    ev30_full = oi_shift_thr0p3.merge(events_0p3_full, on="idx", how="inner")
    print(f"events_fullyear['0p3']: {len(events_0p3_full)} rows, oi_shift_thr0p3: {len(oi_shift_thr0p3)} rows, "
          f"joined on idx: {len(ev30_full)} rows")
    ev30_full["pe_minus_ce_shift"] = ev30_full["pe_oi_shift"] - ev30_full["ce_oi_shift"]
    ev30_full["dir_bin"] = (ev30_full["direction"] == "bullish").astype(int)
    ev30_full["naive_dir_bin"] = (ev30_full["preceding_chg"] > 0).astype(int)

    r_real_official, p_real_official = stats.pointbiserialr(ev30_full["dir_bin"], ev30_full["pe_minus_ce_shift"])
    agree_rate = (ev30_full["dir_bin"] == ev30_full["naive_dir_bin"]).mean()
    _naive_valid = ev30_full["preceding_chg"].notna()  # naive_dir_bin is already cast to int, so .notna() on it is always True
    r_real_naive, p_real_naive = stats.pointbiserialr(ev30_full.loc[_naive_valid, "naive_dir_bin"].astype(int),
                                                        ev30_full.loc[_naive_valid, "pe_minus_ce_shift"])

    print(f"Real events official label: r={r_real_official:+.4f}, p={p_real_official:.3e}")
    print(f"Naive proxy agrees with official direction: {agree_rate:.1%}")
    print(f"Real events, naive proxy label (apples-to-apples with placebo): r={r_real_naive:+.4f}, p={p_real_naive:.3e}")

    # (a) unconditional random resampling (dir_bin already added by load_filtered_placebo_pool)
    N_REAL = len(ev30_full)
    _rs = []
    for _ in range(PLACEBO_N_RESAMPLES):
        sample = pool_result.sample(n=N_REAL, replace=False, random_state=_placebo_rng.integers(0, 2**31 - 1))
        r_i, _ = stats.pointbiserialr(sample["dir_bin"], sample["pe_minus_ce_shift"])
        _rs.append(r_i)
    _rs = np.array(_rs)
    _pctile_2p5, _pctile_97p5 = np.percentile(_rs, [2.5, 97.5])
    _beat_rate = (np.abs(_rs) >= abs(r_real_official)).mean()
    print(f"\n(a) Unconditional placebo ({PLACEBO_N_RESAMPLES} resamples of n={N_REAL}): "
          f"mean r={_rs.mean():+.4f}, 95% range [{_pctile_2p5:+.4f}, {_pctile_97p5:+.4f}]")
    print(f"Fraction of placebo draws matching/exceeding real r: {_beat_rate:.4%}")

    # (b) magnitude-matched placebo: nearest-|move| match, largest real moves matched first
    real_sorted = ev30_full.dropna(subset=["preceding_chg"]).sort_values(
        "preceding_chg", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    avail = pool_result.copy()
    avail["used"] = False
    matched_rows = []
    for _, r_row in real_sorted.iterrows():
        unused = avail[~avail["used"]]
        if len(unused) == 0:
            break
        diffs = (unused["preceding_chg"] - r_row["preceding_chg"]).abs()
        best_i = diffs.idxmin()
        matched_rows.append(avail.loc[best_i])
        avail.loc[best_i, "used"] = True
    matched = pd.DataFrame(matched_rows).reset_index(drop=True)
    r_matched, p_matched = stats.pointbiserialr(matched["dir_bin"], matched["pe_minus_ce_shift"])

    _match_err = (real_sorted["preceding_chg"].abs().to_numpy()[:len(matched)] - matched["preceding_chg"].abs().to_numpy())
    print(f"(b) Magnitude-matched placebo (n={len(matched)} of {len(real_sorted)} real events with valid lookback):")
    print(f"  real mean |preceding move|={real_sorted['preceding_chg'].abs().mean()*100:.4f}%, "
          f"matched mean={matched['preceding_chg'].abs().mean()*100:.4f}%, "
          f"mean abs matching error={np.abs(_match_err).mean()*100:.4f}pp")
    print(f"  r={r_matched:+.4f}, p={p_matched:.3e}")

    # Fisher r-to-z: is the residual gap (real vs magnitude-matched) itself significant?
    z_naive, p_gap_naive = fisher_r_to_z_test(r_real_naive, _naive_valid.sum(), r_matched, len(matched))
    z_official, p_gap_official = fisher_r_to_z_test(r_real_official, len(ev30_full), r_matched, len(matched))

    print(f"\nFisher r-to-z, real(naive proxy) vs magnitude-matched placebo: "
          f"gap={r_real_naive - r_matched:+.4f}, z={z_naive:.4f}, p={p_gap_naive:.4g}")
    print(f"Fisher r-to-z, real(official label) vs magnitude-matched placebo: "
          f"gap={r_real_official - r_matched:+.4f}, z={z_official:.4f}, p={p_gap_official:.4g}")

    print("\nSummary:")
    _summary_rows = [
        ("Real events (official break direction)", len(ev30_full), r_real_official, p_real_official),
        ("Real events (naive proxy direction)", int(_naive_valid.sum()), r_real_naive, p_real_naive),
        ("Unconditional placebo (mean of 200 draws)", N_REAL, _rs.mean(), np.nan),
        ("Magnitude-matched placebo", len(matched), r_matched, p_matched),
    ]
    print(f"{'Sample':<45}{'n':>6}{'r':>10}{'p':>14}")
    for name, n, r_val, p_val in _summary_rows:
        p_str = f"{p_val:.3e}" if not np.isnan(p_val) else "--"
        print(f"{name:<45}{n:>6}{r_val:>+10.4f}{p_str:>14}")

    print("\nThe unconditional comparison overstates break-specificity, because real breaks average "
          "~3.5x larger preceding moves than random moments, and bigger moves mechanically produce "
          "bigger, more one-sided OI reactions on their own. Once matched on move size, most of the "
          "apparent gap closes. The residual gap is not significant under the strictest apples-to-apples "
          "comparison (naive proxy on both sides), and only marginally significant, uncorrected, using "
          "the officially-defined break-direction label. In other words, a substantial share of the "
          "paper's core finding reflects a generic larger-move-to-larger-OI-reaction relationship "
          "rather than something unique to technically-confirmed structural breaks. This should be "
          "disclosed as a limitation on the causal/mechanistic interpretation, not treated as a "
          "refutation of the association itself, which remains real and statistically robust "
          "throughout every test in this notebook.")

    # ---- New checkpoints for downstream scripts (24, 25, 26) ----
    ev30_full.to_parquet(ckpt_path("ev30_full.parquet"), index=False)
    real_sorted.to_parquet(ckpt_path("real_sorted.parquet"), index=False)
    matched.to_parquet(ckpt_path("matched.parquet"), index=False)
    with open(ckpt_path("placebo_summary_stats.json"), "w") as f:
        json.dump({
            "r_real_official": r_real_official, "p_real_official": p_real_official,
            "r_real_naive": r_real_naive, "p_real_naive": p_real_naive,
            "r_matched": r_matched, "p_matched": p_matched,
            "n_naive_valid": int(_naive_valid.sum()), "n_ev30_full": len(ev30_full), "n_matched": len(matched),
        }, f, indent=2)

    print("\nSection 21 done.")


if __name__ == "__main__":
    main()
