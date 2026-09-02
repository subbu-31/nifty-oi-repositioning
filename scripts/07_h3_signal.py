#!/usr/bin/env python3
"""
Section 7: H3 -- real-time predictive signal & genuine out-of-sample test
(paper §4.3, Table 4).

A signal computable in real time: for each trading day, fix a reference
price (09:45 opening-range close) and strike band for the full session,
track per-minute CE/PE OI totals via vectorized as-of lookups, compute a
rolling H3_ROLL_MIN-minute PE-minus-CE shift, and record a forward-looking
label (does a bullish/bearish break occur within the next
H3_FWD_HORIZON_MIN minutes?). The label is used only to score the signal
after the fact, never as an input to the signal itself.

The confound and the fix: H2 shows OI's association strengthening through a
window that can overlap with an already-underway move -- so a forward-
looking break label can overlap with a reactive OI response to a move that
started before the labeling window began. The signal is therefore
re-evaluated only at minutes where the trailing 10-minute absolute price
move is at or below a "quiet" threshold (Section 9 sensitivity-tests this
choice).

Discipline: every parameter (quiet threshold, decile thresholds, logistic
regression coefficients) is fit exclusively on Jan-Jul 2025 data and then
applied, without any refitting, to the Jan-Apr 2026 holdout.

Depends on: 02_data_ingestion.py, 04_event_extraction.py
Produces (checkpoints/): h3_panel_h1window.parquet, h3_panel_oos2026.parquet,
  h3_locked_is_params.json, h3_is_trades.csv, h3_oos_trades.csv,
  h3_auc_summary.json (trades/AUC checkpoints are new -- not present in the
  original notebook -- added so Section 10's day-clustered inference,
  Section 13's figures, and Section 14's summary can run as independent
  scripts)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.h3 import build_h3_panel, add_trailing_move, evaluate_h3


def main():
    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    spot_2026 = pd.read_parquet(ckpt_path("nifty_spot_1min_2026_jan_apr_oos.parquet"))
    events_h1window_0p3 = pd.read_parquet(ckpt_path("events_h1window_thr0p3.parquet"))
    events_oos_0p3 = pd.read_parquet(ckpt_path("events_oos2026_thr0p3.parquet"))

    _opts_2026_ts = pd.read_parquet(ckpt_path("options_oi_oos_2026.parquet"), columns=["timestamp"])
    opts_2026_tmin, opts_2026_tmax = _opts_2026_ts["timestamp"].min(), _opts_2026_ts["timestamp"].max()
    del _opts_2026_ts

    # In-sample panel: Jan 2 - Jul 3, 2025 (the H1-2025 options coverage window)
    h3_panel_is = build_h3_panel(
        ckpt_path("options_oi_h1window_2025.parquet"), spot_2025, events_h1window_0p3,
        pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-07-03").date(),
        "h3_panel_h1window.parquet",
    )

    # Out-of-sample panel: identical construction, Jan-Apr 2026
    h3_panel_oos = build_h3_panel(
        ckpt_path("options_oi_oos_2026.parquet"), spot_2026, events_oos_0p3,
        opts_2026_tmin.date(), opts_2026_tmax.date(),
        "h3_panel_oos2026.parquet",
    )

    # --- Lock every parameter on in-sample data only ---
    import os
    h3_panel_is, is_ts, is_close, is_price_at_vec = add_trailing_move(h3_panel_is, spot_2025)

    locked_path = ckpt_path("h3_locked_is_params.json")
    if os.path.exists(locked_path):
        with open(locked_path) as f:
            locked = json.load(f)
    else:
        from sklearn.linear_model import LogisticRegression

        quiet_thr = float(h3_panel_is["trailing_10min_abs_move_pct"].median())
        quiet = h3_panel_is[h3_panel_is["trailing_10min_abs_move_pct"] <= quiet_thr].copy()
        quiet["abs_pc_shift"] = quiet["pc_shift"].abs()

        hi_thr = float(quiet["pc_shift"].quantile(0.90))
        lo_thr = float(quiet["pc_shift"].quantile(0.10))

        X = quiet[["pc_shift", "abs_pc_shift"]].to_numpy()
        models = {}
        for label_col, name in [("label_up_next15", "up"), ("label_down_next15", "down")]:
            y = quiet[label_col].astype(int).to_numpy()
            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X, y)
            models[name] = {"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}

        locked = {
            "quiet_thr_median_trailing10min_absmove_pct": quiet_thr,
            "pc_shift_hi_decile_thr": hi_thr, "pc_shift_lo_decile_thr": lo_thr,
            "logistic_models": models, "is_n_panel": len(h3_panel_is), "is_n_quiet": len(quiet),
        }
        with open(locked_path, "w") as f:
            json.dump(locked, f, indent=2)

    print("LOCKED in-sample parameters (fit once, applied verbatim to OOS):")
    print(f"  quiet_thr = {locked['quiet_thr_median_trailing10min_absmove_pct']:.5f}")
    print(f"  hi/lo decile thr = {locked['pc_shift_hi_decile_thr']:.2f} / {locked['pc_shift_lo_decile_thr']:.2f}")
    print(f"  IS quiet subset n = {locked['is_n_quiet']} of {locked['is_n_panel']}")

    # --- Score in-sample (cross-validated) and out-of-sample (zero refitting) ---
    is_result = evaluate_h3(h3_panel_is, is_ts, is_close, is_price_at_vec,
                             locked["quiet_thr_median_trailing10min_absmove_pct"],
                             locked["pc_shift_hi_decile_thr"], locked["pc_shift_lo_decile_thr"],
                             day_blocked_cv=True, label="IN-SAMPLE (Jan-Jul 2025, day-blocked CV AUC)")

    h3_panel_oos, oos_ts, oos_close, oos_price_at_vec = add_trailing_move(h3_panel_oos, spot_2026)
    oos_result = evaluate_h3(h3_panel_oos, oos_ts, oos_close, oos_price_at_vec,
                              locked["quiet_thr_median_trailing10min_absmove_pct"],
                              locked["pc_shift_hi_decile_thr"], locked["pc_shift_lo_decile_thr"],
                              models=locked["logistic_models"],
                              label="OUT-OF-SAMPLE (Jan-Apr 2026, locked IS params, zero refitting)")

    is_result["trades"].to_csv(ckpt_path("h3_is_trades.csv"), index=False)
    oos_result["trades"].to_csv(ckpt_path("h3_oos_trades.csv"), index=False)
    with open(ckpt_path("h3_auc_summary.json"), "w") as f:
        json.dump({"is": is_result["auc"], "oos": oos_result["auc"]}, f, indent=2)

    print("\nSection 7 done.")


if __name__ == "__main__":
    main()
