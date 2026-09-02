"""
Section 7 core builders: the real-time H3 signal panel, the trailing-move
confound filter, and the locked-parameter scoring routine (shared between
the headline IS/OOS test in Section 7 and the quiet-threshold sensitivity
sweep in Section 9).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from lib.config import ckpt_path, H3_ROLL_MIN, H3_FWD_HORIZON_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import vectorized_asof, next_event_within_vec


def build_h3_panel(options_path, spot_df, events_df, date_lo, date_hi, ckpt_name):
    ckpt = ckpt_path(ckpt_name)
    if os.path.exists(ckpt):
        return pd.read_parquet(ckpt)

    oi = pd.read_parquet(options_path).sort_values("timestamp")
    groups = {}
    for key, g in oi.groupby(["expiry", "strike", "option_type"], sort=False):
        groups[key] = (g["timestamp"].to_numpy(), g["oi"].to_numpy())
    expiries = sorted(oi["expiry"].unique())

    def active_expiry_for(trading_date):
        for e in expiries:
            if e >= trading_date:
                return e
        return None

    spot_all = spot_df.sort_values("date").copy()
    spot_all["trading_date"] = spot_all["date"].dt.date
    spot_all["date_naive"] = spot_all["date"].dt.tz_localize(None)

    ev = events_df.copy()
    ev["date"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    up_times = np.sort(ev.loc[ev["direction"] == "bullish", "date"].to_numpy())
    down_times = np.sort(ev.loc[ev["direction"] == "bearish", "date"].to_numpy())

    trading_dates = sorted(spot_all["trading_date"].unique())
    trading_dates = [d for d in trading_dates if date_lo <= d <= date_hi]

    all_rows = []
    for d in trading_dates:
        day_spot = spot_all[spot_all["trading_date"] == d].sort_values("date")
        if len(day_spot) < 40:
            continue
        or_window = day_spot[(day_spot["date"].dt.time >= pd.Timestamp("09:15").time()) &
                              (day_spot["date"].dt.time <= pd.Timestamp("09:44").time())]
        if len(or_window) < 20:
            continue
        ref_price = or_window["close"].iloc[-1]
        expiry = active_expiry_for(d)
        if expiry is None:
            continue
        strikes = strikes_near(ref_price)

        minute_ts_naive = day_spot["date_naive"].to_numpy()
        ce_total = np.zeros(len(minute_ts_naive))
        pe_total = np.zeros(len(minute_ts_naive))
        for k in strikes:
            for opt, acc in (("CE", ce_total), ("PE", pe_total)):
                key = (expiry, k, opt)
                if key in groups:
                    tarr, oarr = groups[key]
                    acc += vectorized_asof(tarr, oarr, minute_ts_naive)

        day_df = pd.DataFrame({"date_naive": minute_ts_naive, "trading_date": d,
                                "pc_diff": pe_total - ce_total})
        day_df["pc_shift"] = day_df["pc_diff"].diff(H3_ROLL_MIN)
        all_rows.append(day_df)

    panel = pd.concat(all_rows, ignore_index=True).dropna(subset=["pc_shift"])
    panel_dates = panel["date_naive"].to_numpy()
    panel["label_up_next15"] = next_event_within_vec(up_times, panel_dates, H3_FWD_HORIZON_MIN)
    panel["label_down_next15"] = next_event_within_vec(down_times, panel_dates, H3_FWD_HORIZON_MIN)
    panel.to_parquet(ckpt, index=False)
    print(f"Panel built: {len(panel)} obs across {panel['trading_date'].nunique()} days -> {ckpt}")
    print(f"Base rate: P(up)={panel['label_up_next15'].mean():.4f}, "
          f"P(down)={panel['label_down_next15'].mean():.4f}")
    return panel


def add_trailing_move(panel, spot_df):
    spot_sorted = spot_df.sort_values("date").copy()
    spot_sorted["date_naive"] = spot_sorted["date"].dt.tz_localize(None)
    ts_arr, close_arr = spot_sorted["date_naive"].to_numpy(), spot_sorted["close"].to_numpy()

    def price_at_vec(targets):
        targets = np.asarray(targets, dtype="datetime64[ns]")
        idx = np.searchsorted(ts_arr, targets, side="right") - 1
        return np.where(idx >= 0, close_arr[np.clip(idx, 0, len(close_arr) - 1)], np.nan)

    panel = panel.sort_values("date_naive").reset_index(drop=True)
    t = panel["date_naive"].to_numpy()
    now, ago = price_at_vec(t), price_at_vec(t - np.timedelta64(10, "m"))
    panel["trailing_10min_abs_move_pct"] = np.abs((now - ago) / ago * 100)
    return panel, ts_arr, close_arr, price_at_vec


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def evaluate_h3(panel, ts_arr, close_arr, price_at_vec, quiet_thr, hi_thr, lo_thr,
                 models=None, day_blocked_cv=False, label="IS"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    quiet = panel[panel["trailing_10min_abs_move_pct"] <= quiet_thr].copy()
    quiet["abs_pc_shift"] = quiet["pc_shift"].abs()
    X = quiet[["pc_shift", "abs_pc_shift"]].to_numpy()

    auc = {}
    for label_col, name in [("label_up_next15", "up"), ("label_down_next15", "down")]:
        y = quiet[label_col].astype(int).to_numpy()
        if y.sum() < 5:
            auc[name] = float("nan"); continue
        if day_blocked_cv:
            groups_arr = quiet["trading_date"].astype(str).to_numpy()
            gkf = GroupKFold(n_splits=5)
            oof = np.zeros(len(y))
            for tr, te in gkf.split(X, y, groups=groups_arr):
                if y[tr].sum() < 5:
                    oof[te] = y[tr].mean(); continue
                clf = LogisticRegression(class_weight="balanced", max_iter=1000)
                clf.fit(X[tr], y[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
            auc[name] = roc_auc_score(y, oof) if y.sum() >= 10 else float("nan")
        else:
            coef = np.array(models[name]["coef"][0]); intercept = models[name]["intercept"][0]
            pred = sigmoid(X @ coef + intercept)
            auc[name] = roc_auc_score(y, pred)

    up_sig = quiet[quiet["pc_shift"] >= hi_thr]
    down_sig = quiet[quiet["pc_shift"] <= lo_thr]
    up_t, down_t = up_sig["date_naive"].to_numpy(), down_sig["date_naive"].to_numpy()
    pnl_up = price_at_vec(up_t + np.timedelta64(15, "m")) - price_at_vec(up_t)
    pnl_down = price_at_vec(down_t) - price_at_vec(down_t + np.timedelta64(15, "m"))
    trades = pd.concat([
        pd.DataFrame({"trading_date": up_sig["trading_date"].to_numpy(), "pnl": pnl_up}),
        pd.DataFrame({"trading_date": down_sig["trading_date"].to_numpy(), "pnl": pnl_down}),
    ], ignore_index=True).dropna()

    t_stat, p_val = stats.ttest_1samp(trades["pnl"], 0) if len(trades) > 1 else (float("nan"), float("nan"))
    print(f"===== H3 {label} =====")
    print(f"  Quiet-subset N: {len(quiet)}")
    print(f"  AUC up/down: {auc['up']:.3f} / {auc['down']:.3f}")
    print(f"  Backtest trades: {len(trades)}, net P&L: {trades['pnl'].sum():.1f}")
    print(f"  Trade-level one-sample t-test: t={t_stat:.3f}, p={p_val:.4g}")
    return {"quiet": quiet, "auc": auc, "trades": trades, "t_stat": t_stat, "p_val": p_val}
