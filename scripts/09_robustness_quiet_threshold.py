#!/usr/bin/env python3
"""
Section 9: Robustness -- H3 quiet-subset threshold sensitivity
(paper §5.6, Table 8).

Sweeps the "quiet" quantile cut (25th/40th/50th/60th/75th percentile of
trailing 10-min |move|%), relocking every downstream parameter (decile
thresholds, logistic coefficients) from in-sample data at each quantile,
then scoring out-of-sample with zero refitting -- exactly the discipline
used for the headline result, repeated across five candidate definitions of
"quiet" instead of one.

Depends on: 02_data_ingestion.py, 07_h3_signal.py
Produces (checkpoints/): h3_quiet_sensitivity.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.h3 import add_trailing_move, sigmoid


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    spot_2025 = pd.read_parquet(ckpt_path("nifty_spot_1min_2025.parquet"))
    spot_2026 = pd.read_parquet(ckpt_path("nifty_spot_1min_2026_jan_apr_oos.parquet"))
    h3_panel_is_raw = pd.read_parquet(ckpt_path("h3_panel_h1window.parquet"))
    h3_panel_oos_raw = pd.read_parquet(ckpt_path("h3_panel_oos2026.parquet"))

    h3_panel_is, _, _, _ = add_trailing_move(h3_panel_is_raw, spot_2025)
    h3_panel_oos, _, _, oos_price_at_vec = add_trailing_move(h3_panel_oos_raw, spot_2026)

    quiet_sens_ckpt = ckpt_path("h3_quiet_sensitivity.csv")
    if os.path.exists(quiet_sens_ckpt):
        h3_quiet_sensitivity = pd.read_csv(quiet_sens_ckpt)
    else:
        rows = []
        for q in [0.25, 0.40, 0.50, 0.60, 0.75]:
            quiet_thr = h3_panel_is["trailing_10min_abs_move_pct"].quantile(q)
            quiet_is = h3_panel_is[h3_panel_is["trailing_10min_abs_move_pct"] <= quiet_thr].copy()
            quiet_is["abs_pc_shift"] = quiet_is["pc_shift"].abs()
            X_is = quiet_is[["pc_shift", "abs_pc_shift"]].to_numpy()
            groups_arr = quiet_is["trading_date"].astype(str).to_numpy()

            models, is_auc = {}, {}
            for label_col, name in [("label_up_next15", "up"), ("label_down_next15", "down")]:
                y = quiet_is[label_col].astype(int).to_numpy()
                gkf = GroupKFold(n_splits=5)
                oof = np.zeros(len(y))
                for tr, te in gkf.split(X_is, y, groups=groups_arr):
                    if y[tr].sum() < 5:
                        oof[te] = y[tr].mean(); continue
                    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
                    clf.fit(X_is[tr], y[tr])
                    oof[te] = clf.predict_proba(X_is[te])[:, 1]
                is_auc[name] = roc_auc_score(y, oof) if y.sum() >= 10 else float("nan")
                clf_final = LogisticRegression(class_weight="balanced", max_iter=1000)
                clf_final.fit(X_is, y)
                models[name] = (clf_final.coef_[0], clf_final.intercept_[0])

            hi_thr_q, lo_thr_q = quiet_is["pc_shift"].quantile(0.90), quiet_is["pc_shift"].quantile(0.10)

            quiet_oos = h3_panel_oos[h3_panel_oos["trailing_10min_abs_move_pct"] <= quiet_thr].copy()
            quiet_oos["abs_pc_shift"] = quiet_oos["pc_shift"].abs()
            X_oos = quiet_oos[["pc_shift", "abs_pc_shift"]].to_numpy()

            oos_auc = {}
            for label_col, name in [("label_up_next15", "up"), ("label_down_next15", "down")]:
                y = quiet_oos[label_col].astype(int).to_numpy()
                coef, intercept = models[name]
                pred = sigmoid(X_oos @ coef + intercept)
                oos_auc[name] = roc_auc_score(y, pred) if y.sum() >= 5 else float("nan")

            up_sig = quiet_oos[quiet_oos["pc_shift"] >= hi_thr_q]
            down_sig = quiet_oos[quiet_oos["pc_shift"] <= lo_thr_q]
            pnl_up = oos_price_at_vec(up_sig["date_naive"].to_numpy() + np.timedelta64(15, "m")) - \
                     oos_price_at_vec(up_sig["date_naive"].to_numpy())
            pnl_down = oos_price_at_vec(down_sig["date_naive"].to_numpy()) - \
                       oos_price_at_vec(down_sig["date_naive"].to_numpy() + np.timedelta64(15, "m"))
            all_pnl = np.concatenate([pnl_up[~np.isnan(pnl_up)], pnl_down[~np.isnan(pnl_down)]])
            t_stat, p_val = stats.ttest_1samp(all_pnl, 0) if len(all_pnl) > 1 else (float("nan"), float("nan"))

            rows.append({"quiet_quantile": q, "quiet_thr_pct": quiet_thr,
                         "is_n": len(quiet_is), "oos_n": len(quiet_oos),
                         "is_auc_up": is_auc["up"], "is_auc_down": is_auc["down"],
                         "oos_auc_up": oos_auc["up"], "oos_auc_down": oos_auc["down"],
                         "oos_n_trades": len(all_pnl),
                         "oos_pnl_total": all_pnl.sum() if len(all_pnl) else float("nan"),
                         "oos_pnl_t": t_stat, "oos_pnl_p": p_val})
            print(f"q={q}: thr={quiet_thr:.4f}% | IS AUC {is_auc['up']:.3f}/{is_auc['down']:.3f} "
                  f"| OOS AUC {oos_auc['up']:.3f}/{oos_auc['down']:.3f} | OOS PnL={all_pnl.sum():.1f} "
                  f"(n={len(all_pnl)}, p={p_val:.3g})", flush=True)
        h3_quiet_sensitivity = pd.DataFrame(rows)
        h3_quiet_sensitivity.to_csv(quiet_sens_ckpt, index=False)

    print(h3_quiet_sensitivity.to_string(index=False))
    print("\nSection 9 done.")


if __name__ == "__main__":
    main()
