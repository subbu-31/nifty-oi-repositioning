#!/usr/bin/env python3
"""
Section 13: Figures.

Reproduces the three figures used in the paper: event-time OI response by
break direction (Figure 2), the H2 correlation curve at both thresholds
(Figure 3), and the H3 in-sample-vs-out-of-sample comparison (Figure 4).

Depends on: 06_h2_event_time.py, 07_h3_signal.py
Produces (checkpoints/): fig2_event_time_oi.png, fig3_h2_correlation_curve.png,
  fig4_is_vs_oos.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path, CKPT, H2_OFFSETS_MIN

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def main():
    h2_thr0p3 = pd.read_parquet(ckpt_path("h2_event_time_fullyear_thr0p3.parquet"))
    h2_thr0p25 = pd.read_parquet(ckpt_path("h2_event_time_fullyear_thr0p25.parquet"))
    table3_thr0p3 = pd.read_csv(ckpt_path("table3_thr0p3.csv"))
    table3_thr0p25 = pd.read_csv(ckpt_path("table3_thr0p25.csv"))
    is_trades = pd.read_csv(ckpt_path("h3_is_trades.csv"))
    oos_trades = pd.read_csv(ckpt_path("h3_oos_trades.csv"))
    with open(ckpt_path("h3_auc_summary.json")) as f:
        auc_summary = json.load(f)
    is_auc, oos_auc = auc_summary["is"], auc_summary["oos"]

    FIG_DIR = CKPT
    OFFSETS = H2_OFFSETS_MIN

    # ---- Figure 2: event-time OI response, bullish vs bearish ----
    cols = [f"pc_shift_{o:+d}" for o in OFFSETS]
    bullish_mean = h2_thr0p3.loc[h2_thr0p3["direction"] == "bullish", cols].mean()
    bearish_mean = h2_thr0p3.loc[h2_thr0p3["direction"] == "bearish", cols].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.plot(OFFSETS, bullish_mean.values, marker="o", color="#2166AC", label="Bullish breaks", linewidth=2)
    ax.plot(OFFSETS, bearish_mean.values, marker="o", color="#B2182B", label="Bearish breaks", linewidth=2)
    ax.set_xlabel("Minutes relative to break confirmation")
    ax.set_ylabel("PE - CE OI differential, shift from t-15min baseline")
    ax.set_title(f"Figure 2 -- Event-time OI response around structural breaks\n"
                 f"(0.30% threshold, full-year 2025, n={len(h2_thr0p3)})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_event_time_oi.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: H2 correlation curve, both thresholds ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.plot(OFFSETS, table3_thr0p3["r"], marker="o", color="#2166AC",
            label=f"0.30% threshold (n={len(h2_thr0p3)})", linewidth=2)
    ax.plot(OFFSETS, table3_thr0p25["r"], marker="s", color="#4DAF4A",
            label=f"0.25% threshold (n={len(h2_thr0p25)})", linewidth=2)
    ax.set_xlabel("Minutes relative to break confirmation")
    ax.set_ylabel("Point-biserial r (OI shift vs. break direction)")
    ax.set_title("Figure 3 -- H2 event-time lead-lag correlation curve\n"
                  "Rises to a peak at +5min post-confirmation, then partially declines")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_h2_correlation_curve.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4: IS vs OOS comparison ----
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["AUC\n(UP)", "AUC\n(DOWN)"]
    is_vals = [is_auc["up"], is_auc["down"]]
    oos_vals = [oos_auc["up"], oos_auc["down"]]
    x = np.arange(len(labels)); w = 0.35
    axes[0].bar(x - w/2, is_vals, w, label="In-sample\n(Jan-Jul 2025)", color="#2166AC")
    axes[0].bar(x + w/2, oos_vals, w, label="Out-of-sample\n(Jan-Apr 2026)", color="#B2182B")
    axes[0].axhline(0.5, color="#333333", linewidth=1, linestyle=":", label="Chance (0.50)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_title("Classifier AUC")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    pnl_labels = [f"In-sample\n(Jan-Jul 2025)\nn={len(is_trades)}",
                  f"Out-of-sample\n(Jan-Apr 2026)\nn={len(oos_trades)}"]
    pnl_vals = [is_trades["pnl"].sum(), oos_trades["pnl"].sum()]
    axes[1].bar(pnl_labels, pnl_vals, color=["#2166AC", "#B2182B"])
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_title("Backtest net P&L (index points)")
    axes[1].spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(pnl_vals):
        axes[1].text(i, v + (150 if v > 0 else -300), f"{v:+.0f}", ha="center", fontsize=10)

    fig.suptitle("Figure 4 -- H3 signal: in-sample edge vs. genuine out-of-sample test", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_is_vs_oos.png", dpi=200)
    plt.close(fig)

    print(f"Saved fig2/fig3/fig4 PNGs to {FIG_DIR}")
    print("\nSection 13 done.")


if __name__ == "__main__":
    main()
