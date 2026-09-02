#!/usr/bin/env python3
"""
Section 26: Robustness -- placebo-matching-algorithm sensitivity check.

The magnitude-matched placebo comparison (Section 21) is a greedy
nearest-neighbor match with a specific tie-breaking order (largest real
moves matched first, without replacement). This checks whether the result
is sensitive to that specific choice: matching smallest-first, in random
order (20 seeds), and independently with replacement.

Depends on: 21_placebo_test.py (real_sorted.parquet, placebo_pool_thr0p3.parquet,
  placebo_summary_stats.json)
Produces: nothing new (prints only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import ckpt_path
from lib.placebo import load_filtered_placebo_pool


def _greedy_match(real_df, pool_df, order="desc", seed=None):
    if order == "desc":
        real_ord = real_df.sort_values("preceding_chg", key=lambda s: s.abs(), ascending=False)
    elif order == "asc":
        real_ord = real_df.sort_values("preceding_chg", key=lambda s: s.abs(), ascending=True)
    elif order == "random":
        real_ord = real_df.sample(frac=1, random_state=seed)
    avail = pool_df.copy()
    avail["used"] = False
    rows = []
    for _, rr in real_ord.iterrows():
        unused = avail[~avail["used"]]
        diffs = (unused["preceding_chg"] - rr["preceding_chg"]).abs()
        best_i = diffs.idxmin()
        rows.append(avail.loc[best_i])
        avail.loc[best_i, "used"] = True
    return pd.DataFrame(rows).reset_index(drop=True)


def main():
    real_sorted = pd.read_parquet(ckpt_path("real_sorted.parquet"))
    pool_result = load_filtered_placebo_pool()
    with open(ckpt_path("placebo_summary_stats.json")) as f:
        summary_stats = json.load(f)
    r_real_naive = summary_stats["r_real_naive"]
    r_matched = summary_stats["r_matched"]

    _match_real = real_sorted  # already built desc-sorted in Section 21; reuse directly
    _sens_results = {}
    for order in ("desc", "asc"):
        m = _greedy_match(_match_real, pool_result, order=order)
        r, p = stats.pointbiserialr(m["dir_bin"], m["pe_minus_ce_shift"])
        _sens_results[order] = r
        print(f"Matching order={order}: matched-placebo r={r:.4f}, p={p:.3e}")

    _random_rs = []
    for seed in range(20):
        m = _greedy_match(_match_real, pool_result, order="random", seed=seed)
        r, _ = stats.pointbiserialr(m["dir_bin"], m["pe_minus_ce_shift"])
        _random_rs.append(r)
    _random_rs = np.array(_random_rs)
    print(f"\nRandom matching order (20 seeds): mean r={_random_rs.mean():.4f}, std={_random_rs.std():.4f}, "
          f"range=[{_random_rs.min():.4f}, {_random_rs.max():.4f}]")

    _pool_chg = pool_result["preceding_chg"].to_numpy()
    _rows_wr = []
    for _, rr in _match_real.iterrows():
        j = np.argmin(np.abs(_pool_chg - rr["preceding_chg"]))
        _rows_wr.append(pool_result.iloc[j])
    _m_wr = pd.DataFrame(_rows_wr).reset_index(drop=True)
    r_wr, p_wr = stats.pointbiserialr(_m_wr["dir_bin"], _m_wr["pe_minus_ce_shift"])
    print(f"\nMatching WITH replacement (each real event independently nearest-matched): r={r_wr:.4f}, p={p_wr:.3e}")

    _all_variant_rs = list(_sens_results.values()) + [r_wr] + list(_random_rs)
    print(f"\nFor reference: real(naive label) r={r_real_naive:.4f}")
    print(f"Original (desc, no replacement) matched r={r_matched:.4f} -- range across all variants: "
          f"[{min(_all_variant_rs):.4f}, {max(_all_variant_rs):.4f}]")

    print("\nSection 26 done.")


if __name__ == "__main__":
    main()
