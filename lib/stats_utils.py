"""Small statistical helpers shared across the robustness/inference sections."""
import numpy as np
from scipy import stats


def fisher_r_to_z_test(r1, n1, r2, n2):
    """Is the gap between two independent Pearson/point-biserial correlations
    itself significant? (Sections 21, 22, 24.)"""
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def block_bootstrap_ci(daily_pnl, n_boot=10000, seed=12345):
    rng = np.random.default_rng(seed)
    n = len(daily_pnl)
    means = np.array([rng.choice(daily_pnl, size=n, replace=True).mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])


def day_level_inference(trades_df, label):
    daily = trades_df.groupby("trading_date")["pnl"].sum()
    t_stat, p_val = stats.ttest_1samp(daily.to_numpy(), 0)
    ci_lo, ci_hi = block_bootstrap_ci(daily.to_numpy())
    print(f"===== {label} =====")
    print(f"  Trading days with >=1 trade: {len(daily)}")
    print(f"  Mean daily P&L: {daily.mean():.2f} pts/day")
    print(f"  Day-clustered one-sample t-test: t={t_stat:.3f}, p={p_val:.4g}")
    print(f"  Block-bootstrap 95% CI (10,000 resamples): [{ci_lo:.2f}, {ci_hi:.2f}]")
    return {"label": label, "n_days": len(daily), "mean_daily_pnl": daily.mean(),
            "t_stat": t_stat, "p_value": p_val, "boot_ci_lo": ci_lo, "boot_ci_hi": ci_hi}


def pointbiserial_fast(x, y):
    """Vectorized point-biserial correlation for the inner loop of a bootstrap
    (Section 24) -- avoids scipy's per-call overhead across 10,000 resamples."""
    xm, ym = x.mean(), y.mean()
    xd, yd = x - xm, y - ym
    den = np.sqrt(np.sum(xd ** 2) * np.sum(yd ** 2))
    return np.sum(xd * yd) / den if den > 0 else 0.0
