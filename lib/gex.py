"""
Section 15 core builders: Black-Scholes gamma exposure (GEX) reframing.

Two constructs, both derived from Black-Scholes gamma (IV inverted from the
traded option `close` price at each snapshot; flat 6.5% rate, no dividend
adjustment -- reasonable simplifications at weekly tenors):
- `unsigned_gex` = sum(OI x gamma) over calls and puts alike (magnitude
  only, no assumption about who holds the OI)
- `dealer_gex` = -sum(OI_call x gamma_call) + sum(OI_put x gamma_put)
  (standard convention: dealers assumed net short calls / net long puts;
  positive = dealers long gamma = expected to dampen moves, negative =
  short gamma = expected to amplify moves)
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from lib.config import ckpt_path, H1_AFTER_WINDOW_MIN
from lib.market_structure import strikes_near
from lib.oi_lookup import asof_price_oi

GEX_R = 0.065          # flat annual risk-free rate (no term structure)
GEX_VOL_LO, GEX_VOL_HI = 1e-4, 4.0


def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt_type == "CE" else max(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return norm.pdf(d1) / (S * sigma * sqrtT)


def implied_vol(price, S, K, T, r, opt_type):
    if T <= 0 or price <= 0 or S <= 0 or K <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if opt_type == "CE" else max(K - S, 0.0)
    if price < intrinsic - 1e-6:
        return np.nan
    upper_bound = S if opt_type == "CE" else K * np.exp(-r * T)
    if price > upper_bound + 1e-6:
        return np.nan
    f_lo = bs_price(S, K, T, r, GEX_VOL_LO, opt_type) - price
    f_hi = bs_price(S, K, T, r, GEX_VOL_HI, opt_type) - price
    if f_lo > 0 or f_hi < 0:
        return np.nan
    try:
        return brentq(lambda sig: bs_price(S, K, T, r, sig, opt_type) - price,
                       GEX_VOL_LO, GEX_VOL_HI, xtol=1e-6, maxiter=100)
    except ValueError:
        return np.nan


def time_to_expiry_years(expiry_date, ts_naive):
    expiry_cutoff = pd.Timestamp(expiry_date) + pd.Timedelta(hours=15, minutes=30)
    return max((expiry_cutoff - ts_naive).total_seconds(), 0.0) / (365 * 86400)


def gex_snapshot(groups, strikes, expiry, ts_naive, spot_px):
    T = time_to_expiry_years(expiry, ts_naive)
    unsigned = dealer = 0.0
    n_ok = n_try = 0
    for k in strikes:
        for opt in ("CE", "PE"):
            price, oi = asof_price_oi(groups, k, opt, ts_naive)
            if oi <= 0 or np.isnan(price) or price <= 0:
                continue
            n_try += 1
            iv = implied_vol(price, spot_px, k, T, GEX_R, opt)
            if np.isnan(iv):
                continue
            g = bs_gamma(spot_px, k, T, GEX_R, iv)
            n_ok += 1
            unsigned += oi * g
            dealer += (-oi * g) if opt == "CE" else (oi * g)
    return unsigned, dealer, n_ok, n_try


def build_gex_events(events_df, options_path, tag, asof_spot_fn):
    """asof_spot_fn: callable(ts_naive) -> spot price, e.g. an as-of lookup
    against the same spot series the events were built from."""
    ckpt = ckpt_path(f"gex_events_{tag}.parquet")
    if os.path.exists(ckpt):
        return pd.read_parquet(ckpt)

    expiries_local = sorted(pd.read_parquet(options_path, columns=["expiry"])["expiry"].unique())

    def active_expiry_for(trading_date):
        for e in expiries_local:
            if e >= trading_date:
                return e
        return None

    ev = events_df.copy()
    ev["date_naive"] = pd.to_datetime(ev["date"]).dt.tz_localize(None)
    ev["trading_date_computed"] = ev["date_naive"].dt.date
    ev["active_expiry"] = ev["trading_date_computed"].apply(active_expiry_for)

    results = []
    for expiry, ev_batch in ev.groupby("active_expiry", sort=False):
        if expiry is None:
            continue
        batch = pd.read_parquet(options_path, filters=[("expiry", "=", expiry)])
        groups = {}
        for key, g in batch.groupby(["strike", "option_type"], sort=False):
            groups[key] = (g["timestamp"].to_numpy(), g["close"].to_numpy(), g["oi"].to_numpy())
        del batch
        for _, row in ev_batch.iterrows():
            spot_before = row["close"]
            before_t = row["date_naive"]
            after_t = before_t + pd.Timedelta(minutes=H1_AFTER_WINDOW_MIN)
            spot_after = asof_spot_fn(after_t)
            strikes = strikes_near(spot_before)
            u_b, d_b, ok_b, try_b = gex_snapshot(groups, strikes, expiry, before_t, spot_before)
            u_a, d_a, ok_a, try_a = gex_snapshot(groups, strikes, expiry, after_t, spot_after)
            results.append({
                **{k2: v2 for k2, v2 in row.to_dict().items()
                   if k2 not in ("trading_date_computed", "active_expiry")},
                "expiry": expiry, "spot_before": spot_before, "spot_after": spot_after,
                "n_inverted_before": ok_b, "n_attempted_before": try_b,
                "n_inverted_after": ok_a, "n_attempted_after": try_a,
                "unsigned_gex_before": u_b, "unsigned_gex_after": u_a, "unsigned_gex_shift": u_a - u_b,
                "dealer_gex_before": d_b, "dealer_gex_after": d_a, "dealer_gex_shift": d_a - d_b,
            })
        del groups
    out = pd.DataFrame(results)
    out.to_parquet(ckpt, index=False)
    return out
