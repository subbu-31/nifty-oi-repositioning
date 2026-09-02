"""
As-of (forward-fill) OI/price/volume lookup helpers shared across sections.

OI is an exchange-reported snapshot taken at the minute of each trade, not a
continuous per-minute feed, so every lookup here returns the most recent
trade at or before the requested timestamp (0 / NaN if the strike/leg never
traded yet in the active contract -- see the Section 12 and 27 diagnostics
for how often that actually happens and whether it's concerning).
"""
import numpy as np


def asof_oi_from_groups(groups, strike, opt_type, ts):
    """groups: {(strike, opt_type): (timestamp_array, oi_array)}."""
    key = (strike, opt_type)
    if key not in groups:
        return 0.0
    tarr, oarr = groups[key]
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    idx = np.searchsorted(tarr, np.datetime64(ts), side="right") - 1
    return float(oarr[idx]) if idx >= 0 else 0.0


def asof_oi_expiry(groups, expiry, strike, opt_type, ts):
    """Same as asof_oi_from_groups but groups are keyed by (expiry, strike, opt_type)."""
    key = (expiry, strike, opt_type)
    if key not in groups:
        return 0.0
    tarr, oarr = groups[key]
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    idx = np.searchsorted(tarr, np.datetime64(ts), side="right") - 1
    return float(oarr[idx]) if idx >= 0 else 0.0


def pc_diff_at(groups, strikes, ts):
    """Sum of PE OI minus sum of CE OI across `strikes`, as-of `ts`."""
    pe_total = ce_total = 0.0
    for k in strikes:
        pe_total += asof_oi_from_groups(groups, k, "PE", ts)
        ce_total += asof_oi_from_groups(groups, k, "CE", ts)
    return pe_total - ce_total


def vectorized_asof(tarr, oarr, targets_naive):
    idx = np.searchsorted(tarr, targets_naive, side="right") - 1
    return np.where(idx >= 0, oarr[np.clip(idx, 0, len(oarr) - 1)], 0.0)


def next_event_within_vec(times_sorted, targets, horizon_minutes):
    idx = np.searchsorted(times_sorted, targets, side="right")
    valid = idx < len(times_sorted)
    next_t = np.where(valid, times_sorted[np.clip(idx, 0, len(times_sorted) - 1)],
                       np.datetime64("2100-01-01"))
    return valid & (next_t <= targets + np.timedelta64(horizon_minutes, "m"))


def classify_oi_lookup(groups, strike, opt_type, ts):
    key = (strike, opt_type)
    if key not in groups:
        return "never_traded_in_archive"
    tarr, _ = groups[key]
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    idx = np.searchsorted(tarr, np.datetime64(ts), side="right") - 1
    return "traded_later_this_week_not_yet" if idx < 0 else "has_prior_trade"


def windowed_volume_sum(groups, strike, opt_type, t0, t1):
    """Sum of the per-bar `volume` flow quantity over [t0, t1) -- unlike OI
    (a cumulative running total, safe to point-sample as-of a timestamp),
    volume must be summed over the window rather than snapshotted."""
    key = (strike, opt_type)
    if key not in groups:
        return 0.0
    tarr, varr = groups[key]
    i0 = np.searchsorted(tarr, np.datetime64(t0), side="left")
    i1 = np.searchsorted(tarr, np.datetime64(t1), side="left")
    return float(varr[i0:i1].sum()) if i1 > i0 else 0.0


def asof_price_oi(groups, strike, opt_type, ts_naive):
    """groups: {(strike, opt_type): (timestamp_array, close_array, oi_array)}.
    Returns (price, oi) as of ts_naive, or (nan, 0.0) if never traded yet."""
    key = (strike, opt_type)
    if key not in groups:
        return np.nan, 0.0
    tarr, carr, oarr = groups[key]
    idx = np.searchsorted(tarr, np.datetime64(ts_naive), side="right") - 1
    if idx < 0:
        return np.nan, 0.0
    return float(carr[idx]), float(oarr[idx])
