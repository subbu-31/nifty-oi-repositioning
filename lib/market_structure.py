"""
Section 3 of the original notebook: the Market Structure Engine.

A swing detector built for this project: a dynamic, ZigZag-style algorithm
with no fixed lookback window. A new swing high/low is confirmed when price
reverses by a fixed percentage threshold from the running extreme of the
current leg. Break of Structure (BoS) is a close beyond the last confirmed
swing in the direction of the prevailing trend; Change of Character (CHoCH)
is the same event when it reverses the prevailing trend. Runs as one
continuous series with no intraday reset (BoS/CHoCH are multi-bar/multi-day
structural concepts).

Only this swing/BoS/CHoCH logic is implemented -- related constructs such as
Order Blocks, Fair Value Gaps, and Premium/Discount zones, common to similar
swing-based technical-analysis tools, aren't needed to test whether OI is
reactive to these events, so they're omitted.

`run_engine_adaptive` (Section 19) is the same algorithm with a per-bar
threshold array instead of one fixed scalar, used for the vol-adaptive
break-threshold robustness check.
"""
import numpy as np
import pandas as pd

from lib.config import STRIKE_BAND


def strikes_near(spot_price, band=STRIKE_BAND):
    lo = round((spot_price - band) / 50) * 50
    hi = round((spot_price + band) / 50) * 50
    return list(range(int(lo), int(hi) + 1, 50))


def run_market_structure_engine(df, threshold_pct):
    """
    df: DataFrame with columns [date, open, high, low, close], sorted ascending,
        one continuous bar series (no gaps introduced).
    threshold_pct: swing reversal threshold in percent (e.g. 0.25 or 0.30).

    Returns (swings_df, events_df).
    """
    thr = threshold_pct / 100.0
    dates = df["date"].to_numpy()
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    leg_direction, leg_extreme_px, leg_extreme_idx = None, None, None
    high1 = high2 = low1 = low2 = None
    trend = "undefined"
    last_swing_high = last_swing_low = None
    bos_flagged_high = bos_flagged_low = None

    swings, events = [], []

    for i in range(n):
        if leg_direction is None:
            leg_direction = "up"
            leg_extreme_px, leg_extreme_idx = h[i], i

        new_swing_high = new_swing_low = False

        if leg_direction == "up":
            if h[i] > leg_extreme_px:
                leg_extreme_px, leg_extreme_idx = h[i], i
            elif c[i] <= leg_extreme_px * (1 - thr):
                new_swing_high = True
                swing_high_px, swing_high_idx = leg_extreme_px, leg_extreme_idx
                high1, high2 = high2, leg_extreme_px
                label = "First Swing" if high1 is None else ("HH" if high2 > high1 else "LH")
                last_swing_high = leg_extreme_px
                bos_flagged_high = None
                leg_direction = "down"
                leg_extreme_px, leg_extreme_idx = l[i], i
        else:
            if l[i] < leg_extreme_px:
                leg_extreme_px, leg_extreme_idx = l[i], i
            elif c[i] >= leg_extreme_px * (1 + thr):
                new_swing_low = True
                swing_low_px, swing_low_idx = leg_extreme_px, leg_extreme_idx
                low1, low2 = low2, leg_extreme_px
                label = "First Swing" if low1 is None else ("HL" if low2 > low1 else "LL")
                last_swing_low = leg_extreme_px
                bos_flagged_low = None
                leg_direction = "up"
                leg_extreme_px, leg_extreme_idx = h[i], i

        if new_swing_high:
            swings.append({"idx": swing_high_idx, "date": dates[swing_high_idx],
                            "price": swing_high_px, "type": "high", "label": label})
        if new_swing_low:
            swings.append({"idx": swing_low_idx, "date": dates[swing_low_idx],
                            "price": swing_low_px, "type": "low", "label": label})

        bullish_break = (last_swing_high is not None and c[i] > last_swing_high
                          and bos_flagged_high != last_swing_high)
        bearish_break = (last_swing_low is not None and c[i] < last_swing_low
                          and bos_flagged_low != last_swing_low)

        if bullish_break:
            bos_flagged_high = last_swing_high
            event_type = "CHoCH" if trend == "bearish" else "BoS"
            prior_trend, trend = trend, "bullish"
            events.append({"idx": i, "date": dates[i], "close": c[i], "event_type": event_type,
                            "direction": "bullish", "prior_trend": prior_trend, "new_trend": trend,
                            "broken_level": bos_flagged_high})

        if bearish_break:
            bos_flagged_low = last_swing_low
            event_type = "CHoCH" if trend == "bullish" else "BoS"
            prior_trend, trend = trend, "bearish"
            events.append({"idx": i, "date": dates[i], "close": c[i], "event_type": event_type,
                            "direction": "bearish", "prior_trend": prior_trend, "new_trend": trend,
                            "broken_level": bos_flagged_low})

    return pd.DataFrame(swings), pd.DataFrame(events)


def run_engine_adaptive(df, thr_pct_array):
    """
    Same algorithm as run_market_structure_engine, but with a per-bar
    threshold array (fraction of price, already in percent units before
    division by 100 here) instead of one fixed scalar. Used by the
    vol-adaptive break-threshold robustness check (Section 19). Returns
    events_df only (no swings_df -- not needed by that check).
    """
    thr_frac = thr_pct_array / 100.0
    dates = df["date"].to_numpy()
    h, l, c = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float), df["close"].to_numpy(dtype=float)
    n = len(df)
    leg_direction = leg_extreme_px = leg_extreme_idx = None
    high1 = high2 = low1 = low2 = None
    trend = "undefined"
    last_swing_high = last_swing_low = bos_flagged_high = bos_flagged_low = None
    events = []
    for i in range(n):
        thr = thr_frac[i]
        if leg_direction is None:
            leg_direction = "up"; leg_extreme_px, leg_extreme_idx = h[i], i
        new_swing_high = new_swing_low = False
        if leg_direction == "up":
            if h[i] > leg_extreme_px:
                leg_extreme_px, leg_extreme_idx = h[i], i
            elif c[i] <= leg_extreme_px * (1 - thr):
                new_swing_high = True
                swing_high_px = leg_extreme_px
                high1, high2 = high2, leg_extreme_px
                label = "First Swing" if high1 is None else ("HH" if high2 > high1 else "LH")
                last_swing_high = leg_extreme_px; bos_flagged_high = None
                leg_direction = "down"; leg_extreme_px, leg_extreme_idx = l[i], i
        else:
            if l[i] < leg_extreme_px:
                leg_extreme_px, leg_extreme_idx = l[i], i
            elif c[i] >= leg_extreme_px * (1 + thr):
                new_swing_low = True
                swing_low_px = leg_extreme_px
                low1, low2 = low2, leg_extreme_px
                label = "First Swing" if low1 is None else ("HL" if low2 > low1 else "LL")
                last_swing_low = leg_extreme_px; bos_flagged_low = None
                leg_direction = "up"; leg_extreme_px, leg_extreme_idx = h[i], i
        bullish_break = (last_swing_high is not None and c[i] > last_swing_high and bos_flagged_high != last_swing_high)
        bearish_break = (last_swing_low is not None and c[i] < last_swing_low and bos_flagged_low != last_swing_low)
        if bullish_break:
            bos_flagged_high = last_swing_high
            event_type = "CHoCH" if trend == "bearish" else "BoS"
            prior_trend, trend = trend, "bullish"
            events.append({"idx": i, "date": dates[i], "close": c[i], "event_type": event_type,
                            "direction": "bullish", "prior_trend": prior_trend, "new_trend": trend, "broken_level": bos_flagged_high})
        if bearish_break:
            bos_flagged_low = last_swing_low
            event_type = "CHoCH" if trend == "bullish" else "BoS"
            prior_trend, trend = trend, "bearish"
            events.append({"idx": i, "date": dates[i], "close": c[i], "event_type": event_type,
                            "direction": "bearish", "prior_trend": prior_trend, "new_trend": trend, "broken_level": bos_flagged_low})
    return pd.DataFrame(events)
