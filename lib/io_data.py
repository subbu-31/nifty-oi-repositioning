"""
Section 2a/2b of the original notebook: data ingestion.

Generic loaders for a directory of Kite-style `historical_data` JSON exports
(spot bars) and for a weekly options-chain archive (one CSV per traded
(strike, option_type) leg). OI values are exchange-reported snapshots
recorded at the minute of each trade -- not a continuous per-minute feed --
which is why every downstream lookup in this project uses an *as-of*
(forward-fill) convention rather than assuming a dense per-minute series
(see the paper's Data section, "OI construction and missing-data handling").
"""
import glob
import json
import os
import re
from pathlib import Path

import pandas as pd

from lib.config import ckpt_path

STRIKE_RE = re.compile(r"^(\d+)(CE|PE)_(\d{8})\.csv$")


def load_spot_from_json(json_glob, out_name, expected_days_note=""):
    ckpt = ckpt_path(out_name)
    if os.path.exists(ckpt):
        print(f"[cache hit] {ckpt}")
        return pd.read_parquet(ckpt)

    files = sorted(glob.glob(json_glob))
    print(f"Found {len(files)} JSON files matching {json_glob}")
    all_rows = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if not data:
            continue
        dates = [row["date"] for row in data]
        print(f"  {os.path.basename(f)}: {len(data)} rows, {min(dates)} -> {max(dates)}")
        all_rows.extend(data)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    before = len(df)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    print(f"Rows before/after dedup: {before} / {len(df)}")

    trading_date = df["date"].dt.date
    print(f"Distinct trading days: {trading_date.nunique()} {expected_days_note}")
    bar_counts = df.groupby(trading_date).size()
    print(f"Bar-count/day: min={bar_counts.min()}, median={bar_counts.median()}, max={bar_counts.max()}")
    low_days = bar_counts[bar_counts < 300]
    print("Days with <300 bars (possible half-days/gaps):")
    print(low_days.to_string() if len(low_days) else "  none")
    print(f"Nulls: {df.isnull().sum().sum()}, non-positive OHLC: "
          f"{(df[['open','high','low','close']] <= 0).any(axis=1).sum()}")

    df.to_parquet(ckpt, index=False)
    print(f"Saved: {ckpt}\n")
    return df


def load_options_week(week_dir):
    rows = []
    for f in Path(week_dir).glob("*.csv"):
        if f.name == "nifty_spot.csv":
            continue
        m = STRIKE_RE.match(f.name)
        if not m:
            continue
        strike, opt_type, expiry_str = m.groups()
        expiry = pd.to_datetime(expiry_str, format="%Y%m%d").date()
        df = pd.read_csv(f, usecols=["Timestamp", "Close", "Volume", "OI"])
        df["timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
        df["strike"] = int(strike)
        df["option_type"] = opt_type
        df["expiry"] = expiry
        df = df.rename(columns={"Close": "close", "Volume": "volume", "OI": "oi"})
        rows.append(df[["timestamp", "strike", "option_type", "expiry", "close", "volume", "oi"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parquet_row_count(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def load_options_period(week_glob, out_name):
    # IMPORTANT: does NOT return the loaded DataFrame -- a full year of options
    # data is ~35M rows and every downstream stage in this project reads it
    # back off disk in per-expiry batches (pyarrow filter pushdown) rather than
    # holding it all in memory at once. Keeping a 35M-row frame alive as a
    # long-lived variable on top of that is what OOM-kills a process that
    # per-expiry-safe scripts never have a problem with.
    ckpt = ckpt_path(out_name)
    if os.path.exists(ckpt):
        print(f"[cache hit] {ckpt} ({parquet_row_count(ckpt):,} rows, not loaded into memory)")
        return

    week_dirs = sorted(glob.glob(week_glob))
    all_weeks, summary = [], []
    for wd in week_dirs:
        wdf = load_options_week(wd)
        if len(wdf) == 0:
            print(f"{os.path.basename(wd)}: EMPTY")
            continue
        summary.append({"week": os.path.basename(wd), "rows": len(wdf),
                         "n_strikes": wdf["strike"].nunique(), "expiry": wdf["expiry"].iloc[0]})
        all_weeks.append(wdf)
        print(f"{os.path.basename(wd)}: {len(wdf)} rows, {wdf['strike'].nunique()} strikes, "
              f"expiry={wdf['expiry'].iloc[0]}")

    combined = pd.concat(all_weeks, ignore_index=True)
    print(f"\nTOTAL: {len(combined)} rows, {combined['expiry'].nunique()} expiries")
    combined.to_parquet(ckpt, index=False)
    print(f"Saved: {ckpt}")
    print(pd.DataFrame(summary).to_string(index=False))
    del combined, all_weeks
