"""
Section 1 of the original reproduction notebook: configuration & data layout.

Adjust the two glob patterns and the checkpoint directory below to point at
your own data if you don't like the defaults. `CKPT` is where every
intermediate result gets cached as parquet/CSV/JSON -- each script checks
for its checkpoint file first and only recomputes if it's missing, so
re-running the pipeline after the first pass is fast, and each script below
can be run on its own once the scripts it depends on have populated the
checkpoints it needs (see the table in the top-level README).

Paths are resolved relative to the project root (the parent of this `lib/`
directory), not to the current working directory, so scripts run correctly
regardless of where you invoke them from.
"""
import os
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 160)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- raw data locations (edit these, or override via environment variables)
DATA_DIR = Path(os.environ.get("OI_REPRO_DATA_DIR", PROJECT_ROOT / "data"))
SPOT_2025_JSON_GLOB = str(DATA_DIR / "spot_2025" / "*.json")            # Kite historical_data exports
SPOT_2026_JSON_GLOB = str(DATA_DIR / "spot_2026_jan_apr" / "*.json")
OI_WEEKS_2025_GLOB = str(DATA_DIR / "oi_extracted" / "2025*")           # one dir per weekly expiry
OI_WEEKS_2026_GLOB = str(DATA_DIR / "oos_weeks_2026" / "*")

# ---- checkpoint directory (all intermediate + final outputs cached here) ----
CKPT = Path(os.environ.get("OI_REPRO_CKPT_DIR", PROJECT_ROOT / "checkpoints"))
CKPT.mkdir(parents=True, exist_ok=True)


def ckpt_path(name):
    return str(CKPT / name)


# ---- study constants, fixed throughout (see paper Sections 3-4) ------------
STRIKE_BAND = 150          # +/-150 points around spot, headline choice (robustness in §5.4)
H1_AFTER_WINDOW_MIN = 5    # H1/H2 "after" snapshot, headline choice (robustness in §5.5)
H2_BASELINE_MIN = 15       # H2 baseline offset before confirmation
H2_OFFSETS_MIN = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
H3_ROLL_MIN = 5            # H3 rolling PE-CE shift window
H3_FWD_HORIZON_MIN = 15    # H3 forward-looking label horizon / holding period
BREAK_THRESHOLDS_PCT = [0.25, 0.30]   # Market Structure Engine reversal thresholds tested
