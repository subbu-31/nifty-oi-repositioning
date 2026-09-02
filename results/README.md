# Reference results

These three figures are the actual output of `scripts/13_figures.py`, generated
from the author's own 2025 in-sample / 2026 out-of-sample data. They're checked
into the repo so the headline results are visible without first sourcing and
running the full data pipeline — running `python run_all.py` against your own
data will regenerate byte-for-byte equivalent figures at `checkpoints/fig2_event_time_oi.png`
etc. (`13_figures.py`'s output directory).

## `fig2_event_time_oi.png` — H2, event-time OI response

PE-minus-CE open-interest differential in the minutes around a confirmed
structural break, split by break direction. Bullish and bearish breaks pull
the OI differential apart in the corresponding direction, and the separation
is already present *before* confirmation (t=0) and continues widening after
it — the shape underlying the paper's H2 result.

## `fig3_h2_correlation_curve.png` — H2, correlation curve

The point-biserial correlation between OI shift and break direction as a
function of event time, at both break thresholds (0.30% and 0.25%). The two
thresholds trace almost the same curve, rising from ~0.22 ten minutes before
confirmation to a peak of ~0.63 five minutes after it, then partially
declining — read together with Figure 2, this is the paper's evidence that
OI repositioning is a *reactive*, not anticipatory, response to the break.

## `fig4_is_vs_oos.png` — H3, in-sample vs. out-of-sample

The result the paper's abstract leads with. The H3 signal's classifier AUC
sits meaningfully above chance (0.50) both in-sample and out-of-sample —
but the backtest that actually trades it swings from +6,414 index points
in-sample (Jan-Jul 2025, the period its parameters were locked on) to -715
out-of-sample (Jan-Apr 2026, held out and untouched during tuning). A real
statistical association did not survive translation into a persistent
tradable edge.
