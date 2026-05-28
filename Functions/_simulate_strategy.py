"""Long/flat backtesting strategy, ported from Model_Case_Scenario."""

import numpy as np
import pandas as pd


def _simulate_strategy(
    timestamps, y_pred, gold_close, stride,
    starting_cash=1000.0,
):
    """Simulate a long/flat strategy driven by the sign of y_pred.

    Decisions are made once every *stride* bars; the position is held for
    exactly *stride* bars before the next decision.  Returns a dict with the
    equity curve, terminal value, trade count, and time-in-market fraction.
    """
    timestamps = pd.DatetimeIndex(timestamps)
    y_pred     = np.asarray(y_pred, dtype=float)
    n          = len(timestamps)
    decision_idx = np.arange(0, n - stride, stride, dtype=int)
    if len(decision_idx) == 0:
        raise ValueError(f'Not enough timestamps ({n}) for stride={stride}')
    decision_times = timestamps[decision_idx]
    next_times     = timestamps[decision_idx + stride]
    decision_preds = y_pred[decision_idx]

    closes      = gold_close.reindex(decision_times).to_numpy(dtype=float)
    closes_next = gold_close.reindex(next_times).to_numpy(dtype=float)
    valid = ~(np.isnan(closes) | np.isnan(closes_next))
    if not valid.all():
        decision_times = decision_times[valid]
        next_times     = next_times[valid]
        decision_preds = decision_preds[valid]
        closes         = closes[valid]
        closes_next    = closes_next[valid]

    realised  = np.log(closes_next) - np.log(closes)
    positions = (decision_preds > 0).astype(int)

    strategy_logrets = positions * realised
    equity_strategy  = starting_cash * np.exp(np.cumsum(strategy_logrets))
    return {
        'decision_times':  decision_times,
        'equity_strategy': equity_strategy,
        'terminal':        float(equity_strategy[-1]),
        'positions':       positions,
        'n_trades':        int((np.diff(np.concatenate(([0], positions))) != 0).sum()),
        'pct_time_long':   float(positions.mean()),
    }
