"""
Metrics module for evaluating daily long/short trading strategies.
Consistent semantics:
  r_t = (close - open) / open           # asset open→close return
  a_t ∈ {−1, 0, +1}                     # position short/flat/long
  R_t = a_t * r_t                        # strategy daily return
  E_t = Π_{i≤t} (1 + R_i)                # equity curve (factor)
"""

import pandas as pd
import numpy as np
from typing import Any, Sequence, TypedDict, Dict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# === TypedDicts for return schemas ===
class ProfitabilityDict(TypedDict):
    cpp: float
    cpp_hyst: list[float]
    spp: float
    spp_hyst: list[float]

class StreaksDict(TypedDict):
    streak_no_errors: int
    streak_no_correct: int
    max_drawup_streak: float
    max_drawdown_streak: float

class MainMetricsDict(TypedDict):
    Sharpe: float
    Sortino: float
    max_drawdown: float
    Calmar: float

class NamedMetricsDict(TypedDict):
    total_return: float
    CAGR: float
    annual_vol: float
    hit_rate_daily: float
    profit_factor_daily: float
    turnover: float
    exposure: float
    n_trades: int
    win_rate_trades: float
    payoff_ratio: float
    profit_factor_trades: float
    expectancy_per_trade: float
    avg_holding_period: float
    information_ratio: float

class PerformanceSummary(TypedDict):
    profitability: ProfitabilityDict
    streaks: StreaksDict
    main_metrics: MainMetricsDict
    classification: Dict[str, float]
    named_metrics: NamedMetricsDict


def complicated_percent_profit(ts: pd.DataFrame, preds: Sequence[float], threshold: float = 0.5) -> dict:
    """
    Long-only, all-in/all-out profitability ("complicated percentage").
    When signal ≥ threshold we hold qty = bank/open, otherwise we hold cash.
    Equity is recorded at each open; any open position is liquidated at the final close.

    Returns
    -------
    dict with keys:
      - 'cpp': final capital factor relative to initial bank
      - 'cpp_hyst': equity curve (list of floats)
    """
    # validation of input data
    assert len(ts) == len(preds)
    if not {'open', 'close'}.issubset(ts.columns):
        raise ValueError("ts must contain 'open' and 'close' columns")

    # initialize trading variables
    initial_price = float(ts.iloc[0]['open'])
    initial_bank = 10.0 * initial_price  # scale for stability

    # simulation variables
    bank = initial_bank
    qty = 0.0
    in_pos = False
    equity_curve = []

    # start main loop of simulator over time
    for idx, p in enumerate(preds):
        open_price = float(ts.iloc[idx]['open'])

        # we buy/sell currency by open price at the start of the day
        if (p >= threshold) and not in_pos:
            qty = bank / open_price
            bank = 0.0
            in_pos = True
        elif (p < threshold) and in_pos:
            bank = qty * open_price
            qty = 0.0
            in_pos = False

        # record current equity
        equity_curve.append(qty * open_price if in_pos else bank)

    # close at the final close
    if in_pos:
        final_close = float(ts.iloc[-1]['close'])  # fixed indexing bug
        bank = qty * final_close
        equity_curve[-1] = bank  # reflect liquidation at last point

    return {
        'cpp': bank / initial_bank,
        'cpp_hyst': equity_curve
    }


def simple_percent_profit(ts: pd.DataFrame, preds: Sequence[float], threshold: float = 0.5, compound: bool = True) -> dict:
    """
    Day-trading long/short profitability ("simple percentage").
    Asset return r_t = (close - open)/open. Position a_t = +1 if pred ≥ threshold else −1.
    If compound=True: equity_t = Π (1 + a_t * r_t). Otherwise: equity_t = 1 + Σ (a_t * r_t).

    Returns
    -------
    dict with keys:
      - 'spp': final equity factor (compounded if compound=True)
      - 'spp_hyst': equity curve (list of floats)
    """
    assert len(ts) == len(preds)
    if not {'open', 'close'}.issubset(ts.columns):
        raise ValueError("ts must contain 'open' and 'close' columns")

    equity = 1.0
    curve = []

    for idx, pred in enumerate(preds):
        open_price = float(ts.iloc[idx]['open'])
        close_price = float(ts.iloc[idx]['close'])
        r = (close_price - open_price) / open_price
        a = 1.0 if pred >= threshold else -1.0

        if compound:
            equity *= (1.0 + a * r)
        else:
            equity += (a * r)

        curve.append(equity)

    return {
        'spp': equity,
        'spp_hyst': curve
    }

def get_classification_metrics(ts: pd.DataFrame, preds: Sequence[bool]) -> dict:
    """
    Binary classification metrics for the daily direction sign.
    Ground truth: open > close (down day = 1). Predictions are boolean for the same event.
    """
    assert len(ts) == len(preds)
    if not {'open', 'close'}.issubset(ts.columns):
        raise ValueError("ts must contain 'open' and 'close' columns")

    y_true = (ts['open'] > ts['close']).astype(int).to_numpy()
    y_pred = np.array(preds, dtype=int)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }

# Backward-compatible alias (original misspelling retained)
get_classification_metrices = get_classification_metrics
# === Small helpers for summarize_performance ===
def _profitability_from_actions(ts: pd.DataFrame, actions: Sequence[float]) -> ProfitabilityDict:
    """Build cpp/spp metrics from actions ∈ {−1,+1} keeping prior semantics."""
    cpp_preds = [1.0 if v > 0 else 0.0 for v in actions]
    cpp_out = complicated_percent_profit(ts, cpp_preds, threshold=0.5)

    spp_preds = [1.0 if v > 0 else 0.0 for v in actions]
    spp_out = simple_percent_profit(ts, spp_preds, threshold=0.5, compound=True)

    return {
        'cpp': cpp_out['cpp'],
        'cpp_hyst': list(map(float, cpp_out['cpp_hyst'])),
        'spp': spp_out['spp'],
        'spp_hyst': list(map(float, spp_out['spp_hyst'])),
    }


def _streaks_from_actions(ts: pd.DataFrame, actions: Sequence[float], strat_returns: np.ndarray) -> StreaksDict:
    """Compute streak metrics: longest correct/incorrect runs and max positive/negative simple sums."""
    y_true = (ts['open'] > ts['close']).to_numpy()
    y_pred = np.array([1 if v < 0 else 0 for v in actions], dtype=int)
    correct_flags = (y_true == y_pred)

    max_streak_no_errors = _longest_streak_bool(correct_flags, True)
    max_streak_no_correct = _longest_streak_bool(correct_flags, False)

    max_simple_increase_delta = _max_positive_run_sum(strat_returns)
    max_simple_decrease_delta = abs(_min_negative_run_sum(strat_returns))

    return {
        'streak_no_errors': int(max_streak_no_errors),
        'streak_no_correct': int(max_streak_no_correct),
        'max_drawup_streak': float(max_simple_increase_delta),
        'max_drawdown_streak': float(max_simple_decrease_delta),
    }



# === Generic strategy evaluation helpers ===

def _coerce_actions(actions: Sequence[float]) -> np.ndarray:
    """Map actions to positions a_t in {-1, 0, +1} as float array.
    Accepts booleans (True->+1, False->-1), ints, floats; values are clipped to [-1,1]
    and then snapped to -1, 0, or +1 via sign with tolerance.
    """
    a = np.asarray(actions, dtype=float)
    # Treat very small magnitudes as flat (0)
    a[np.abs(a) < 1e-12] = 0.0
    a = np.clip(a, -1.0, 1.0)
    # Normalize exact -1/0/+1 for cleanliness
    a = np.sign(a)
    return a


def _daily_returns_open_to_close(ts: pd.DataFrame) -> np.ndarray:
    """Compute daily raw asset returns r_t = (close - open)/open as numpy array."""
    o = ts['open'].to_numpy(dtype=float)
    c = ts['close'].to_numpy(dtype=float)
    return (c - o) / o


def _equity_curve(returns: np.ndarray) -> np.ndarray:
    """Cumulative equity curve from daily returns (geometric compounding)."""
    return np.cumprod(1.0 + returns, dtype=float)


def _drawdown_curve(equity: np.ndarray) -> tuple[np.ndarray, float]:
    """Return drawdown series and max drawdown (as negative number)."""
    peaks = np.maximum.accumulate(equity)
    dd = equity / peaks - 1.0
    mdd = float(dd.min()) if dd.size else 0.0
    return dd, mdd


def _annualization_factor(ts: pd.DataFrame, trading_days_per_year: int = 252) -> float:
    """Infer annualization based on calendar span if datetime index/column exists; fallback to trading_days_per_year."""
    if hasattr(ts.index, 'inferred_type') and 'datetime' in str(ts.index.inferred_type):
        dt0 = ts.index[0]
        dt1 = ts.index[-1]
    elif 'date' in ts.columns and np.issubdtype(ts['date'].dtype, np.datetime64):
        dt0 = ts['date'].iloc[0]
        dt1 = ts['date'].iloc[-1]
    else:
        return float(trading_days_per_year)
    days = max((dt1 - dt0).days, 1)
    years = days / 365.25
    if years <= 0:
        return float(trading_days_per_year)
    return len(ts) / years


def evaluate_strategy_timeseries(
    ts: pd.DataFrame,
    actions: Sequence[float],
    risk_free_rate_annual: float = 0.0,
    trading_days_per_year: int = 252,
    target_return_annual: float = 0.0,
) -> dict:
    """
    Build key time series for a daily long/short/flat strategy.

    Parameters
    ----------
    ts : DataFrame with columns ['open','close'] (and optional datetime index)
    actions : Sequence with values in {-1,0,+1} (booleans are mapped to {+1,-1})
    risk_free_rate_annual : annualized risk-free rate (e.g., 0.03 for 3%)
    trading_days_per_year : used for annualization if calendar time not available
    target_return_annual : target/MAR used in Sortino (0 by default)

    Returns dict with arrays: r (asset returns), a (positions), ret (strategy daily returns),
    equity, drawdown, and scalars with daily equivalents of risk-free rate and target.
    """
    assert len(ts) == len(actions), "ts and actions must have the same length"
    if not {'open', 'close'}.issubset(ts.columns):
        raise ValueError("ts must contain 'open' and 'close' columns")

    a = _coerce_actions(actions)
    r = _daily_returns_open_to_close(ts)
    strat_r = a * r

    eq = _equity_curve(strat_r)
    dd, mdd = _drawdown_curve(eq)

    ann_fac = _annualization_factor(ts, trading_days_per_year)
    rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / ann_fac) - 1.0
    tgt_daily = (1.0 + target_return_annual) ** (1.0 / ann_fac) - 1.0

    return {
        'r': r,
        'a': a,
        'ret': strat_r,
        'equity': eq,
        'drawdown': dd,
        'max_drawdown': mdd,
        'annualization_factor': ann_fac,
        'risk_free_daily': rf_daily,
        'target_daily': tgt_daily,
    }


def _downside_deviation(excess_returns: np.ndarray) -> float:
    """Downside deviation (root mean square of negative excess returns)."""
    neg = np.minimum(excess_returns, 0.0)
    return float(np.sqrt(np.mean(neg * neg)))


# === Additional helper functions for main metrics ===
def _longest_streak_bool(flags: np.ndarray, target: bool) -> int:
    """Return length of the longest consecutive run where flags == target."""
    longest = 0
    cur = 0
    tgt = bool(target)
    for v in flags.astype(bool):
        if v == tgt:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return int(longest)


def _max_positive_run_sum(x: np.ndarray) -> float:
    """Maximum sum over all contiguous segments with strictly positive elements.
    Returns 0.0 if there is no positive element.
    """
    best = 0.0
    cur = 0.0
    for v in x:
        if v > 0:
            cur += float(v)
            if cur > best:
                best = cur
        else:
            cur = 0.0
    return float(best)


def _min_negative_run_sum(x: np.ndarray) -> float:
    """Minimum (most negative) sum over all contiguous segments with strictly negative elements.
    Returns 0.0 if there is no negative element.
    """
    worst = 0.0
    cur = 0.0
    for v in x:
        if v < 0:
            cur += float(v)
            if cur < worst:
                worst = cur
        else:
            cur = 0.0
    return float(worst)


def segment_trades(actions: Sequence[float], returns: Sequence[float]) -> list[dict]:
    """Segment contiguous non-zero positions into trades and compute trade-level returns.
    Each trade return is geometric over its segment: prod(1 + a_t*r_t) - 1.
    Returns a list of dicts with keys: 'start', 'end', 'length', 'position', 'ret'.
    """
    a = _coerce_actions(actions)
    r = np.asarray(returns, dtype=float)
    trades = []
    f = 1.0
    length = 0
    pos = 0.0
    start = None

    for t in range(len(a)):
        if a[t] == 0.0:
            # If we are in a trade and now flat, close the trade
            if pos != 0.0:
                trades.append({'start': start, 'end': t-1, 'length': length, 'position': pos, 'ret': f - 1.0})
                f, length, pos, start = 1.0, 0, 0.0, None
            continue
        # If entering a new trade or flipping side
        if pos == 0.0 or np.sign(a[t]) != np.sign(pos):
            if pos != 0.0:
                # close previous trade before flipping
                trades.append({'start': start, 'end': t-1, 'length': length, 'position': pos, 'ret': f - 1.0})
            f, length, pos, start = 1.0, 0, a[t], t
        # Accumulate within trade
        f *= (1.0 + a[t] * r[t])
        length += 1

    # close at end if in trade
    if pos != 0.0:
        trades.append({'start': start, 'end': len(a)-1, 'length': length, 'position': pos, 'ret': f - 1.0})

    return trades


def summarize_performance(
    ts: pd.DataFrame,
    actions: Sequence[float],
    risk_free_rate_annual: float = 0.0,
    trading_days_per_year: int = 252,
    target_return_annual: float = 0.0,
    benchmark_returns: Sequence[float] | None = None,
) -> dict:
    """
    Compute a comprehensive set of performance metrics for a daily strategy.

    Returns
    -------
    PerformanceSummary
    """
    ts_data = evaluate_strategy_timeseries(
        ts, actions,
        risk_free_rate_annual=risk_free_rate_annual,
        trading_days_per_year=trading_days_per_year,
        target_return_annual=target_return_annual,
    )

    r = ts_data['r']
    a = ts_data['a']
    sr = ts_data['ret']
    eq = ts_data['equity']
    dd = ts_data['drawdown']
    mdd = ts_data['max_drawdown']
    ann = ts_data['annualization_factor']
    rf_d = ts_data['risk_free_daily']
    tgt_d = ts_data['target_daily']

    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    years = len(ts) / ann if ann > 0 else 0.0
    CAGR = float((eq[-1]) ** (1.0 / years) - 1.0) if (eq.size and years > 0) else 0.0

    # Risk metrics
    mean_excess = float(np.mean(sr - rf_d)) if sr.size else 0.0
    std_daily = float(np.std(sr, ddof=1)) if sr.size > 1 else 0.0
    annual_vol = std_daily * np.sqrt(ann) if std_daily > 0 else 0.0
    Sharpe = (mean_excess / std_daily) * np.sqrt(ann) if std_daily > 0 else np.nan
    Sortino = np.nan
    if sr.size:
        ddn = _downside_deviation(sr - tgt_d)
        Sortino = (np.mean(sr) - tgt_d) / ddn if ddn > 0 else np.nan
    Calmar = CAGR / abs(mdd) if (abs(mdd) > 1e-12 and CAGR != 0) else np.nan

    # Daily trade-quality metrics
    wins = sr[sr > 0]
    losses = sr[sr < 0]
    hit_rate_daily = float((sr > 0).mean()) if sr.size else 0.0
    profit_factor_daily = float(wins.sum() / abs(losses.sum())) if losses.size else np.inf

    # Position dynamics
    exposure = float((np.abs(a) > 0).mean()) if a.size else 0.0
    turnover = float((np.abs(np.diff(a)) > 0).mean()) if a.size > 1 else 0.0

    # Trade segmentation metrics
    trades = segment_trades(a, r)
    n_trades = len(trades)
    if n_trades:
        tr_rets = np.array([t['ret'] for t in trades], dtype=float)
        tr_lens = np.array([t['length'] for t in trades], dtype=int)
        win_mask = tr_rets > 0
        loss_mask = tr_rets < 0
        win_rate_trades = float(win_mask.mean())
        avg_win = float(tr_rets[win_mask].mean()) if win_mask.any() else np.nan
        avg_loss = float(np.abs(tr_rets[loss_mask].mean())) if loss_mask.any() else np.nan
        payoff_ratio = (avg_win / avg_loss) if (avg_win and avg_loss and avg_loss > 0) else np.nan
        profit_factor_trades = float(tr_rets[win_mask].sum() / np.abs(tr_rets[loss_mask].sum())) if loss_mask.any() else np.inf
        expectancy_per_trade = float(tr_rets.mean())
        avg_holding_period = float(tr_lens.mean())
    else:
        win_rate_trades = np.nan
        payoff_ratio = np.nan
        profit_factor_trades = np.nan
        expectancy_per_trade = np.nan
        avg_holding_period = np.nan

    # Information ratio if benchmark provided (active return / tracking error)
    information_ratio = np.nan
    if benchmark_returns is not None:
        b = np.asarray(benchmark_returns, dtype=float)
        if b.shape[0] == sr.shape[0]:
            active = sr - b
            te = float(np.std(active, ddof=1)) if active.size > 1 else 0.0
            information_ratio = float(np.mean(active) / te * np.sqrt(ann)) if te > 0 else np.nan

    # Top-level metrics
    profitability: ProfitabilityDict = _profitability_from_actions(ts, actions)
    classification = get_classification_metrics(ts, [True if v < 0 else False for v in actions])
    streaks: StreaksDict = _streaks_from_actions(ts, actions, sr)

    # main metrics
    main_metrics = {
        'Sharpe': Sharpe,
        'Sortino': Sortino,
        'max_drawdown': mdd,
        'Calmar': Calmar,
    }

    # Package everything else under 'additional'
    named_metrics = {
        'total_return': total_return,
        'CAGR': CAGR,
        'annual_vol': annual_vol,
        'hit_rate_daily': hit_rate_daily,
        'profit_factor_daily': profit_factor_daily,
        'turnover': turnover,
        'exposure': exposure,
        'n_trades': n_trades,
        'win_rate_trades': win_rate_trades,
        'payoff_ratio': payoff_ratio,
        'profit_factor_trades': profit_factor_trades,
        'expectancy_per_trade': expectancy_per_trade,
        'avg_holding_period': avg_holding_period,
        'information_ratio': information_ratio,
    }

    return {
        'profitability': profitability,
        'streaks': streaks,
        'main_metrics': main_metrics,
        'classification': classification,
        'named_metrics': named_metrics,
    }
