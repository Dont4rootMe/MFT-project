from matplotlib import pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np


def plot_candles_finplot(
    df,
    predictions=None,
    prediction_time=None,
    start=None,
    stop=None,
    coin_name=None,
    save_path=None
):
    """
    Visualize stock price candles with classification and regression predictions using finplot.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'open', 'high', 'low', 'close'
        predictions (array-like): Binary predictions (1=rise, 0/-1=fall)
        prediction_time (array-like): Indices corresponding to prediction points
        start: Start index for slicing data
        stop: Stop index for slicing data
        coin_name (str): Name of currency to investigate (optional, not used here)
        save_path (str): Path to save figure (optional)
    """
    # Validate input DataFrame
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Slice DataFrame if start/stop specified
    df_plot = df.loc[start:stop].copy()
    
    # Handle prediction_time if predictions exist
    if predictions is not None:
        # Convert to numpy arrays
        predictions = np.array(predictions)
            
        # Determine number of predictions
        n_pred = len(predictions)

        # Set default prediction_time if not provided
        if prediction_time is None:
            prediction_time = df.index[-n_pred:]
        else:
            prediction_time = np.array(prediction_time)
            if len(prediction_time) != n_pred:
                raise ValueError("prediction_time length must match predictions")
    else:
        prediction_time = None
    
    # mplfinance expects columns named ['Open','High','Low','Close'] in that order
    _df_mpf = df_plot.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'})
    _df_mpf = _df_mpf[['Open','High','Low','Close']]
    fig, axlist = mpf.plot(
        _df_mpf,
        type='candle',
        style='charles',
        volume=False,
        show_nontrading=False,
        returnfig=True,
    )
    initial_limits = axlist[0].get_xlim()
    
    # Adding title
    title  = coin_name if coin_name is not None else 'Stock'
    title += ' Price Candles'
    title += ' with Predictions' if predictions is not None else ''
    fig.suptitle(f'{title}')

    # Highlight predictions
    if predictions is not None:
        # Filter and align predictions within plot range (preserve order)
        valid_pairs = []
        for dt, pr in zip(prediction_time, predictions):
            ts_dt = pd.Timestamp(dt)
            if ts_dt in df_plot.index:
                valid_pairs.append((ts_dt, pr))
        if not valid_pairs:
            valid_times = []
            valid_preds = []
        else:
            valid_times, valid_preds = zip(*valid_pairs)

        for date, pred in zip(valid_times, valid_preds):
            # Integer bar position in mplfinance (one unit per bar)
            try:
                pos = df_plot.index.get_loc(date)
                if isinstance(pos, slice):
                    # pick the left boundary for safety
                    pos = pos.start
            except KeyError:
                continue
            x0 = pos - 0.5
            x1 = pos + 0.5

            # Get candle data
            candle = df_plot.loc[date]
            o, c = candle['open'], candle['close']
            
            # Determine correctness and color by action
            if pred in [1, 1.0]:
                correct = c >= o
                color = '#00ff00' if correct else '#ff0000'  # green/red
            elif pred in [-1, -1.0]:
                correct = c < o
                color = '#00ff00' if correct else '#ff0000'
            elif pred in [0, 0.0]:
                # Neutral (hold): shade with gray, no correctness notion
                color = '#808080'
            else:
                # Unknown value: skip
                continue
            
            # Shade the full vertical span between x0 and x1 on the same Axes as the mplfinance candles
            axlist[0].axvspan(x0, x1, ymin=0.0, ymax=1.0, facecolor=color, alpha=0.15, zorder=3)

    # Show plot
    axlist[0].set_xlim(*initial_limits)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=450)
    else:
        plt.tight_layout()
    
    plt.show()
    


if __name__ == '__main__':
    # Sample data preparation
    dates = pd.date_range(start='2023-01-01', periods=30)
    df = pd.DataFrame({
        'open': np.random.uniform(100, 200, 30),
        'high': np.random.uniform(200, 300, 30),
        'low': np.random.uniform(50, 100, 30),
        'close': np.random.uniform(100, 200, 30)
    }, index=dates)

    # Classification predictions (last 5 days)
    class_preds = [1, 0, 1, 1, 0]

    # Regression predictions (3-day horizon)
    reg_preds = np.random.uniform(150, 250, (5, 3))

    # Generate visualization
    plot_candles_finplot(
        df,
        predictions=class_preds,
        start='2023-01-15',
        coin_name='SampleCoin',
        save_path='sample_plot.png'
    )
