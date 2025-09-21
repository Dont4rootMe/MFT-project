# Copyright 2020 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import sys
import logging
import importlib

from abc import abstractmethod
from datetime import datetime
from typing import Optional, Union, Tuple
from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from IPython.display import display, clear_output
from pandas.plotting import register_matplotlib_converters

from tensortrade.oms.orders import TradeSide
from tensortrade.env.generic import Renderer, TradingEnv


LOGGER = logging.getLogger(__name__)


if importlib.util.find_spec("matplotlib"):
    import matplotlib.pyplot as plt

    from matplotlib import style

    style.use("ggplot")
    register_matplotlib_converters()

if importlib.util.find_spec("plotly"):
    import plotly.graph_objects as go

    from plotly.subplots import make_subplots


def _as_plotly_figure(fig):
    """Return a regular Plotly Figure to avoid widget issues."""
    # Always use regular Figure to avoid anywidget/VS Code compatibility issues
    # This provides stable chart rendering without interactive widget dependencies
    return fig


def _create_auto_file_name(filename_prefix: str,
                           ext: str,
                           timestamp_format: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = datetime.now().strftime(timestamp_format)
    filename = filename_prefix + timestamp + '.' + ext
    return filename


def _check_path(path: str, auto_create: bool = True) -> None:
    if not path or os.path.exists(path):
        return

    if auto_create:
        os.mkdir(path)
    else:
        raise OSError(f"Path '{path}' not found.")


def _check_valid_format(valid_formats: list, save_format: str) -> None:
    if save_format not in valid_formats:
        raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), save_format))


class BaseRenderer(Renderer):
    """The abstract base renderer to be subclassed when making a renderer
    the incorporates a `Portfolio`.
    """

    def __init__(self):
        super().__init__()
        self._max_episodes = None
        self._max_steps = None

    @staticmethod
    def _create_log_entry(episode: int = None,
                          max_episodes: int = None,
                          step: int = None,
                          max_steps: int = None,
                          date_format: str = "%Y-%m-%d %H:%M:%S",
                          chart_name: str = None) -> str:
        """
        Creates a log entry to be used by a renderer.

        Parameters
        ----------
        episode : int
            The current episode.
        max_episodes : int
            The maximum number of episodes that can occur.
        step : int
            The current step of the current episode.
        max_steps : int
            The maximum number of steps within an episode that can occur.
        date_format : str
            The format for logging the date.
        chart_name : str
            The name of the chart.
            
        Returns
        -------
        str
            a log entry
        """
        log_entry = ""

        # if episode is not None:
        #     log_entry += f" Episode: {episode + 1}/{max_episodes if max_episodes else ''}"

        # if step is not None:
        #     log_entry += f" Step: {step}/{max_steps if max_steps else ''}"

        if chart_name is not None:
            log_entry += f" Chart: {chart_name}"

        return log_entry

    def render(self, env: 'TradingEnv',  **kwargs):

        price_history = None
        if len(env.observer.renderer_history) > 0:
            price_history = pd.DataFrame(env.observer.renderer_history)

        performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')

        self.render_env(
            episode=kwargs.get("episode", None),
            max_episodes=kwargs.get("max_episodes", None),
            step=env.clock.step,
            max_steps=kwargs.get("max_steps", None),
            price_history=price_history,
            net_worth=performance.net_worth,
            performance=performance,
            trades=env.action_scheme.broker.trades
        )

    @abstractmethod
    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        """Renderers the current state of the environment.

        Parameters
        ----------
        episode : int
            The episode that the environment is being rendered for.
        max_episodes : int
            The maximum number of episodes that will occur.
        step : int
            The step of the current episode that is happening.
        max_steps : int
            The maximum number of steps that will occur in an episode.
        price_history : `pd.DataFrame`
            The history of instrument involved with the environment. The
            required columns are: date, open, high, low, close, and volume.
        net_worth : `pd.Series`
            The history of the net worth of the `portfolio`.
        performance : `pd.Series`
            The history of performance of the `portfolio`.
        trades : `OrderedDict`
            The history of trades for the current episode.
        """
        raise NotImplementedError()

    def save(self, root_path: str | None = None) -> None:
        """Saves the rendering of the `TradingEnv`.
        """
        pass

    def reset(self) -> None:
        """Resets the renderer.
        """
        pass


class EmptyRenderer(Renderer):
    """A renderer that does renders nothing.

    Needed to make sure that environment can function without requiring a
    renderer.
    """

    def render(self, env, **kwargs):
        pass


class ScreenLogger(BaseRenderer):
    """Logs information the screen of the user.

    Parameters
    ----------
    date_format : str
        The format for logging the date.
    """

    DEFAULT_FORMAT: str = "[%(asctime)-15s] %(message)s"

    def __init__(self, date_format: str = "%Y-%m-%d %H:%M:%S"):
        super().__init__()
        self._date_format = date_format

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None):
        print(self._create_log_entry(episode, max_episodes, step, max_steps, date_format=self._date_format))


class FileLogger(BaseRenderer):
    """Logs information to a file.

    Parameters
    ----------
    filename : str
        The file name of the log file. If omitted, a file name will be
        created automatically.
    path : str
        The path to save the log files to. None to save to same script directory.
    log_format : str
        The log entry format as per Python logging. None for default. For
        more details, refer to https://docs.python.org/3/library/logging.html
    timestamp_format : str
        The format of the timestamp of the log entry. Node for default.
    """

    DEFAULT_LOG_FORMAT: str = '[%(asctime)-15s] %(message)s'
    DEFAULT_TIMESTAMP_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    def __init__(self,
                 filename: str = None,
                 path: str = 'log',
                 log_format: str = 'html',
                 timestamp_format: str = None) -> None:
        super().__init__()
        _check_path(path)

        if not filename:
            filename = _create_auto_file_name('log_', 'log')

        self._logger = logging.getLogger(self.id)
        self._logger.setLevel(logging.INFO)

        if path:
            filename = os.path.join(path, filename)
        handler = logging.FileHandler(filename)
        handler.setFormatter(
            logging.Formatter(
                log_format if log_format is not None else self.DEFAULT_LOG_FORMAT,
                datefmt=timestamp_format if timestamp_format is not None else self.DEFAULT_TIMESTAMP_FORMAT
            )
        )
        self._logger.addHandler(handler)

    @property
    def log_file(self) -> str:
        """The filename information is being logged to. (str, read-only)
        """
        return self._logger.handlers[0].baseFilename

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        log_entry = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._logger.info(f"{log_entry} - Performance:\n{performance}")


class PlotlyTradingChart(BaseRenderer):
    """Trading visualization for TensorTrade using Plotly.

    Parameters
    ----------
    display : bool
        True to display the chart on the screen, False for not.
    height : int
        Chart height in pixels. Affects both display and saved file
        charts. Set to None for automatic height. Default is None.
    width : int
        Chart width in pixels. Affects both display and saved file
        charts. Set to None for automatic width. Default is None.
    save_format : str
        A format to save the chart to. Acceptable formats are
        html, png, jpeg, webp, svg, pdf, eps. All the formats except for
        'html' require Orca. Default is None for no saving.
    path : str
        The path to save the char to if save_format is not None. The folder
        will be created if not found.
    filename_prefix : str
        A string that precedes automatically-created file name
        when charts are saved. Default 'chart_'.
    timestamp_format : str
        The format of the date shown in the chart title.
    auto_open_html : bool
        Works for save_format='html' only. True to automatically
        open the saved chart HTML file in the default browser, False otherwise.
    include_plotlyjs : Union[bool, str]
        Whether to include/load the plotly.js library in the saved
        file. 'cdn' results in a smaller file by loading the library online but
        requires an Internet connect while True includes the library resulting
        in much larger file sizes. False to not include the library. For more
        details, refer to https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html

    Notes
    -----
    Possible Future Enhancements:
        - Saving images without using Orca.
        - Limit displayed step range for the case of a large number of steps and let
          the shown part of the chart slide after filling that range to keep showing
          recent data as it's being added.

    References
    ----------
    .. [1] https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html
    .. [2] https://plot.ly/python/figurewidget/
    .. [3] https://plot.ly/python/subplots/
    .. [4] https://plot.ly/python/reference/#candlestick
    .. [5] https://plot.ly/python/#chart-events
    """

    def __init__(self,
                 display: bool = True,
                 height: int = None,
                 width: int = None,
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'trading_chart_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._timestamp_format = timestamp_format
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html

        # if self._save_format and self._path and not os.path.exists(path):
        #     os.mkdir(path)

        self.fig = None
        self._price_chart_indices = {}  # Dictionary to store price chart indices for each currency
        self._balance_chart_indices = {}  # Dictionary to store balance chart indices
        self._performance_chart_indices = {}  # Dictionary to store performance chart indices
        self._base_annotations = None
        self._last_trade_step = 0
        self._show_chart = display
        self._currencies = []  # List to store detected currencies

    def _detect_currencies(self, price_history: pd.DataFrame) -> list:
        """Detect currencies from price_history columns ending with '_open'.
        
        Parameters
        ----------
        price_history : pd.DataFrame
            The price history DataFrame
            
        Returns
        -------
        list
            List of detected currency names
        """
        currencies = []
        for col in price_history.columns:
            if col.endswith('_open'):
                currency = col[:-5]  # Remove '_open' suffix
                currencies.append(currency)
            elif col == 'open':  # Single currency case
                currencies.append('default')
        
        return currencies

    def _create_figure(self, performance_keys: dict, currencies: list) -> None:
        # Simpler layout similar to original: 4 rows, 1 column main structure
        n_currencies = len(currencies)
        
        # Create a 3-row layout (removed Volume chart)
        fig = make_subplots(
            rows=3, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],  # Give most space to net worth chart
            subplot_titles=['Net Worth (Portfolio Value)', 'Performance', 'Price & Balances']
        )
        
        # 1. Add Net Worth chart (row 1)
        net_worth_trace = go.Scatter(
            mode='lines', 
            name='Net Worth', 
            line=dict(color='DarkGreen', width=3),
            showlegend=True
        )
        fig.add_trace(net_worth_trace, row=1, col=1)
        
        # 2. Add Performance charts (row 2)
        self._performance_chart_indices = {}
        trace_index = 1
        for key in performance_keys:
            if key != 'net_worth':  # Skip net_worth as it's already displayed
                perf_trace = go.Scatter(
                    mode='lines',
                    name=key,
                    line=dict(width=2),
                    showlegend=True
                )
                fig.add_trace(perf_trace, row=2, col=1)
                self._performance_chart_indices[key] = trace_index
                trace_index += 1
        
        # 3. Add OHLC candlestick chart (row 3) - primary price chart
        self._price_chart_indices = {}
        self._balance_chart_indices = {}
        
        # Add main OHLC chart for the first/primary currency
        primary_currency = currencies[0] if currencies else 'default'
        candlestick_trace = go.Candlestick(
            name='Price',
            showlegend=False,
            increasing=dict(line=dict(color='green')),
            decreasing=dict(line=dict(color='red'))
        )
        fig.add_trace(candlestick_trace, row=3, col=1)
        self._price_chart_indices[primary_currency] = trace_index
        trace_index += 1
        
        # Add balance charts to the same row as additional traces
        for currency in currencies:
            # Free balance
            free_trace = go.Scatter(
                mode='lines',
                name=f'{currency.upper()} Free',
                line=dict(color='blue', width=2),
                showlegend=True,
                yaxis='y4'  # Use secondary y-axis for balances
            )
            fig.add_trace(free_trace, row=3, col=1)
            free_index = trace_index
            trace_index += 1
            
            # Reserved balance
            reserved_trace = go.Scatter(
                mode='lines',
                name=f'{currency.upper()} Reserved',
                line=dict(color='orange', width=2),
                showlegend=True,
                yaxis='y4'  # Use secondary y-axis for balances
            )
            fig.add_trace(reserved_trace, row=3, col=1)
            reserved_index = trace_index
            trace_index += 1
            
            # Store balance chart indices
            self._balance_chart_indices[currency] = {
                'free': free_index,
                'reserved': reserved_index
            }
        
        # Update layout for better appearance
        fig.update_layout(
            template='plotly_white',
            height=self._height or 800,
            width=self._width,
            margin=dict(t=80, b=60, l=60, r=60),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.36,
                xanchor="right",
                x=1
            ),
            # Add secondary y-axis for balances
            yaxis4=dict(
                overlaying='y3',
                side='right',
                title='Balance'
            )
        )
        
        # Update axes styling
        fig.update_xaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_yaxes(linecolor='Grey', gridcolor='Gainsboro')
        
        # Disable x-axis rangeslider for all subplots
        fig.update_xaxes(rangeslider_visible=False)
        
        # Set axis titles
        fig.update_xaxes(title_text='Time', row=3, col=1)
        
        # Y-axis titles
        fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)
        fig.update_yaxes(title_text='Performance', row=2, col=1)
        fig.update_yaxes(title_text='Price ($)', row=3, col=1)

        self.fig = _as_plotly_figure(fig)
        
        self.fig.update_annotations({'font': {'size': 12}})
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(self,
                                  trades: 'OrderedDict',
                                  price_history: 'pd.DataFrame',
                                  currencies: list,
                                  net_worth: 'pd.Series') -> 'Tuple[go.layout.Annotation]':
        """Creates annotations of the new trades after the last one in the chart.

        Parameters
        ----------
        trades : `OrderedDict`
            The history of trades for the current episode.
        price_history : `pd.DataFrame`
            The price history of the current episode.
        currencies : list
            List of currencies in the environment

        Returns
        -------
        `Tuple[go.layout.Annotation]`
            A tuple of annotations used in the renderering process.
        """
        annotations = []
        
        # Determine which row the OHLC chart is on
        # For single currency, it's row 4 (after net worth, volume, performance)
        # For multi-currency, we need to determine which currency the trade is for
        ohlc_row = 4  # Default for first/only currency
        
        for trade in reversed(trades.values()):
            trade = trade[0]

            tp = float(trade.price)
            ts = float(trade.size)

            if trade.step <= self._last_trade_step:
                break

            # For our simple 3-row layout:
            # Row 1: x1, y1 (net worth - Portfolio Value)
            # Row 2: x2, y2 (performance)  
            # Row 3: x3, y3 (OHLC price chart)
            
            # Trade annotations go on the Net Worth chart (row 1) where portfolio value is shown
            xref = 'x1'
            yref = 'y1'

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
                qty = round(ts / tp, trade.quote_instrument.precision) if tp > 0 else ts

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'] if 'date' in price_history.columns else f'Step {trade.step}',
                    side=trade.side.value.upper(),
                    qty=qty,
                    size=ts,
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )

            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -15

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'] if 'date' in price_history.columns else f'Step {trade.step}',
                    side=trade.side.value.upper(),
                    qty=ts,
                    size=round(ts * tp, trade.base_instrument.precision) if tp > 0 else ts,
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>' \
                        '{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>' \
                        'Total: {size} {base_instrument} - Comm.: {commission}'.format(**text_info)

            # For Net Worth chart, use the portfolio value at the trade step
            trade_step_idx = trade.step - 1
            # Get net worth value at the trade step
            if trade_step_idx < len(net_worth) and trade_step_idx >= 0:
                net_worth_value = net_worth.iloc[trade_step_idx]
            else:
                net_worth_value = net_worth.iloc[-1] if len(net_worth) > 0 else 1000
            
            annotations += [go.layout.Annotation(
                x=trade_step_idx, y=net_worth_value,
                ax=0, ay=ay, xref=xref, yref=yref, showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
            )]

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step

        return tuple(annotations)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        # Detect currencies from price_history
        currencies = self._detect_currencies(price_history)
        
        if not self.fig or self._currencies != currencies:
            self._currencies = currencies
            self._create_figure(performance.keys(), currencies)

        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='plotly: all trading info')
        
        # 1. Update net worth chart (trace 0)
        x_data = net_worth.index if hasattr(net_worth.index, 'dtype') and net_worth.index.dtype.kind in 'biufc' else range(len(net_worth))
        self.fig.data[0].x = x_data
        self.fig.data[0].y = net_worth
        
        # 2. Update performance charts
        perf_x_data = performance.index if hasattr(performance.index, 'dtype') and performance.index.dtype.kind in 'biufc' else range(len(performance))
        for key, trace_idx in self._performance_chart_indices.items():
            if key in performance.columns:
                self.fig.data[trace_idx].x = perf_x_data
                self.fig.data[trace_idx].y = performance[key]
        
        # 4. Update OHLC chart (primary currency only for now)
        primary_currency = currencies[0] if currencies else 'default'
        
        if primary_currency == 'default':
            # Single currency case
            open_col, high_col, low_col, close_col = 'open', 'high', 'low', 'close'
        else:
            # Multi-currency case with prefixes
            open_col = f'{primary_currency}_open'
            high_col = f'{primary_currency}_high'
            low_col = f'{primary_currency}_low'
            close_col = f'{primary_currency}_close'
        
        # Update OHLC candlestick data for primary currency
        if primary_currency in self._price_chart_indices:
            price_x_data = price_history.index if hasattr(price_history.index, 'dtype') and price_history.index.dtype.kind in 'biufc' else range(len(price_history))
            price_trace_idx = self._price_chart_indices[primary_currency]
            
            # Check if required columns exist
            if all(col in price_history.columns for col in [open_col, high_col, low_col, close_col]):
                self.fig.data[price_trace_idx].x = price_x_data
                self.fig.data[price_trace_idx].open = price_history[open_col]
                self.fig.data[price_trace_idx].high = price_history[high_col]
                self.fig.data[price_trace_idx].low = price_history[low_col]
                self.fig.data[price_trace_idx].close = price_history[close_col]
        
        # 5. Update balance charts for all currencies
        for currency in currencies:
            if currency == 'default':
                free_col, reserved_col = 'free', 'reserved'
            else:
                free_col = f'{currency}_free'
                reserved_col = f'{currency}_reserved'
                
            # Update balance charts if columns exist
            if currency in self._balance_chart_indices:
                balance_x_data = performance.index if hasattr(performance.index, 'dtype') and performance.index.dtype.kind in 'biufc' else range(len(performance))
                
                # Update free balance
                free_trace_idx = self._balance_chart_indices[currency]['free']
                if free_col in performance.columns:
                    self.fig.data[free_trace_idx].x = balance_x_data
                    self.fig.data[free_trace_idx].y = performance[free_col]
                else:
                    # Fallback: use a portion of net worth as free balance
                    self.fig.data[free_trace_idx].x = balance_x_data
                    self.fig.data[free_trace_idx].y = net_worth * 0.8  # Assume 80% is free
                
                # Update reserved balance
                reserved_trace_idx = self._balance_chart_indices[currency]['reserved']
                if reserved_col in performance.columns:
                    self.fig.data[reserved_trace_idx].x = balance_x_data
                    self.fig.data[reserved_trace_idx].y = performance[reserved_col]
                else:
                    # Fallback: use a portion of net worth as reserved balance
                    self.fig.data[reserved_trace_idx].x = balance_x_data
                    self.fig.data[reserved_trace_idx].y = net_worth * 0.2  # Assume 20% is reserved
        
        # Add trade annotations to Net Worth chart
        self.fig.layout.annotations = self._base_annotations + self._create_trade_annotations(trades, price_history, currencies, net_worth)
        
        # Display the chart AFTER all data has been populated
        if self._show_chart:
            display(self.fig)


    def save(self, root_path: str | None = None) -> None:
        """Saves the current chart to a file.

        Notes
        -----
        All formats other than HTML require Orca installed and server running.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
            _check_valid_format(valid_formats, self._save_format)

        _check_path(self._path)

        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        if root_path is not None:
            filename = os.path.join(root_path, filename)
        else:
            filename = os.path.join(self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self._last_trade_step = 0
        self._currencies = []
        self._price_chart_indices = {}
        self._balance_chart_indices = {}
        self._performance_chart_indices = {}
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)


class PnLWithOrdersChart(BaseRenderer):
    """Separate PnL chart with trade order annotations using Plotly."""
    
    def __init__(self,
                 display: bool = True,
                 height: int = 600,
                 width: int = None,
                 save_format: str = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'pnl_orders_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html
        self._show_chart = display
        
        self.fig = None
        self._base_annotations = None
        self._last_trade_step = 0

    def _create_figure(self) -> None:
        """Create the PnL figure with orders."""
        if not importlib.util.find_spec("plotly"):
            raise RuntimeError("Plotly is not installed but required for PnLWithOrdersChart.")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', name='PnL', 
                                line=dict(color='darkgreen', width=2)))
        
        fig.update_layout(
            template='plotly_white',
            height=self._height,
            width=self._width,
            margin=dict(b=120),
            title='PnL with Trade Orders',
            xaxis_title='Time Step',
            yaxis_title='PnL',
            annotations=[
                dict(
                    text="<b>PnL with Trade Orders</b><br>"
                         "Shows portfolio value over time with trade execution points marked.<br>"
                         "• Green triangles (▲): Buy orders executed<br>"
                         "• Red triangles (▼): Sell orders executed<br>"
                         "• Line color indicates overall trend and performance",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                    font=dict(size=11, color="darkblue"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            ]
        )
        
        self.fig = _as_plotly_figure(fig)
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(self, trades: 'OrderedDict', net_worth: 'pd.Series') -> 'Tuple[go.layout.Annotation]':
        """Create trade annotations for the PnL chart."""
        annotations = []
        
        for trade in reversed(trades.values()):
            trade = trade[0]
            
            if trade.step <= self._last_trade_step:
                break
                
            # Get PnL value at trade step
            pnl_at_trade = net_worth.iloc[trade.step - 1] if trade.step <= len(net_worth) and trade.step > 0 and len(net_worth) > 0 else net_worth.iloc[-1] if len(net_worth) > 0 else 0
            
            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 20
                symbol = '▲'
            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -20
                symbol = '▼'
            else:
                continue
                
            hovertext = f'{symbol} {trade.side.value.upper()}<br>' \
                       f'Size: {trade.size}<br>' \
                       f'Price: {trade.price}<br>' \
                       f'Step: {trade.step}'
            
            annotations += [go.layout.Annotation(
                x=trade.step - 1,
                y=pnl_at_trade,
                ax=0, ay=ay,
                xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                arrowwidth=3,
                arrowsize=1.0,
                hovertext=hovertext,
                opacity=0.8,
                hoverlabel=dict(bgcolor=color)
            )]
        
        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step
            
        return tuple(annotations)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        
        if net_worth is None:
            raise ValueError("PnLWithOrdersChart requires 'net_worth' parameter.")
            
        if trades is None:
            raise ValueError("PnLWithOrdersChart requires 'trades' parameter.")
            
        if not self.fig:
            self._create_figure()
            
        # Calculate PnL from net worth
        initial_value = net_worth.iloc[0] if len(net_worth) > 0 else 1000
        pnl = net_worth - initial_value
        
        # Update PnL trace
        x_data = pnl.index if hasattr(pnl.index, 'dtype') and pnl.index.dtype.kind in 'biufc' else range(len(pnl))
        self.fig.data[0].update({'x': x_data, 'y': pnl})
        
        # Update title
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='PnL with Orders')
        
        # Add trade annotations
        self.fig.layout.annotations = self._base_annotations + self._create_trade_annotations(trades, net_worth)
        
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        """Save the PnL chart."""
        if not self._save_format:
            return
            
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        if root_path is not None:
            filename = os.path.join(root_path, filename)
        else:
            filename = os.path.join(self._path, filename)
            
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs=self._include_plotlyjs, auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self._last_trade_step = 0
        if self.fig is not None:
            self.fig.layout.annotations = self._base_annotations


class CurrencyPriceChart(BaseRenderer):
    """Separate OHLC price chart for individual currencies using Plotly."""
    
    def __init__(self,
                 currency: str = 'default',
                 display: bool = True,
                 height: int = 500,
                 width: int = None,
                 save_format: str = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'price_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._currency = currency
        self._height = height
        self._width = width
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html
        self._show_chart = display
        
        self.fig = None

    def _create_figure(self) -> None:
        """Create the price figure."""
        if not importlib.util.find_spec("plotly"):
            raise RuntimeError("Plotly is not installed but required for CurrencyPriceChart.")
            
        fig = go.Figure()
        fig.add_trace(go.Candlestick(name=f'{self._currency.upper()} Price'))
        
        fig.update_layout(
            template='plotly_white',
            height=self._height,
            width=self._width,
            margin=dict(b=120),
            title=f'{self._currency.upper()} OHLC Price Chart',
            xaxis_title='Time Step',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            annotations=[
                dict(
                    text=f"<b>{self._currency.upper()} OHLC Price Chart</b><br>"
                         "Shows Open, High, Low, Close price data as candlesticks.<br>"
                         "• Green candles: Close > Open (bullish period)<br>"
                         "• Red candles: Close < Open (bearish period)<br>"
                         "• Wicks show the high/low range for each time period",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                    font=dict(size=11, color="darkblue"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            ]
        )
        
        self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        
        if price_history is None:
            raise ValueError("CurrencyPriceChart requires 'price_history' parameter.")
            
        if not self.fig:
            self._create_figure()
            
        # Determine column names based on currency
        if self._currency == 'default':
            open_col, high_col, low_col, close_col = 'open', 'high', 'low', 'close'
        else:
            open_col = f'{self._currency}_open'
            high_col = f'{self._currency}_high'
            low_col = f'{self._currency}_low'
            close_col = f'{self._currency}_close'
        
        # Check if required columns exist
        required_cols = [open_col, high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in price_history.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for currency '{self._currency}': {missing_cols}")
        
        # Update candlestick data
        x_data = price_history.index if hasattr(price_history.index, 'dtype') and price_history.index.dtype.kind in 'biufc' else range(len(price_history))
        self.fig.data[0].update(dict(
            x=x_data,
            open=price_history[open_col],
            high=price_history[high_col],
            low=price_history[low_col],
            close=price_history[close_col]
        ))
        
        # Update title
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, 
                                                      chart_name=f'{self._currency.upper()} Price')
        
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        """Save the price chart."""
        if not self._save_format:
            return
            
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        
        filename = _create_auto_file_name(f"{self._filename_prefix}{self._currency}_", self._save_format)
        if root_path is not None:
            filename = os.path.join(root_path, filename)
        else:
            filename = os.path.join(self._path, filename)
            
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs=self._include_plotlyjs, auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        pass


class SimplePnLChart(BaseRenderer):
    """Simple PnL chart without trade annotations using Plotly."""
    
    def __init__(self,
                 display: bool = True,
                 height: int = 400,
                 width: int = None,
                 save_format: str = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'simple_pnl_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html
        self._show_chart = display
        
        self.fig = None

    def _create_figure(self) -> None:
        """Create the simple PnL figure."""
        if not importlib.util.find_spec("plotly"):
            raise RuntimeError("Plotly is not installed but required for SimplePnLChart.")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode='lines', name='PnL', 
                                line=dict(color='blue', width=2)))
        
        fig.update_layout(
            template='plotly_white',
            height=self._height,
            width=self._width,
            margin=dict(b=120),
            title='PnL Chart',
            xaxis_title='Time Step',
            yaxis_title='PnL',
            annotations=[
                dict(
                    text="<b>Simple PnL Chart</b><br>"
                         "Shows cumulative profit/loss over time without trade markers.<br>"
                         "• Clean view of portfolio performance trend<br>"
                         "• Positive values: Cumulative gains<br>"
                         "• Negative values: Cumulative losses • Zero line: Break-even",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                    font=dict(size=11, color="darkblue"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            ]
        )
        
        self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        
        if net_worth is None:
            raise ValueError("SimplePnLChart requires 'net_worth' parameter.")
            
        if not self.fig:
            self._create_figure()
            
        # Calculate PnL from net worth
        initial_value = net_worth.iloc[0] if len(net_worth) > 0 else 1000
        pnl = net_worth - initial_value
        
        # Update PnL trace
        x_data = pnl.index if hasattr(pnl.index, 'dtype') and pnl.index.dtype.kind in 'biufc' else range(len(pnl))
        self.fig.data[0].update({'x': x_data, 'y': pnl})
        
        # Update title
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='Simple PnL')
        
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        """Save the simple PnL chart."""
        if not self._save_format:
            return
            
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        if root_path is not None:
            filename = os.path.join(root_path, filename)
        else:
            filename = os.path.join(self._path, filename)
            
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs=self._include_plotlyjs, auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        pass


class MultiChartRenderer(BaseRenderer):
    """Composite renderer that manages multiple separate charts."""
    
    def __init__(self, 
                 charts: list = None,
                 auto_detect_currencies: bool = True) -> None:
        """
        Initialize MultiChartRenderer.
        
        Parameters
        ----------
        charts : list
            List of renderer instances to manage
        auto_detect_currencies : bool
            If True, automatically create CurrencyPriceChart instances for detected currencies
        """
        super().__init__()
        self.charts = charts or []
        self.auto_detect_currencies = auto_detect_currencies
        self._currency_charts = {}
        self._detected_currencies = []

    def add_chart(self, chart: BaseRenderer) -> None:
        """Add a chart to the renderer."""
        self.charts.append(chart)

    def _detect_currencies(self, price_history: pd.DataFrame) -> list:
        """Detect currencies from price_history columns."""
        currencies = []
        for col in price_history.columns:
            if col.endswith('_open'):
                currency = col[:-5]
                currencies.append(currency)
            elif col == 'open':
                currencies.append('default')
        return currencies

    def _create_currency_charts(self, currencies: list) -> None:
        """Create currency charts for detected currencies."""
        for currency in currencies:
            if currency not in self._currency_charts:
                self._currency_charts[currency] = CurrencyPriceChart(
                    currency=currency,
                    display=True,
                    height=400
                )

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        
        # Auto-detect and create currency charts if enabled
        # Note: If price_history is None, we skip detection without warning since other charts might still render
        if self.auto_detect_currencies and price_history is not None:
            currencies = self._detect_currencies(price_history)
            if currencies != self._detected_currencies:
                self._detected_currencies = currencies
                self._create_currency_charts(currencies)
        
        # Render all managed charts
        for chart in self.charts:
            try:
                chart.render_env(
                    episode=episode,
                    max_episodes=max_episodes,
                    step=step,
                    max_steps=max_steps,
                    price_history=price_history,
                    net_worth=net_worth,
                    performance=performance,
                    trades=trades
                )
            except Exception as e:
                print(f"Error rendering chart {type(chart).__name__}: {e}")
        
        # Render auto-created currency charts
        for currency_chart in self._currency_charts.values():
            try:
                currency_chart.render_env(
                    episode=episode,
                    max_episodes=max_episodes,
                    step=step,
                    max_steps=max_steps,
                    price_history=price_history,
                    net_worth=net_worth,
                    performance=performance,
                    trades=trades
                )
            except Exception as e:
                print(f"Error rendering currency chart: {e}")

    def save(self, root_path: str | None = None) -> None:
        """Save all charts."""
        for chart in self.charts:
            if hasattr(chart, 'save'):
                chart.save(root_path)
        
        for currency_chart in self._currency_charts.values():
            if hasattr(currency_chart, 'save'):
                currency_chart.save(root_path)

    def reset(self) -> None:
        """Reset all charts."""
        for chart in self.charts:
            if hasattr(chart, 'reset'):
                chart.reset()
        
        for currency_chart in self._currency_charts.values():
            if hasattr(currency_chart, 'reset'):
                currency_chart.reset()


class MatplotlibTradingChart(BaseRenderer):
    """ Trading visualization for TensorTrade using Matplotlib
    Parameters
    ---------
    display : bool
        True to display the chart on the screen, False for not.
    save_format : str
        A format to save the chart to. Acceptable formats are
        png, jpg, svg, pdf.
    path : str
        The path to save the char to if save_format is not None. The folder
        will be created if not found.
    filename_prefix : str
        A string that precedes automatically-created file name
        when charts are saved. Default 'chart_'.
    """
    def __init__(self,
                 display: bool = True,
                 save_format: str = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'chart_') -> None:
        super().__init__()
        self._volume_chart_height = 0.33

        self._df = None
        self.fig = None
        self._price_ax = None
        self._volume_ax = None
        self.net_worth_ax = None
        self._show_chart = display

        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix

        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _create_figure(self) -> None:
        self.fig = plt.figure()

        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8,
                                         colspan=1, sharex=self.net_worth_ax)
        self.volume_ax = self.price_ax.twinx()
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

    def _render_trades(self, step_range, trades) -> None:
        trades = [trade for sublist in trades.values() for trade in sublist]

        for trade in trades:
            if trade.step in range(sys.maxsize)[step_range]:
                date = self._df.index.values[trade.step]
                close = self._df['close'].values[trade.step]
                color = 'green'

                if trade.side is TradeSide.SELL:
                    color = 'red'

                self.price_ax.annotate(' ', (date, close),
                                       xytext=(date, close),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def _render_volume(self, step_range, times) -> None:
        self.volume_ax.clear()

        volume = np.array(self._df['volume'].values[step_range])

        self.volume_ax.plot(times, volume,  color='blue')
        self.volume_ax.fill_between(times, volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / self._volume_chart_height)
        self.volume_ax.yaxis.set_ticks([])

    def _render_price(self, step_range, times, current_step) -> None:
        self.price_ax.clear()

        self.price_ax.plot(times, self._df['close'].values[step_range], color="black")

        last_time = self._df.index.values[current_step]
        last_close = self._df['close'].values[current_step]
        last_high = self._df['high'].values[current_step]

        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_time, last_close),
                               xytext=(last_time, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * self._volume_chart_height, ylim[1])

    # def _render_net_worth(self, step_range, times, current_step, net_worths, benchmarks):
    def _render_net_worth(self, step_range, times, current_step, net_worths) -> None:
        self.net_worth_ax.clear()
        self.net_worth_ax.plot(times, net_worths[step_range], label='Net Worth', color="g")
        self.net_worth_ax.legend()

        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_time = times[-1]
        last_net_worth = list(net_worths[step_range])[-1]

        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_time, last_net_worth),
                                   xytext=(last_time, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        self.net_worth_ax.set_ylim(min(net_worths) / 1.25, max(net_worths) * 1.25)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure()

        if self._show_chart:
            plt.show(block=False)

        current_step = step - 1

        self._df = price_history
        if max_steps:
            window_size = max_steps
        else:
            window_size = 20

        current_net_worth = round(net_worth[len(net_worth)-1], 1)
        initial_net_worth = round(net_worth[0], 1)
        profit_percent = round((current_net_worth - initial_net_worth) / initial_net_worth * 100, 2)

        self.fig.suptitle('Net worth: $' + str(current_net_worth) +
                          ' | Profit: ' + str(profit_percent) + '%')

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step)

        times = self._df.index.values[step_range]

        if len(times) > 0:
            # self._render_net_worth(step_range, times, current_step, net_worths, benchmarks)
            self._render_net_worth(step_range, times, current_step, net_worth)
            self._render_price(step_range, times, current_step)
            self._render_volume(step_range, times)
            self._render_trades(step_range, trades)

        self.price_ax.set_xticklabels(times, rotation=45, horizontalalignment='right')

        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        """Saves the rendering of the `TradingEnv`.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['png', 'jpeg', 'svg', 'pdf']
            _check_valid_format(valid_formats, self._save_format)

        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        if root_path is not None:
            filename = os.path.join(root_path, filename)
        else:
            filename = os.path.join(self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        """Resets the renderer.
        """
        self.fig = None
        self._price_ax = None
        self._volume_ax = None
        self.net_worth_ax = None
        self._df = None



def _pnl_series(net_worth: 'pd.Series') -> 'pd.Series':
    """Cumulative Profit/Loss in base currency since start."""
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute PnL.")
    return net_worth - net_worth.iloc[0]
def _rolling_sharpe_from_net_worth(net_worth: 'pd.Series',
                                   window: int = 30,
                                   annualize: bool = True,
                                   periods_per_year: int = 252) -> 'pd.Series':
    """Compute rolling Sharpe from net worth. Risk-free assumed zero."""
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute Sharpe.")
    rets = net_worth.pct_change().fillna(0.0)
    mean = rets.rolling(window).mean()
    std = rets.rolling(window).std(ddof=1)
    sr = mean / std
    if annualize:
        sr = np.sqrt(periods_per_year) * sr
    sr.name = f"Sharpe(window={window})"
    return sr


class SharpePlotlyRenderer(BaseRenderer):
    """Plots rolling Sharpe ratio computed from net worth using Plotly."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 window: int = 30,
                 annualize: bool = True,
                 periods_per_year: int = 252,
                 save_format: str | None = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'sharpe_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._window = window
        self._annualize = annualize
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for SharpePlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Sharpe Ratio</b><br>"
                             "Measures risk-adjusted return. Higher values indicate better performance per unit of risk.<br>"
                             "• >1.0: Good performance • >2.0: Very good • >3.0: Excellent<br>"
                             "• Negative values indicate underperformance vs risk-free rate",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Sharpe')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        sr = _rolling_sharpe_from_net_worth(net_worth, self._window, self._annualize, self._ppy)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='sharpe')
        self.fig.data[0].update({'x': sr.index, 'y': sr, 'name': sr.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class SharpeMatplotlibRenderer(BaseRenderer):
    """Plots rolling Sharpe ratio computed from net worth using Matplotlib."""
    def __init__(self,
                 display: bool = True,
                 window: int = 30,
                 annualize: bool = True,
                 periods_per_year: int = 252,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'sharpe_') -> None:
        super().__init__()
        self._show_chart = display
        self._window = window
        self._annualize = annualize
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        sr = _rolling_sharpe_from_net_worth(net_worth, self._window, self._annualize, self._ppy)
        self._ensure_fig()
        self.ax.clear()
        self.ax.plot(sr.index, sr)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='sharpe'))
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel(sr.name)
        if self._show_chart:
            plt.show(block=False)
            plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['png', 'jpeg', 'svg', 'pdf']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        self.fig = None
        self.ax = None


# 3) PnL extrema (max delta rise / delta downfall) renderers (Plotly / Matplotlib)
def _pnl_extrema_series(net_worth: 'pd.Series') -> tuple['pd.Series', 'pd.Series']:
    """For each timestamp y, compute cumulative extrema relative to initial value (timestamp 0):
       - max_uprise[y]   = max_{x<=y} (pnl[x]) = maximum PnL achieved up to timestamp y
       - max_downfall[y] = min_{x<=y} (pnl[x]) = maximum loss (minimum PnL) up to timestamp y
       where pnl = net_worth - net_worth.iloc[0] (PnL relative to initial net worth).
       
       This shows the running maximum profit and maximum loss from the starting point.
    """
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute PnL extrema.")
    
    # Calculate PnL relative to initial value (timestamp 0)
    pnl = _pnl_series(net_worth)  # This is net_worth - net_worth.iloc[0]
    
    # Running maximum PnL (maximum uprise from initial value)
    max_uprise = pnl.cummax()
    
    # Running minimum PnL (maximum downfall from initial value) 
    max_downfall = pnl.cummin()
    
    max_uprise.name = "Max Uprise from Start"
    max_downfall.name = "Max Downfall from Start"
    
    return max_uprise, max_downfall


class PnLExtremaPlotlyRenderer(BaseRenderer):
    """Plots time series of maximum uprise and maximum downfall from initial value in two subplots (Plotly)."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 600,
                 width: int | None = None,
                 save_format: str | None = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'pnl_extrema_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for PnLExtremaPlotlyRenderer.")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Max Uprise from Start", "Max Downfall from Start"))
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=140),
                annotations=[
                    dict(
                        text="<b>PnL Extrema Analysis</b><br>"
                             "Shows maximum profit (uprise) and maximum loss (downfall) achieved from start.<br>"
                             "• Top chart: Best performance reached so far (always ≥ 0)<br>"
                             "• Bottom chart: Worst loss experienced so far (always ≤ 0)<br>"
                             "• Helps assess peak performance and maximum risk exposure",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Max Uprise from Start', row=1, col=1)
            fig.add_scatter(mode='lines', name='Max Downfall from Start', row=2, col=1)
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        up, down = _pnl_extrema_series(net_worth)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='pnl_extrema')
        # Update traces
        up_x = up.index if hasattr(up.index, 'dtype') and up.index.dtype.kind in 'biufc' else range(len(up))
        down_x = down.index if hasattr(down.index, 'dtype') and down.index.dtype.kind in 'biufc' else range(len(down))
        self.fig.data[0].update({'x': up_x, 'y': up})
        self.fig.data[1].update({'x': down_x, 'y': down})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class PnLExtremaMatplotlibRenderer(BaseRenderer):
    """Plots time series of maximum uprise and maximum downfall from initial value in two subplots (Matplotlib)."""
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'pnl_extrema_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax_up = None
        self.ax_down = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, (self.ax_up, self.ax_down) = plt.subplots(2, 1, sharex=True)
            self.fig.subplots_adjust(hspace=0.25)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        up, down = _pnl_extrema_series(net_worth)
        self._ensure_fig()
        self.ax_up.clear()
        self.ax_down.clear()

        self.ax_up.plot(up.index, up)
        self.ax_up.set_title("Max Uprise from Start")

        self.ax_down.plot(down.index, down)
        self.ax_down.set_title("Max Downfall from Start")

        self.fig.suptitle(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='pnl_extrema'))
        self.ax_down.set_xlabel('Time')
        if self._show_chart:
            plt.show(block=False)
            plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['png', 'jpeg', 'svg', 'pdf']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        self.fig = None
        self.ax_up = None
        self.ax_down = None


#
# --- ROI and Drawdown helpers ---
def _roi_series(net_worth: 'pd.Series', scale_100: bool = True) -> 'pd.Series':
    """Return-on-Investment since start. ROI_t = NW_t / NW_0 - 1."""
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute ROI.")
    nw0 = net_worth.iloc[0]
    if nw0 == 0:
        raise ZeroDivisionError("Initial net worth is zero; cannot compute ROI.")
    roi = net_worth / nw0 - 1.0
    if scale_100:
        roi = 100.0 * roi
        roi.name = "ROI % (since start)"
    else:
        roi.name = "ROI (fraction, since start)"
    return roi


def _drawdown_series(net_worth: 'pd.Series', scale_100: bool = True) -> 'pd.Series':
    """Drawdown series (≤ 0): DD_t = NW_t / max_{τ≤t} NW_τ - 1."""
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute Drawdown.")
    running_max = net_worth.cummax()
    dd = net_worth / running_max - 1.0
    if scale_100:
        dd = 100.0 * dd
        dd.name = "Drawdown % (≤0)"
    else:
        dd.name = "Drawdown (fraction, ≤0)"
    return dd


# --- Calmar, Hit Ratio, Turnover helpers ---
def _returns_from_net_worth(net_worth: 'pd.Series') -> 'pd.Series':
    """Period-to-period returns inferred from net worth."""
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute returns.")
    return net_worth.pct_change().fillna(0.0)


def _calmar_series(net_worth: 'pd.Series', periods_per_year: int = 252) -> 'pd.Series':
    """Calmar-to-date time series: CAGR_to_date / |MDD_to_date|.
    CAGR_t = (NW_t / NW_0) ** (ppy / t) - 1, with t counted in periods.
    MDD_to_date is the absolute value of cumulative minimum drawdown up to t.
    """
    nw = net_worth
    if nw is None or len(nw) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute Calmar.")
    idx = nw.index
    # CAGR-to-date
    rel = nw / nw.iloc[0]
    steps = np.arange(1, len(nw) + 1, dtype=float)  # 1..T
    cagr = np.power(rel.values, periods_per_year / steps) - 1.0
    # Max drawdown absolute value up to date
    dd_frac = _drawdown_series(nw, scale_100=False)  # ≤ 0
    mdd_abs_to_date = (-dd_frac).cummax().values     # ≥ 0
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        calmar = np.where(mdd_abs_to_date > 0, cagr / mdd_abs_to_date, np.nan)
    calmar = pd.Series(calmar, index=idx, name="Calmar (to date)")
    return calmar


def _hit_ratio_series_from_returns(returns: 'pd.Series') -> 'pd.Series':
    """Cumulative hit ratio over time based on period returns.
    Hit ratio_t = (# periods with r>0 up to t) / (# periods up to t).
    Note: This is period-level, not trade-level, to keep dependency minimal.
    """
    if returns is None or len(returns) == 0:
        raise ValueError("`returns` Series is empty; cannot compute hit ratio.")
    wins = (returns > 0).astype(float)
    # cumulative mean
    denom = np.arange(1, len(wins) + 1, dtype=float)
    hr = wins.cumsum().values / denom
    hr_series = pd.Series(hr, index=returns.index, name="Hit Ratio (periods)")
    return hr_series


def _turnover_series(trades: 'OrderedDict', net_worth: 'pd.Series') -> 'pd.Series':
    """Per-step turnover rate ≈ traded notional / prior net worth.
    For each step:
      - BUY: notional ≈ trade.size  (assumed in base/quote cash as used by env)
      - SELL: notional ≈ trade.size * trade.price
    This matches how annotations compute totals in PlotlyTradingChart.
    Falls back to 0 when no trades occurred on a step.
    """
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute turnover.")
    idx = net_worth.index
    per_step_notional = pd.Series(0.0, index=idx)

    if trades:
        # trades is an OrderedDict: {step: [Trade, ...]}
        for step_key, trade_list in trades.items():
            if trade_list is None or len(trade_list) == 0:
                continue
            # Convert env step (1-based in annotations) to 0-based index into series
            # We clamp to index range just in case.
            step_idx = int(getattr(trade_list[0], "step", step_key)) - 1
            step_idx = max(0, min(step_idx, len(idx) - 1))
            total = 0.0
            for tr in trade_list:
                side = getattr(tr, "side", None)
                price = float(getattr(tr, "price", 0.0))
                size = float(getattr(tr, "size", 0.0))
                if side is None:
                    continue
                # TradeSide enum check via value string
                side_val = getattr(side, "value", str(side)).lower()
                if side_val == 'buy':
                    # Notional approximated by cash spent (size)
                    total += abs(size)
                elif side_val == 'sell':
                    # Notional approximated by proceeds (qty * price)
                    total += abs(size * price)
                else:
                    # Unknown side; try best-effort notional
                    total += abs(size * price) if price != 0 else abs(size)
            # Accumulate if multiple entries map to same step
            per_step_notional.iloc[step_idx] += total

    # Use prior net worth as denominator; fall back to initial NW for the first observation
    denom = net_worth.shift(1)
    denom.iloc[0] = net_worth.iloc[0]
    denom = denom.replace(0.0, np.nan)
    turnover = per_step_notional / denom
    turnover = turnover.fillna(0.0)
    turnover.name = "Turnover (per step)"
    return turnover


class CalmarMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 periods_per_year: int = 252,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'calmar_') -> None:
        super().__init__()
        self._show_chart = display
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        calmar = _calmar_series(net_worth, periods_per_year=self._ppy)
        self._ensure_fig()
        self.ax.clear()
        self.ax.plot(calmar.index, calmar)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='calmar'))
        self.ax.set_xlabel('Time'); self.ax.set_ylabel(calmar.name)
        if self._show_chart:
            plt.show(block=False); plt.pause(0.001)
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['png','jpeg','svg','pdf']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)
    def reset(self) -> None:
        self.fig=None; self.ax=None


# --- Hit Ratio Renderers ---

class HitRatioMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'hitratio_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        rets = _returns_from_net_worth(net_worth)
        hr = _hit_ratio_series_from_returns(rets)
        self._ensure_fig()
        self.ax.clear()
        self.ax.plot(hr.index, hr)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='hitratio'))
        self.ax.set_xlabel('Time'); self.ax.set_ylabel(hr.name)
        if self._show_chart:
            plt.show(block=False); plt.pause(0.001)
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['png','jpeg','svg','pdf']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)
    def reset(self) -> None:
        self.fig=None; self.ax=None


# --- Turnover Renderers ---


class TurnoverMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'turnover_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        to = _turnover_series(trades, net_worth)
        self._ensure_fig()
        self.ax.clear()
        self.ax.bar(to.index, to)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='turnover'))
        self.ax.set_xlabel('Time'); self.ax.set_ylabel(to.name)
        if self._show_chart:
            plt.show(block=False); plt.pause(0.001)
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['png','jpeg','svg','pdf']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)
    def reset(self) -> None:
        self.fig=None; self.ax=None


# --- ROI Renderers (Plotly / Matplotlib) ---
class ROIPlotlyRenderer(BaseRenderer):
    """Plots ROI (since start) computed from net worth using Plotly."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 save_format: str | None = None,
                 path: str | None = None,
                 filename_prefix: str = 'roi_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for ROIPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Return on Investment (ROI %)</b><br>"
                             "Shows percentage return relative to initial investment.<br>"
                             "• Positive values: Profit above initial capital<br>"
                             "• Negative values: Loss below initial capital<br>"
                             "• 100% ROI = doubled your money • 50% ROI = 1.5x return",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='ROI')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        roi = _roi_series(net_worth, scale_100=True)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='roi')
        self.fig.data[0].update({'x': roi.index, 'y': roi, 'name': roi.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class ROIMatplotlibRenderer(BaseRenderer):
    """Plots ROI (since start) computed from net worth using Matplotlib."""
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'roi_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        roi = _roi_series(net_worth, scale_100=True)
        self._ensure_fig()
        self.ax.clear()
        self.ax.plot(roi.index, roi)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='roi'))
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel(roi.name)
        if self._show_chart:
            plt.show(block=False)
            plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['png', 'jpeg', 'svg', 'pdf']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        self.fig = None
        self.ax = None


# --- Drawdown Renderers (Plotly / Matplotlib) ---
class DrawdownPlotlyRenderer(BaseRenderer):
    """Plots drawdown series (≤ 0) computed from net worth using Plotly."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 save_format: str | None = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'drawdown_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for DrawdownPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Drawdown (%)</b><br>"
                             "Shows percentage decline from previous peak. Always ≤ 0.<br>"
                             "• 0%: At new peak • -10%: 10% below recent high<br>"
                             "• -50%: Lost half of peak value • Lower is worse<br>"
                             "• Measures downside risk and recovery ability",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Drawdown')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        dd = _drawdown_series(net_worth, scale_100=True)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='drawdown')
        self.fig.data[0].update({'x': dd.index, 'y': dd, 'name': dd.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class DrawdownMatplotlibRenderer(BaseRenderer):
    """Plots drawdown series (≤ 0) computed from net worth using Matplotlib."""
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str | None = None,
                 filename_prefix: str = 'drawdown_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        dd = _drawdown_series(net_worth, scale_100=True)
        self._ensure_fig()
        self.ax.clear()
        self.ax.plot(dd.index, dd)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='drawdown'))
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel(dd.name)
        if self._show_chart:
            plt.show(block=False)
            plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['png', 'jpeg', 'svg', 'pdf']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        self.fig = None
        self.ax = None

#
# --- PnL-only Renderers (Plotly / Matplotlib) ---
class PnLPlotlyRenderer(BaseRenderer):
    """Plots cumulative PnL (since start) computed from net worth using Plotly."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 save_format: str | None = 'html',
                 path: str | None = None,
                 filename_prefix: str = 'pnl_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for PnLPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Profit & Loss (PnL)</b><br>"
                             "Shows cumulative profit/loss in absolute terms from start.<br>"
                             "• Positive values: Total profit made<br>"
                             "• Negative values: Total loss incurred<br>"
                             "• Zero line: Break-even point • Slope indicates performance trend",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='PnL (since start)')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        pnl = _pnl_series(net_worth)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='pnl')
        self.fig.data[0].update({'x': pnl.index, 'y': pnl, 'name': 'PnL (since start)'})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class PnLMatplotlibRenderer(BaseRenderer):
    """Plots cumulative PnL (since start) computed from net worth using Matplotlib."""
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = None,
                 path: str | None = None,
                 filename_prefix: str = 'pnl_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        pnl = _pnl_series(net_worth)
        self._ensure_fig()
        self.ax.clear()
        next_x = pnl.index
        self.ax.plot(next_x, pnl)
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='pnl'))
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('PnL (since start)')
        if self._show_chart:
            plt.show(block=False)
            plt.pause(0.001)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['png', 'jpeg', 'svg', 'pdf']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        self.fig = None
        self.ax = None


class CalmarPlotlyRenderer(BaseRenderer):
    """Plots running Calmar ratio computed from net worth using Plotly."""

    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 periods_per_year: int = 252,
                 save_format: str | None = None,
                 path: str | None = None,
                 filename_prefix: str = 'calmar_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for CalmarPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Calmar Ratio</b><br>"
                             "Risk-adjusted return: Annualized return ÷ Maximum Drawdown.<br>"
                             "• Higher values indicate better risk-adjusted performance<br>"
                             "• >1.0: Good • >2.0: Very good • >3.0: Excellent<br>"
                             "• Considers both returns and maximum loss exposure",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Calmar Ratio')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        calmar = _calmar_series(net_worth, periods_per_year=self._ppy)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='calmar')
        self.fig.data[0].update({'x': calmar.index, 'y': calmar, 'name': calmar.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class HitRatioPlotlyRenderer(BaseRenderer):
    """Plots running hit ratio computed from net worth using Plotly."""

    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 save_format: str | None = None,
                 path: str | None = None,
                 filename_prefix: str = 'hit_ratio_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for HitRatioPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Hit Ratio</b><br>"
                             "Percentage of profitable periods (based on returns, not trades).<br>"
                             "• 0.5 (50%): Half of periods were profitable<br>"
                             "• >0.6 (60%): Good consistency • >0.7 (70%): Very consistent<br>"
                             "• Measures win rate but not magnitude of wins/losses",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Hit Ratio')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        rets = _returns_from_net_worth(net_worth)
        hit_ratio = _hit_ratio_series_from_returns(rets)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='hitratio')
        self.fig.data[0].update({'x': hit_ratio.index, 'y': hit_ratio, 'name': hit_ratio.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None


class TurnoverPlotlyRenderer(BaseRenderer):
    """Plots cumulative turnover computed from executed trades using Plotly."""

    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 width: int | None = None,
                 initial_capital: float | None = None,
                 save_format: str | None = None,
                 path: str | None = None,
                 filename_prefix: str = 'turnover_') -> None:
        super().__init__()
        self._height = height
        self._width = width
        self._show_chart = display
        self._initial_capital = initial_capital
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if path is not None and self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for TurnoverPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(
                template='plotly_white',
                height=self._height,
                width=self._width,
                margin=dict(b=120),
                annotations=[
                    dict(
                        text="<b>Turnover Rate</b><br>"
                             "Trading activity relative to portfolio value (per step).<br>"
                             "• 0.1: 10% of portfolio traded • 1.0: 100% portfolio turnover<br>"
                             "• Higher values indicate more active trading<br>"
                             "• Useful for assessing transaction costs and strategy intensity",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.36, xanchor='center', yanchor='bottom',
                        font=dict(size=11, color="darkblue"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="lightgray",
                        borderwidth=1
                    )
                ]
            )
            fig.add_scatter(mode='lines', name='Cumulative Turnover')
            self.fig = _as_plotly_figure(fig)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:

        turnover = _turnover_series(trades, net_worth)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps, chart_name='turnover')
        self.fig.data[0].update({'x': turnover.index, 'y': turnover, 'name': turnover.name})
        if self._show_chart:
            display(self.fig)

    def save(self, root_path: str | None = None) -> None:
        if not self._save_format:
            return
        valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        _check_valid_format(valid_formats, self._save_format)
        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self.fig = None

_registry = {
    "screen-log": ScreenLogger,
    "file-log": FileLogger,
    "plotly": PlotlyTradingChart,
    "matplot": MatplotlibTradingChart,
    "pnl-orders": PnLWithOrdersChart,
    "currency-price": CurrencyPriceChart,
    "simple-pnl": SimplePnLChart,
    "multi-chart": MultiChartRenderer,
    "sharpe-plotly": SharpePlotlyRenderer,
    "sharpe-matplot": SharpeMatplotlibRenderer,
    "pnl-extrema-plotly": PnLExtremaPlotlyRenderer,
    "pnl-extrema-matplot": PnLExtremaMatplotlibRenderer,
    "roi-plotly": ROIPlotlyRenderer,
    "roi-matplot": ROIMatplotlibRenderer,
    "drawdown-plotly": DrawdownPlotlyRenderer,
    "drawdown-matplot": DrawdownMatplotlibRenderer,
    "pnl-plotly": PnLPlotlyRenderer,
    "pnl-matplot": PnLMatplotlibRenderer,
    "calmar-plotly": CalmarPlotlyRenderer,
    "calmar-matplot": CalmarMatplotlibRenderer,
    "hitratio-plotly": HitRatioPlotlyRenderer,
    "hitratio-matplot": HitRatioMatplotlibRenderer,
    "turnover-plotly": TurnoverPlotlyRenderer,
    "turnover-matplot": TurnoverMatplotlibRenderer,
}


class MultiFormatRenderer(Renderer):
    """Wrapper that saves renderer outputs in multiple formats."""

    def __init__(self, renderer: Renderer, formats: Sequence[str]) -> None:
        super().__init__()
        self._renderer = renderer
        self._formats = [str(fmt).lower() for fmt in formats if fmt]

    def render(self, env: 'TradingEnv', **kwargs):
        self._renderer.render(env, **kwargs)

    def save(self, root_path: str | None = None) -> None:
        if not self._formats:
            self._renderer.save(root_path)
            return

        original = getattr(self._renderer, "_save_format", None)
        for fmt in self._formats:
            try:
                setattr(self._renderer, "_save_format", fmt)
                self._renderer.save(root_path)
            except ValueError as exc:
                LOGGER.warning(
                    "Renderer %s does not support format '%s': %s",
                    self._renderer.__class__.__name__,
                    fmt,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "Renderer %s failed to save format '%s': %s",
                    self._renderer.__class__.__name__,
                    fmt,
                    exc,
                )
        if original is not None or hasattr(self._renderer, "_save_format"):
            setattr(self._renderer, "_save_format", original)

    def reset(self) -> None:
        self._renderer.reset()

    def close(self) -> None:
        self._renderer.close()


def construct_renderers(identifier: str,
                        display: bool = True,
                        save_formats: Optional[Sequence[str]] = ("png", "html")) -> 'BaseRenderer':
    """Gets the `BaseRenderer` that matches the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `BaseRenderer`

    Returns
    -------
    `BaseRenderer`
        The renderer associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `BaseRenderer`
    """
    
    formats = list(save_formats) if save_formats is not None else []

    def _wrap(renderer: 'BaseRenderer') -> Renderer:
        if formats:
            return MultiFormatRenderer(renderer, formats)
        return renderer

    def _get_one(identifier: str) -> Renderer:
        if identifier not in _registry.keys():
            msg = f"Identifier {identifier} is not associated with any `BaseRenderer`."
            raise KeyError(msg)
        renderer = _registry[identifier](display=display)
        return _wrap(renderer)
    
    print(f"Constructing renderers: {type(identifier)=}")
    
    if identifier == 'all':
        return [_get_one(i) for i in _registry.keys()]
    elif isinstance(identifier, Iterable):
        return [_get_one(i) for i in identifier]
    elif isinstance(identifier, str):
        return _get_one(identifier)
    else:
        raise ValueError(f"Invalid identifier: {identifier}")
