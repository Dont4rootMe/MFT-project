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
from typing import Union, Tuple
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from IPython.display import display, clear_output
from pandas.plotting import register_matplotlib_converters

from tensortrade.oms.orders import TradeSide
from tensortrade.env.generic import Renderer, TradingEnv


if importlib.util.find_spec("matplotlib"):
    import matplotlib.pyplot as plt

    from matplotlib import style

    style.use("ggplot")
    register_matplotlib_converters()

if importlib.util.find_spec("plotly"):
    import plotly.graph_objects as go

    from plotly.subplots import make_subplots


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
                          date_format: str = "%Y-%m-%d %H:%M:%S") -> str:
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

        Returns
        -------
        str
            a log entry
        """
        log_entry = f"[{datetime.now().strftime(date_format)}]"

        if episode is not None:
            log_entry += f" Episode: {episode + 1}/{max_episodes if max_episodes else ''}"

        if step is not None:
            log_entry += f" Step: {step}/{max_steps if max_steps else ''}"

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
        charts. Set to None for 100% height. Default is None.
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
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'chart_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._timestamp_format = timestamp_format
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html

        # if self._save_format and self._path and not os.path.exists(path):
        #     os.mkdir(path)

        self.fig = None
        self._price_charts = {}  # Dictionary to store price charts for each currency
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
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
        # Calculate number of rows needed: currencies + volume + performance + net_worth
        n_currencies = len(currencies)
        total_rows = n_currencies + 3  # currencies + volume + performance + net_worth
        
        # Calculate row heights dynamically
        price_height_per_currency = 0.6 / n_currencies if n_currencies > 0 else 0.6
        row_heights = [price_height_per_currency] * n_currencies + [0.15, 0.15, 0.1]
        
        fig = make_subplots(
            rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=[f'{currency.upper()} Price' for currency in currencies] + 
                          ['Volume', 'Performance', 'Net Worth']
        )
        
        # Add candlestick chart for each currency
        for i, currency in enumerate(currencies):
            fig.add_trace(go.Candlestick(name=f'{currency.upper()} Price', 
                                       showlegend=False), row=i+1, col=1)
        
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Add volume chart (after all currency charts)
        volume_row = n_currencies + 1
        fig.add_trace(go.Bar(name='Volume', showlegend=False,
                             marker={'color': 'DodgerBlue'}),
                      row=volume_row, col=1)

        # Add performance charts
        performance_row = n_currencies + 2
        for k in performance_keys:
            fig.add_trace(go.Scatter(mode='lines', name=k), row=performance_row, col=1)

        # Add net worth chart
        net_worth_row = n_currencies + 3
        fig.add_trace(go.Scatter(mode='lines', name='Net Worth', marker={'color': 'DarkGreen'}),
                      row=net_worth_row, col=1)

        fig.update_xaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_yaxes(linecolor='Grey', gridcolor='Gainsboro')
        
        # Update axis titles for each currency
        for i, currency in enumerate(currencies):
            fig.update_xaxes(title_text=f'{currency.upper()} Price', row=i+1)
        
        fig.update_xaxes(title_text='Volume', row=volume_row)
        fig.update_xaxes(title_text='Performance', row=performance_row)
        fig.update_xaxes(title_text='Net Worth', row=net_worth_row)
        fig.update_xaxes(title_standoff=7, title_font=dict(size=12))

        self.fig = go.FigureWidget(fig)
        
        # Store references to price charts for each currency
        self._price_charts = {}
        for i, currency in enumerate(currencies):
            self._price_charts[currency] = self.fig.data[i]
        
        # Store references to other charts (adjust indices based on number of currencies)
        chart_start_idx = n_currencies
        self._volume_chart = self.fig.data[chart_start_idx]
        self._performance_chart = self.fig.data[chart_start_idx + 1]
        self._net_worth_chart = self.fig.data[-1]

        self.fig.update_annotations({'font': {'size': 12}})
        self.fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(self,
                                  trades: 'OrderedDict',
                                  price_history: 'pd.DataFrame') -> 'Tuple[go.layout.Annotation]':
        """Creates annotations of the new trades after the last one in the chart.

        Parameters
        ----------
        trades : `OrderedDict`
            The history of trades for the current episode.
        price_history : `pd.DataFrame`
            The price history of the current episode.

        Returns
        -------
        `Tuple[go.layout.Annotation]`
            A tuple of annotations used in the renderering process.
        """
        annotations = []
        for trade in reversed(trades.values()):
            trade = trade[0]

            tp = float(trade.price)
            ts = float(trade.size)

            if trade.step <= self._last_trade_step:
                break

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
                qty = round(ts / tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
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
                # qty = round(ts * tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=ts,
                    size=round(ts * tp, trade.base_instrument.precision),
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

            annotations += [go.layout.Annotation(
                x=trade.step - 1, y=tp,
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
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

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)

        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        
        # Update price charts for each currency
        for currency in currencies:
            if currency == 'default':
                # Single currency case
                open_col, high_col, low_col, close_col = 'open', 'high', 'low', 'close'
            else:
                # Multi-currency case with prefixes
                open_col = f'{currency}_open'
                high_col = f'{currency}_high'
                low_col = f'{currency}_low'
                close_col = f'{currency}_close'
            
            self._price_charts[currency].update(dict(
                open=price_history[open_col],
                high=price_history[high_col],
                low=price_history[low_col],
                close=price_history[close_col]
            ))
        
        self.fig.layout.annotations += self._create_trade_annotations(trades, price_history)

        # Update volume chart - handle both single and multi-currency cases
        if 'volume' in price_history.columns:
            self._volume_chart.update({'y': price_history['volume']})
        else:
            # For multi-currency, we might need to aggregate volumes or use first currency
            volume_cols = [col for col in price_history.columns if col.endswith('_volume')]
            if volume_cols:
                # Use the first currency's volume as default, or sum all volumes
                total_volume = price_history[volume_cols].sum(axis=1)
                self._volume_chart.update({'y': total_volume})

        # Update performance charts (row number is now n_currencies + 2)
        performance_row = len(currencies) + 2
        for trace in self.fig.select_traces(row=performance_row):
            trace.update({'y': performance[trace.name]})

        self._net_worth_chart.update({'y': net_worth})

        if self._show_chart:
            self.fig.show()

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
        self._price_charts = {}
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)


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
                 path: str = 'charts',
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

        if self._save_format and self._path and not os.path.exists(path):
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
                 window: int = 30,
                 annualize: bool = True,
                 periods_per_year: int = 252,
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'sharpe_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._window = window
        self._annualize = annualize
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for SharpePlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='Sharpe')
            self.fig = go.FigureWidget(fig)

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
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': sr.index, 'y': sr, 'name': sr.name})
        if self._show_chart:
            display(self.fig)
            self.fig.show()

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
                 path: str = 'charts',
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
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
    """For each timestamp y, compute:
       - max_delta_rise[y]  = pnl[y] - min_{x<=y} pnl[x]
       - max_delta_fall[y]  = pnl[y] - max_{x<=y} pnl[x]   (typically ≤ 0)
       where pnl = net_worth - net_worth.iloc[0].
    """
    if net_worth is None or len(net_worth) == 0:
        raise ValueError("`net_worth` Series is empty; cannot compute PnL extrema.")
    pnl = _pnl_series(net_worth)
    running_min = pnl.cummin()
    running_max = pnl.cummax()
    max_delta_rise = pnl - running_min
    max_delta_fall = pnl - running_max
    max_delta_rise.name = "Max Δ Rise (since start)"
    max_delta_fall.name = "Max Δ Downfall (since start)"
    return max_delta_rise, max_delta_fall


class PnLExtremaPlotlyRenderer(BaseRenderer):
    """Plots time series of max delta rise and max delta downfall in two subplots (Plotly)."""
    def __init__(self,
                 display: bool = True,
                 height: int | None = 600,
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'pnl_extrema_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for PnLExtremaPlotlyRenderer.")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=("Max Δ Rise", "Max Δ Downfall"))
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=80))
            fig.add_scatter(mode='lines', name='Max Δ Rise', row=1, col=1)
            fig.add_scatter(mode='lines', name='Max Δ Downfall', row=2, col=1)
            self.fig = go.FigureWidget(fig)

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
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        # Update traces
        self.fig.data[0].update({'x': up.index, 'y': up})
        self.fig.data[1].update({'x': down.index, 'y': down})
        if self._show_chart:
            display(self.fig)
            self.fig.show()

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
    """Plots time series of max delta rise and max delta downfall in two subplots (Matplotlib)."""
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str = 'charts',
                 filename_prefix: str = 'pnl_extrema_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax_up = None
        self.ax_down = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax_up.set_title("Max Δ Rise")

        self.ax_down.plot(down.index, down)
        self.ax_down.set_title("Max Δ Downfall")

        self.fig.suptitle(self._create_log_entry(episode, max_episodes, step, max_steps))
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


# --- Calmar Renderers ---
class CalmarPlotlyRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 periods_per_year: int = 252,
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'calmar_') -> None:
        super().__init__()
        self._height = height
        self._ppy = periods_per_year
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for CalmarPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='Calmar (to date)')
            self.fig = go.FigureWidget(fig)
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        calmar = _calmar_series(net_worth, periods_per_year=self._ppy)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': calmar.index, 'y': calmar, 'name': calmar.name})
        if self._show_chart:
            display(self.fig)
            self.fig.show()
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['html','png','jpeg','webp','svg','pdf','eps']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False) if self._save_format=='html' else self.fig.write_image(filename)
    def reset(self) -> None:
        self.fig = None


class CalmarMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 periods_per_year: int = 252,
                 save_format: str | None = 'png',
                 path: str = 'charts',
                 filename_prefix: str = 'calmar_') -> None:
        super().__init__()
        self._show_chart = display
        self._ppy = periods_per_year
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
class HitRatioPlotlyRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'hitratio_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for HitRatioPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='Hit Ratio (periods)')
            self.fig = go.FigureWidget(fig)
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        rets = _returns_from_net_worth(net_worth)
        hr = _hit_ratio_series_from_returns(rets)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': hr.index, 'y': hr, 'name': hr.name})
        if self._show_chart:
            display(self.fig); self.fig.show()
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['html','png','jpeg','webp','svg','pdf','eps']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False) if self._save_format=='html' else self.fig.write_image(filename)
    def reset(self) -> None:
        self.fig = None


class HitRatioMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str = 'charts',
                 filename_prefix: str = 'hitratio_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
class TurnoverPlotlyRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 height: int | None = 500,
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'turnover_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)
    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for TurnoverPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_bar(name='Turnover (per step)')
            self.fig = go.FigureWidget(fig)
    def render_env(self, episode=None, max_episodes=None, step=None, max_steps=None,
                   price_history=None, net_worth=None, performance=None, trades=None) -> None:
        to = _turnover_series(trades, net_worth)
        self._ensure_fig()
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': to.index, 'y': to, 'name': to.name})
        if self._show_chart:
            display(self.fig); self.fig.show()
    def save(self, root_path: str | None = None) -> None:
        if not self._save_format: return
        valid_formats = ['html','png','jpeg','webp','svg','pdf','eps']
        _check_valid_format(valid_formats, self._save_format); _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(root_path or self._path, filename)
        self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=False) if self._save_format=='html' else self.fig.write_image(filename)
    def reset(self) -> None:
        self.fig = None


class TurnoverMatplotlibRenderer(BaseRenderer):
    def __init__(self,
                 display: bool = True,
                 save_format: str | None = 'png',
                 path: str = 'charts',
                 filename_prefix: str = 'turnover_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
                 save_format: str | None = None,
                 path: str = 'charts',
                 filename_prefix: str = 'roi_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for ROIPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='ROI')
            self.fig = go.FigureWidget(fig)

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
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': roi.index, 'y': roi, 'name': roi.name})
        if self._show_chart:
            display(self.fig)
            self.fig.show()

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
                 path: str = 'charts',
                 filename_prefix: str = 'roi_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'drawdown_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for DrawdownPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='Drawdown')
            self.fig = go.FigureWidget(fig)

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
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': dd.index, 'y': dd, 'name': dd.name})
        if self._show_chart:
            display(self.fig)
            self.fig.show()

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
                 path: str = 'charts',
                 filename_prefix: str = 'drawdown_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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
                 save_format: str | None = 'html',
                 path: str = 'charts',
                 filename_prefix: str = 'pnl_') -> None:
        super().__init__()
        self._height = height
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _ensure_fig(self):
        if self.fig is None:
            if not importlib.util.find_spec("plotly"):
                raise RuntimeError("Plotly is not installed but required for PnLPlotlyRenderer.")
            fig = go.Figure()
            fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
            fig.add_scatter(mode='lines', name='PnL (since start)')
            self.fig = go.FigureWidget(fig)

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
        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self.fig.data[0].update({'x': pnl.index, 'y': pnl, 'name': 'PnL (since start)'})
        if self._show_chart:
            display(self.fig)
            self.fig.show()

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
                 path: str = 'charts',
                 filename_prefix: str = 'pnl_') -> None:
        super().__init__()
        self._show_chart = display
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self.fig = None
        self.ax = None
        if self._save_format and self._path and not os.path.exists(path):
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
        self.ax.set_title(self._create_log_entry(episode, max_episodes, step, max_steps))
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

_registry = {
    "screen-log": ScreenLogger,
    "file-log": FileLogger,
    "plotly": PlotlyTradingChart,
    "matplot": MatplotlibTradingChart,
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


def construct_renderers(identifier: str, display: bool = True) -> 'BaseRenderer':
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
    
    def _get_one(identifier: str) -> 'BaseRenderer':
        if identifier not in _registry.keys():
            msg = f"Identifier {identifier} is not associated with any `BaseRenderer`."
            raise KeyError(msg)
        return _registry[identifier](display=display)
    
    print(f"Constructing renderers: {type(identifier)=}")
    
    if identifier == 'all':
        return [_get_one(i) for i in _registry.keys()]
    elif isinstance(identifier, Iterable):
        return [_get_one(i) for i in identifier]
    elif isinstance(identifier, str):
        return _get_one(identifier)
    else:
        raise ValueError(f"Invalid identifier: {identifier}")
