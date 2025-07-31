from scripts.fetch_currency import fetch_historical_data
from pathlib import Path
import pandas as pd
import os
from .metrics import summarize_performance

from policy import BasePolicy

class StockSimulator:
    
    def _acquire_data(
        self, 
        coin_name: str, 
        path_to_storage: str = 'data/',
        interval: str | None = None,
        days: int | None = None,
    ):
        path_input = Path(path_to_storage)
        data_name  = coin_name + '.csv'
        
        # check if required data is downloaded or needs to be renewed
        if (not data_name in os.listdir(path_input)) or \
            (interval is not None) or \
            (days is not None):
                
            # download coin data with presetted data
            fetch_historical_data(
                currency=coin_name,
                interval=interval if interval is not None else '1d',
                days=days if days is not None else 0,
                output=path_input / data_name
            )
        
        # reading data and return ts
        return pd.read_csv(path_input / data_name)[['open', 'high', 'low', 'close']]
        
    
    def _core_simulator(self, 
        model: BasePolicy,
        stock_series: pd.DataFrame,
        warm_start: int | float = 0.2
    ):
        assert not isinstance(warm_start, int) or 1 < warm_start < len(stock_series)
        assert not isinstance(warm_start, float) or 0 < warm_start < 1
        
        # devise series into fitting and inference segments
        if isinstance(warm_start, float):
            warm_start = int(len(stock_series) * warm_start)
        fit_series  = stock_series.iloc[:warm_start]
        data_series = stock_series.iloc[warm_start:]
        
        # fit strategy
        model.fit_historycal_data(fit_series)

        # Initialize trading variables
        predictions = [] # store binary predictions (1/-1) of strategy

        # Trading simulation
        for i in range(warm_start, len(data_series) - 1):
            # make prediction for next day market tendency
            pred = model.get_action(data_series.iloc[i])
            predictions.append(pred)
            
            # add current day to model memory
            observation = data_series.iloc[i]
            
            # Firstly, we add correct prediction on current action policy predicted
            model.add_new_day_information(observation)
            
            # Secondly, we add current observation to model memory
            model.add_observation(observation)


        # getting whole metrics
        metrics = summarize_performance(
            predictions,
            data_series
        )
        
        # Store results in simulator_run
        self.simulator_run['final_metrics'] = metrics
        self.simulator_run['predictions'] = predictions
        self.simulator_run['stock_series'] = stock_series
        self.simulator_run['model'] = model

        return metrics


    def __init__(self):
        self.simulator_run = {
            'final_metrics': None,
            'predictions': None,
            'stock_series': None,
            'model': None
        }


    def run_simulation(
        self, 
        model: BasePolicy,
        coin_name: str | list[str],
        path_to_storage: str = 'data/',
        warm_start: int | float = 0.2,
        interval: str | None = None,
        days: int | None = None,
    ):
        """
        Run the stock simulation with the given model and parameters.
        
        Parameters:
            model (BasePolicy): The trading strategy model to be used.
            coin_name (str): The name of the cryptocurrency to simulate.
            path_to_storage (str): Path to store or fetch historical data.
            warm_start (int | float): Initial period for fitting the model.
            interval (str | None): Data interval, e.g., '1d', '1h'. Necessary for fetching new data.
            days (int | None): Number of days of historical data to fetch. Necessary for fetching new data.
        
        Returns:
            dict: Final metrics of the simulation.
        """
        
        # acquire data
        stock_series = self._acquire_data(
            coin_name=coin_name,
            path_to_storage=path_to_storage,
            interval=interval,
            days=days
        )
        
        # run core simulator and return metrics
        return self._core_simulator(
            model=model,
            stock_series=stock_series,
            warm_start=warm_start
        )
        
    def get_simulation_results(self):
        """        Returns:
            dict: The results of the last simulation run.
        """
        return self.simulator_run
    
    def visualize_results(self):
        """
        Visualizes the results of the last simulation run.
        
        Returns:
            None: Displays the performance metrics and predictions.
        """
        raise NotImplementedError("Visualization logic is not implemented yet.")