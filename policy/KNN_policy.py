from .base_policy import BasePolicy
import numpy as np
import pandas as pd
from scipy.special import softmax

class KNNPolicy(BasePolicy):
    """
    K-Nearest Neighbors (KNN) trading policy.
    """

    def __init__(self, 
        day_window = 10, 
        k_neighbours = 1, 
        weighted = True,
        metric = 'euclidean'
    ): 
        super().__init__()

        # settings of model
        self.day_window   = day_window
        self.k_neighbours = k_neighbours
        self.weighted     = weighted
        
        if metric == 'euclidean':
            self.metric = lambda x, y: np.linalg.norm(x - y)
        elif metric == 'manhattan':
            self.metric = lambda x, y: np.abs(x - y).sum()
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'manhattan'.")

        # model's memory
        self.neighbours   = []
        
    def add_object(self, label, x):
        self.neighbours.append((label, x))

    # ============================================
    #        - Redefined abstract methods -       
    # ============================================

    def fit_method(self):
        time_series = self.observation_hub.get_observations('all')
        for idx in range(len(time_series) - self.day_window):
            # Assuming 'label' is a column in the DataFrame indicating the action
            label = 1 if time_series.iloc[idx + self.day_window]['close'] > time_series.iloc[idx + self.day_window]['open'] else -1
            x = time_series.iloc[idx:idx + self.day_window][['open', 'close']]
            x = (x['open'] - x['close']).values
    
            self.add_object(label, x)


    def get_action(self) -> int:
        """
        Get the action for the next time step based on KNN.

        Returns
        -------
        int
            Action to take (e.g., 1 for buy, -1 for sell).
        """
        
        # get the last k observations
        x = self.observation_hub.get_observations(self.day_window)[['open', 'close']]
        x = (x['open'] - x['close']).values
        
        scores = []
        labels = []
        for label, neighbour in self.neighbours:
            labels.append(label)
            scores.append(self.metric(x, neighbour))
            
        scores     = np.array(scores)
        labels     = np.array(labels)
        best_match = np.argpartition(-scores, self.k_neighbours)[:self.k_neighbours]
        
        if self.weighted:
            weights = softmax(1 / scores[best_match])
            return 2 * (labels[best_match] @ weights > 0) - 1
        else:
            return 2 * (labels[best_match].mean() > 0) - 1


    def add_new_day_information(self, reference: pd.Series) -> None:
        """
        Add a new observation to the strategy's memory.
        
        Parameters
        ----------
        reference : pd.Series
            A single market observation (e.g., a row from a DataFrame).
        """
        # Convert the reference Series to a DataFrame for consistency
        label = (reference['open'] - reference['close']) > 0
        x = self.observation_hub.get_observations(self.day_window)[['open', 'close']]
        x = (x['open'] - x['close']).values
        
        self.add_object(1 if label else -1, x)
        