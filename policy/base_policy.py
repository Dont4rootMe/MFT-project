from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Sequence, Mapping, Any
import pandas as pd


class ObservationHub:
    """
    Unified container for market observations consumed by trading strategies.
    Stores a growing (optionally bounded) sequence of observations with a stable
    column schema. Supports loading historical data, appending new observations,
    and retrieving the last k observations as a DataFrame.
    """

    def __init__(self, columns: Optional[Sequence[str]] = None, maxlen: Optional[int] = None) -> None:
        self._columns = list(columns) if columns is not None else None
        self._buffer = deque(maxlen=maxlen)  # deque of dict records

    def load_historical_data(self, time_series: pd.DataFrame) -> None:
        """
        Load and store historical data. Existing buffer is replaced.
        Parameters
        ----------
        time_series : pd.DataFrame
            Historical market data with columns matching (or defining) the schema.
        """
        if not isinstance(time_series, pd.DataFrame):
            raise TypeError("time_series must be a pandas.DataFrame")

        # Establish schema if not set yet; otherwise, ensure compatibility and reorder
        if self._columns is None:
            self._columns = list(time_series.columns)
        else:
            missing = [c for c in self._columns if c not in time_series.columns]
            if missing:
                raise ValueError(f"time_series is missing required columns: {missing}")
            time_series = time_series[self._columns]

        # Replace buffer with records from the DataFrame
        self._buffer.clear()
        self._buffer.extend(time_series.to_dict(orient="records"))

    def add_observation(self, observation: Any) -> None:
        """
        Append a single observation. Accepts:
        - pandas.Series (indexed by column names)
        - Mapping (e.g., dict) with keys matching the schema
        - Iterable of values matching the schema order
        """
        # Determine schema if unknown and observation is labeled
        if self._columns is None:
            if isinstance(observation, pd.Series):
                self._columns = list(observation.index)
            elif isinstance(observation, Mapping):
                self._columns = list(observation.keys())
            else:
                raise ValueError(
                    "Schema unknown. Load historical data first or provide a labeled Series/dict observation."
                )

        # Normalize to a dict record following the schema order
        if isinstance(observation, pd.Series):
            record = observation.reindex(self._columns).to_dict()
        elif isinstance(observation, Mapping):
            try:
                record = {c: observation[c] for c in self._columns}
            except KeyError as e:
                raise ValueError(f"Observation is missing required field: {e}") from e
        else:
            # Treat as positional iterable
            try:
                values = list(observation)  # may raise TypeError for non-iterables
            except TypeError as e:
                raise TypeError("Unsupported observation type; expected Series, Mapping, or iterable of values") from e
            if len(values) != len(self._columns):
                raise ValueError(
                    f"Observation length {len(values)} does not match number of columns {len(self._columns)}"
                )
            record = dict(zip(self._columns, values))

        self._buffer.append(record)

    def get_observations(self, k: int | None | str) -> pd.DataFrame:
        """
        Return the last k observations as a DataFrame (k <= 0 returns empty).
        """
        if k is None or k == 'all':
            k = len(self._buffer)
        elif isinstance(k, str) and k.isdigit():
            k = int(k)
        
        if k <= 0:
            return pd.DataFrame(columns=self._columns or [])
        if self._columns is None:
            return pd.DataFrame()
        k = min(k, len(self._buffer))
        if k == len(self._buffer):
            data = list(self._buffer)
        else:
            data = list(self._buffer)[-k:]
        return pd.DataFrame.from_records(data, columns=self._columns)

    def __repr__(self) -> str:
        cols = self._columns or []
        return (
            f"ObservationHub(n={len(self)}, columns={cols}, maxlen={self._buffer.maxlen})"
        )

    def __str__(self) -> str:
        n = len(self)
        cols = self._columns or []
        k = min(3, n)
        if k > 0:
            records = list(self._buffer)[-k:]
            df_tail = pd.DataFrame.from_records(records, columns=self._columns or None)
            tail_str = df_tail.to_string(index=False)
        else:
            tail_str = "<empty>"
        return (
            "ObservationHub\n"
            f"  columns: {cols}\n"
            f"  n: {n} (maxlen={self._buffer.maxlen})\n"
            f"  tail({k}):\n{tail_str}"
        )

    def __len__(self) -> int:
        return len(self._buffer)


class BasePolicy(ABC):
    """
    Base class for trading strategies.
    """
    
    def __init__(self): 
        self.observation_hub = ObservationHub()

    # ============================================
    #         - Methods of data sdtorage -        
    # ============================================

    def fit_historycal_data(self, time_series: pd.DataFrame) -> None:
        """
        Fit the strategy to historical data.
        Parameters
        ----------
        time_series : pd.DataFrame
            Historical market data with columns 'open', 'high', 'low', 'close'.
        """
        self.observation_hub.load_historical_data(time_series)
        self.fit_method()


    def add_observation(self, observation: Any) -> None:
        """
        Add a new observation to the strategy's memory.
        Parameters
        ----------
        observation : Any
            A single market observation (e.g., a row from a DataFrame).
        """
        self.observation_hub.add_observation(observation)

    # ============================================
    #             - Abstract Methods -            
    # ============================================

    @abstractmethod
    def get_action(self, reference: pd.Series | None) -> int:
        """
        Get the action for the next time step.
        Returns
        -------
        int
            Action to take (e.g., 1 for buy, -1 for sell).
            
        reference : pd.Series | None
            The current market observation to base the action on.
        """
        raise NotImplementedError("Subclasses must implement get_action method")
    
    @abstractmethod
    def fit_method(self) -> int:
        """
        Fit the model to the historical data.
        Returns
        -------
        int
            Status code indicating success (0) or failure (1).
        """
        raise NotImplementedError("Subclasses must implement fit_method")
    
    @abstractmethod
    def add_new_day_information(self, reference: pd.Series) -> None:
        """
        Add new day information to the model's memory.
        Parameters
        ----------
        reference : pd.Series
            A single market observation (e.g., a row from a DataFrame).
        """
        raise NotImplementedError("Subclasses must implement add_new_day_information method")