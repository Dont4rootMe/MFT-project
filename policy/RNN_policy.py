import torch
import pandas as pd

from .base_policy import BasePolicy
from .model import PriceRNN


class RNNPolicy(BasePolicy):
    """Policy using a pre-trained PriceRNN model.

    When ``ts_len`` is ``None`` the policy keeps the LSTM hidden state between
    calls, processing one new day at a time via :meth:`add_new_day_information`.
    """

    def __init__(self, model_path: str, ts_len: int | None = 30):
        super().__init__()
        self.ts_len = ts_len
        self.model = PriceRNN()
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._next_prob: float | None = None

    # ============================================
    #        - Redefined abstract methods -
    # ============================================

    def fit_method(self) -> int:  # pragma: no cover - no fitting required
        return 0

    def get_action(self, reference: pd.Series | None = None) -> int:
        if self.ts_len is None:
            # If no prediction has been computed yet (e.g. at start), run through
            # all available history once to initialize hidden state.
            if self._next_prob is None:
                data = self.observation_hub.get_observations("all")
                if len(data) == 0:
                    return 0
                x = torch.tensor(
                    data[["open", "high", "low", "close"]].values,
                    dtype=torch.float32,
                ).unsqueeze(0)
                with torch.no_grad():
                    logits, _, self.hidden = self.model(x, self.hidden)
                    self._next_prob = torch.softmax(logits, dim=-1)[0, 1].item()

            prob = self._next_prob if self._next_prob is not None else 0.0
            return 1 if prob > 0.5 else -1

        data = self.observation_hub.get_observations(self.ts_len)
        if len(data) < self.ts_len:
            return 0  # not enough data
        x = torch.tensor(
            data[["open", "high", "low", "close"]].values, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.model(x)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return 1 if prob > 0.5 else -1

    def add_new_day_information(self, reference: pd.Series) -> None:
        self.observation_hub.add_observation(reference)
        if self.ts_len is None:
            x = torch.tensor(
                reference[["open", "high", "low", "close"]].values,
                dtype=torch.float32,
            ).view(1, 1, 4)
            with torch.no_grad():
                logits, _, self.hidden = self.model(x, self.hidden)
                self._next_prob = torch.softmax(logits, dim=-1)[0, 1].item()

