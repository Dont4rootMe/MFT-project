from typing import Optional, Tuple

import torch
from torch import nn


class PriceRNN(nn.Module):
    """Simple RNN model for sequence of OHLC prices."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM and return class/regression heads and the new hidden state."""

        out, hidden = self.rnn(x, hidden) if hidden is not None else self.rnn(x)
        last = out[:, -1, :]
        cls = self.classifier(last)
        reg = self.regressor(last).squeeze(-1)
        return cls, reg, hidden

