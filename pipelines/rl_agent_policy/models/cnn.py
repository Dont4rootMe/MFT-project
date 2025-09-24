import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    """Simple 1D convolutional backbone used by the A2C agent.

    Parameters
    ----------
    input_shape: tuple
        Shape of the observation without the batch dimension (C, L).
    """

    def __init__(self, input_shape: tuple):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[0],
            out_channels=64,
            kernel_size=6,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding="same",
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            x = self._forward_conv(dummy)
            self.output_dim = x.shape[1]

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_conv(x)
