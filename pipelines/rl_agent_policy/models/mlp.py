import torch
import torch.nn as nn

class MLPBackbone(nn.Module):
    """Fully connected backbone as an alternative to the CNN version."""

    def __init__(self, input_shape: tuple):
        super().__init__()
        in_dim = input_shape[0] * input_shape[1]
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output_dim = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
