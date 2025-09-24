import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorHead(nn.Module):
    """MLP head that outputs action logits."""

    def __init__(self, in_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CriticHead(nn.Module):
    """MLP head that predicts the state value."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
