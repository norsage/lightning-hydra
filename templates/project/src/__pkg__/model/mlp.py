"""Stub network. Network modules live in model/, the LightningModule in model/lightning.py."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, num_layers: int = 2, out_features: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
