"""Uncertainty heads for calibrated forecasting."""

import torch
import torch.nn as nn


class QuantileHead(nn.Module):
    def __init__(self, d_model: int = 128, quantiles: list = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in quantiles])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [h(x) for h in self.heads]
        return torch.cat(outs, dim=-1)  # (B, len(quantiles))


class GaussianHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.mu = nn.Linear(d_model, 1)
        self.logvar = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        mu = self.mu(x)
        logvar = self.logvar(x).clamp(-10, 10)
        return mu, logvar
