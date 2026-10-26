"""Training losses for calibration."""

import torch


def quantile_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles: list) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        e = y.squeeze(-1) - pred_q[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e).mean())
    return sum(losses)


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar)
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)
    return nll.mean()
