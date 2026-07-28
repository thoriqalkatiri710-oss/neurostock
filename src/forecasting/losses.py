import math

import torch
from torch import nn


class GaussianNLLLoss(nn.Module):
    """
    Negative Log-Likelihood untuk distribusi Gaussian (Persamaan 1.1.1).
    Network memprediksi mu dan log_var secara terpisah.

    NLL = 0.5 * (log(2π σ²) + (y - μ)² / σ²)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var) + self.eps
        nll = 0.5 * (torch.log(2 * math.pi * var) + (target - mu) ** 2 / var)
        return nll.mean()


class QuantileLoss(nn.Module):
    """
    Pinball loss untuk multi-quantile forecasting (Persamaan 1.1.2).
    Memungkinkan model menghasilkan prediction interval (mis. P10-P90).
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds shape: (batch, horizon, n_quantiles)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        return torch.stack(losses).sum()