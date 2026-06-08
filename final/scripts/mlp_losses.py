from __future__ import annotations

import torch


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return weighted MSE normalized by the sum of weights."""
    squared_error = (pred - target) ** 2
    return (weight * squared_error).sum() / weight.sum().clamp_min(eps)
