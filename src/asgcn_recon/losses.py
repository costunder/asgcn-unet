from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import structural_similarity


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3):
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class ReconstructionLoss(nn.Module):
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        defaults = {"charbonnier": 1.0, "ssim": 0.2, "gradient": 0.1}
        self.weights = defaults | (weights or {})

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        terms: dict[str, torch.Tensor] = {
            "charbonnier": charbonnier_loss(prediction, target),
            "ssim": 1.0 - structural_similarity(prediction, target),
            "gradient": gradient_loss(prediction, target),
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        values: dict[str, Any] = {name: float(value.detach().cpu()) for name, value in terms.items()}
        values["total"] = float(total.detach().cpu())
        return total, values
