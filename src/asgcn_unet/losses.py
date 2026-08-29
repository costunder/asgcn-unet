from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import structural_similarity

RECONSTRUCTION_LOSS_NAMES = frozenset({"charbonnier", "ssim", "gradient"})
SUPPORTED_LOSS_WEIGHT_NAMES = RECONSTRUCTION_LOSS_NAMES | {"temporal"}


def validate_loss_weights(weights: dict[str, Any] | None) -> dict[str, float]:
    if weights is None:
        return {}
    if not isinstance(weights, dict):
        raise TypeError("train.loss_weights must be an object")
    if any(not isinstance(name, str) for name in weights):
        raise TypeError("train.loss_weights keys must be strings")
    unknown = sorted(set(weights) - SUPPORTED_LOSS_WEIGHT_NAMES)
    if unknown:
        raise ValueError("Unknown train.loss_weights keys: " + ", ".join(unknown))
    normalized: dict[str, float] = {}
    for name, raw_weight in weights.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"train.loss_weights.{name} must be a finite nonnegative number")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"train.loss_weights.{name} must be a finite nonnegative number")
        normalized[name] = weight
    return normalized


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-3):
    return torch.sqrt((prediction - target).square() + epsilon**2).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class ReconstructionLoss(nn.Module):
    def __init__(self, weights: dict[str, Any] | None = None) -> None:
        super().__init__()
        defaults = {"charbonnier": 1.0, "ssim": 0.2, "gradient": 0.1}
        configured = validate_loss_weights(weights)
        self.weights = defaults | {
            name: weight for name, weight in configured.items() if name in RECONSTRUCTION_LOSS_NAMES
        }

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms: dict[str, torch.Tensor] = {
            "charbonnier": charbonnier_loss(prediction, target),
            "ssim": 1.0 - structural_similarity(prediction, target),
            "gradient": gradient_loss(prediction, target),
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        return total, {name: value.detach() for name, value in terms.items()}
