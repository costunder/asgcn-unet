from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    min_side = min(prediction.shape[-2:])
    size = min(window_size, min_side)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    padding = size // 2
    mu_x = F.avg_pool2d(prediction, size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, size, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean().clamp(-1.0, 1.0)


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse.clamp_min(1e-12))


@torch.no_grad()
def frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module | None = None,
) -> dict[str, float]:
    result = {
        "psnr": float(peak_signal_to_noise_ratio(prediction, target).cpu()),
        "ssim": float(structural_similarity(prediction, target).cpu()),
        "rmse": float(torch.sqrt(F.mse_loss(prediction, target)).cpu()),
    }
    if lpips_model is not None:
        pred3 = prediction.repeat(1, 3, 1, 1) if prediction.shape[1] == 1 else prediction
        target3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        result["lpips"] = float(lpips_model(pred3 * 2 - 1, target3 * 2 - 1).mean().cpu())
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def update(self, scene: str, sample_id: str, metrics: dict[str, float]) -> None:
        self.frames.append({"scene": scene, "sample_id": sample_id, **metrics})

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {"frames": 0, "micro": {}, "macro": {}, "per_scene": {}}
        names = [key for key in self.frames[0] if key not in {"scene", "sample_id"}]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for frame in self.frames:
            grouped[str(frame["scene"])].append(frame)
        per_scene = {
            scene: {
                "frames": len(items),
                **{name: sum(item[name] for item in items) / len(items) for name in names},
            }
            for scene, items in grouped.items()
        }
        micro = {name: sum(item[name] for item in self.frames) / len(self.frames) for name in names}
        macro = {
            name: sum(scene[name] for scene in per_scene.values()) / len(per_scene) for name in names
        }
        return {"frames": len(self.frames), "micro": micro, "macro": macro, "per_scene": per_scene}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)
