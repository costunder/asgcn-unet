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
    """Compute mean SSIM with the standard Gaussian local-statistics window.

    The default 11x11 window and sigma of 1.5 follow the original SSIM
    formulation.  For images smaller than the requested window, the largest
    fitting odd window is used so that validation crops of any positive size
    remain supported.
    """

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4:
        raise ValueError("prediction and target must be BCHW tensors")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    min_side = min(prediction.shape[-2:])
    if min_side < 1:
        raise ValueError("prediction and target must have non-empty spatial dimensions")
    size = min(window_size, min_side)
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)

    # CUDA autocast commonly produces float16 predictions against float32 targets.
    # Promote local statistics to at least float32 while preserving gradients.
    computation_dtype = torch.promote_types(prediction.dtype, target.dtype)
    if computation_dtype in {torch.float16, torch.bfloat16}:
        computation_dtype = torch.float32
    # An outer training autocast context would otherwise cast conv2d back to
    # float16 after the explicit promotion above, defeating the stabilization.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.to(dtype=computation_dtype)
        target = target.to(dtype=computation_dtype)

        # Build the Gaussian in the same stable computation dtype.
        kernel_dtype = torch.float64 if computation_dtype == torch.float64 else torch.float32
        coordinates = torch.arange(size, dtype=kernel_dtype, device=prediction.device)
        coordinates = coordinates - (size - 1) / 2
        gaussian_1d = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).to(dtype=computation_dtype)
        channels = prediction.shape[1]
        window = gaussian_2d.expand(channels, 1, size, size).contiguous()

        def local_mean(value: torch.Tensor) -> torch.Tensor:
            return F.conv2d(value, window, groups=channels)

        mu_x = local_mean(prediction)
        mu_y = local_mean(target)
        sigma_x = local_mean(prediction * prediction) - mu_x.square()
        sigma_y = local_mean(target * target) - mu_y.square()
        sigma_xy = local_mean(prediction * target) - mu_x * mu_y
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
        minimum = torch.finfo(denominator.dtype).tiny
        return (numerator / denominator.clamp_min(minimum)).mean().clamp(-1.0, 1.0)


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse.clamp_min(1e-12))


def temporal_consistency_error(
    prediction: torch.Tensor,
    previous_prediction: torch.Tensor,
    target: torch.Tensor,
    previous_target: torch.Tensor,
) -> torch.Tensor:
    """Measure L1 error between predicted and target frame-to-frame changes.

    This is a no-flow temporal consistency diagnostic. It is intentionally named
    ``temporal_l1`` in reports so it cannot be confused with a flow-warped metric.
    """

    shapes = {
        prediction.shape,
        previous_prediction.shape,
        target.shape,
        previous_target.shape,
    }
    if len(shapes) != 1 or prediction.ndim != 4:
        raise ValueError("all temporal metric inputs must have the same BCHW shape")
    predicted_change = prediction - previous_prediction
    target_change = target - previous_target
    return F.l1_loss(predicted_change, target_change)


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
        names = sorted(
            {
                key
                for frame in self.frames
                for key in frame
                if key not in {"scene", "sample_id"}
            }
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for frame in self.frames:
            grouped[str(frame["scene"])].append(frame)
        per_scene: dict[str, dict[str, float | int]] = {}
        for scene, items in grouped.items():
            scene_summary: dict[str, float | int] = {"frames": len(items)}
            for name in names:
                values = [float(item[name]) for item in items if name in item]
                if values:
                    scene_summary[name] = sum(values) / len(values)
                    scene_summary[f"{name}_frames"] = len(values)
            per_scene[scene] = scene_summary
        micro = {}
        for name in names:
            values = [float(item[name]) for item in self.frames if name in item]
            if values:
                micro[name] = sum(values) / len(values)
        macro = {
            name: sum(float(scene[name]) for scene in per_scene.values() if name in scene)
            / sum(1 for scene in per_scene.values() if name in scene)
            for name in names
            if any(name in scene for scene in per_scene.values())
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
