from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Any

import torch
from torch.nn import functional as F

PSNR_MSE_FLOOR = 1e-12


@lru_cache(maxsize=32)
def _gaussian_window(
    device_name: str,
    dtype: torch.dtype,
    size: int,
    channels: int,
) -> torch.Tensor:
    """Build an immutable SSIM window, bounded by a small device-aware cache."""

    device = torch.device(device_name)
    kernel_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    coordinates = torch.arange(size, dtype=kernel_dtype, device=device)
    coordinates = coordinates - (size - 1) / 2
    gaussian_1d = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d).to(dtype=dtype)
    return gaussian_2d.expand(channels, 1, size, size).contiguous()


def structural_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    *,
    reduction: str = "mean",
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
    if reduction not in {"mean", "none"}:
        raise ValueError("SSIM reduction must be 'mean' or 'none'")

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

        channels = prediction.shape[1]
        window = _gaussian_window(
            str(prediction.device), computation_dtype, size, channels
        )

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
        similarity = numerator / denominator.clamp_min(minimum)
        if reduction == "none":
            return similarity.flatten(1).mean(1).clamp(-1.0, 1.0)
        return similarity.mean().clamp(-1.0, 1.0)


def _psnr_from_mse(mse: torch.Tensor, data_range: float) -> torch.Tensor:
    """Return finite PSNR, capped at 120 dB for unit-range exact matches."""
    if data_range <= 0:
        raise ValueError("data_range must be positive")
    return 10.0 * torch.log10((data_range**2) / mse.clamp_min(PSNR_MSE_FLOOR))


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
    extra_metrics: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    mse = F.mse_loss(prediction, target)
    metric_tensors = {
        "psnr": _psnr_from_mse(mse, 1.0),
        "ssim": structural_similarity(prediction, target),
        "rmse": torch.sqrt(mse),
    }
    if lpips_model is not None:
        pred3 = prediction.repeat(1, 3, 1, 1) if prediction.shape[1] == 1 else prediction
        target3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        metric_tensors["lpips"] = lpips_model(pred3 * 2 - 1, target3 * 2 - 1).mean()
    if extra_metrics:
        for name, value in extra_metrics.items():
            if name in metric_tensors:
                raise ValueError(f"Duplicate frame metric: {name}")
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(f"Extra frame metric {name!r} must be a scalar tensor")
            metric_tensors[name] = value.reshape(())

    names = list(metric_tensors)
    # One packed transfer avoids one CUDA synchronization per individual metric.
    values = torch.stack([metric_tensors[name].reshape(()) for name in names])
    cpu_values = values.detach().to(device="cpu", dtype=torch.float64).tolist()
    return dict(zip(names, (float(value) for value in cpu_values), strict=True))


@torch.no_grad()
def batch_frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module | None = None,
    extra_metrics: dict[str, torch.Tensor] | None = None,
) -> list[dict[str, float]]:
    """Per-frame reductions, one SSIM convolution batch and one metrics transfer."""
    if prediction.ndim != 4 or prediction.shape != target.shape or not prediction.shape[0]:
        raise ValueError("Batch metrics require matching nonempty BCHW tensors")
    mse = (prediction - target).square().flatten(1).mean(1)
    values = {
        "psnr": _psnr_from_mse(mse, 1.0),
        "ssim": structural_similarity(prediction, target, reduction="none"),
        "rmse": mse.sqrt(),
    }
    if lpips_model is not None:
        pred3 = prediction.repeat(1, 3, 1, 1) if prediction.shape[1] == 1 else prediction
        target3 = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
        values["lpips"] = lpips_model(pred3 * 2 - 1, target3 * 2 - 1).flatten(1).mean(1)
    for name, tensor in (extra_metrics or {}).items():
        if name in values or tensor.shape != (prediction.shape[0],):
            raise ValueError(f"Invalid per-frame extra metric: {name}")
        values[name] = tensor
    names = list(values)
    rows = torch.stack([values[name] for name in names], dim=1).double().cpu().tolist()
    return [dict(zip(names, row, strict=True)) for row in rows]


class MetricAccumulator:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    def update(self, scene: str, sample_id: str, metrics: dict[str, float]) -> None:
        self.frames.append({"scene": scene, "sample_id": sample_id, **metrics})

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {"frames": 0, "micro": {}, "macro": {}, "per_scene": {}}
        names = sorted(
            {key for frame in self.frames for key in frame if key not in {"scene", "sample_id"}}
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
