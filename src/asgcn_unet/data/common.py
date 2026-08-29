from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

_TARGET_NORMALIZATION_MODES = {
    "integer_dtype_max",
    "known_scale",
    "already_normalized",
    "percentile_debug_only",
}


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int


def normalize_polarity(p: np.ndarray) -> np.ndarray:
    p = p.astype(np.float32, copy=False)
    if p.size and p.min() >= 0:
        p = p * 2.0 - 1.0
    return np.where(p >= 0, 1.0, -1.0).astype(np.float32, copy=False)


def uniform_cap_ratio(event_count: int, max_events: int | None) -> float:
    """Return the source-to-retained ratio of the exact-size uniform cap."""
    if max_events is None or max_events <= 0 or event_count <= max_events:
        return 1.0
    return float(event_count) / float(max_events)


def stratified_subsample(events: np.ndarray, max_events: int | None) -> np.ndarray:
    """Select exactly ``max_events`` time-spread events when a cap is required.

    A ceil-stride cap has a severe boundary discontinuity: 8,193 inputs retain only
    4,097 values for an 8,192 cap.  Linspace selection keeps the requested count,
    includes both temporal endpoints, and remains deterministic.
    """
    if max_events is None or max_events <= 0 or len(events) <= max_events:
        return events.astype(np.float32, copy=False)
    indices = np.linspace(0, len(events) - 1, num=int(max_events), dtype=np.int64)
    return events[indices].astype(np.float32, copy=False)


def choose_crop(
    image_height: int,
    image_width: int,
    crop_size: tuple[int, int] | None,
    random_crop: bool,
    rng: np.random.Generator,
) -> Crop:
    if crop_size is None:
        return Crop(0, 0, image_width, image_height)
    crop_h = min(int(crop_size[0]), image_height)
    crop_w = min(int(crop_size[1]), image_width)
    if random_crop:
        top = int(rng.integers(0, image_height - crop_h + 1))
        left = int(rng.integers(0, image_width - crop_w + 1))
    else:
        top = (image_height - crop_h) // 2
        left = (image_width - crop_w) // 2
    return Crop(left, top, crop_w, crop_h)


def crop_events(events: np.ndarray, crop: Crop) -> np.ndarray:
    if len(events) == 0:
        return events
    x, y = events[:, 0], events[:, 1]
    keep = (
        (x >= crop.left)
        & (x < crop.left + crop.width)
        & (y >= crop.top)
        & (y < crop.top + crop.height)
    )
    result = events[keep].copy()
    result[:, 0] -= crop.left
    result[:, 1] -= crop.top
    return result


def validate_target_normalization(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and canonicalize the target radiometric contract.

    The default preserves the official integer image path used by EventHDR and
    EventAid-R. Float targets must opt into an explicit, auditable scale instead
    of receiving frame-dependent normalization implicitly.
    """
    if value is None:
        value = {"mode": "integer_dtype_max"}
    if not isinstance(value, dict):
        raise TypeError("target_normalization must be an object with a 'mode' field")
    if any(not isinstance(name, str) for name in value):
        raise TypeError("target_normalization field names must be strings")
    mode = value.get("mode")
    if not isinstance(mode, str) or mode not in _TARGET_NORMALIZATION_MODES:
        supported = ", ".join(sorted(_TARGET_NORMALIZATION_MODES))
        raise ValueError(f"target_normalization.mode must be one of: {supported}")

    allowed_keys = {"mode"}
    normalized: dict[str, Any] = {"mode": mode}
    if mode == "known_scale":
        allowed_keys.add("scale")
        scale = value.get("scale")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("target_normalization.scale must be a finite positive number")
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("target_normalization.scale must be a finite positive number")
        normalized["scale"] = scale
    elif mode == "percentile_debug_only":
        allowed_keys.update({"debug_only", "lower_percentile", "upper_percentile"})
        if value.get("debug_only") is not True:
            raise ValueError("percentile_debug_only requires target_normalization.debug_only=true")
        lower = value.get("lower_percentile", 0.1)
        upper = value.get("upper_percentile", 99.9)
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in (lower, upper)
        ):
            raise TypeError("target normalization percentiles must be finite numbers")
        lower, upper = float(lower), float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or not 0 <= lower < upper <= 100:
            raise ValueError(
                "target normalization percentiles must satisfy 0 <= lower < upper <= 100"
            )
        normalized.update(
            {
                "debug_only": True,
                "lower_percentile": lower,
                "upper_percentile": upper,
            }
        )
    unknown = sorted(set(value) - allowed_keys)
    if unknown:
        raise ValueError("Unknown target_normalization fields: " + ", ".join(unknown))
    return normalized


def _normalize_target_array(
    image: np.ndarray,
    normalization: dict[str, Any],
    *,
    source: str,
) -> np.ndarray:
    original_dtype = image.dtype
    if not np.issubdtype(original_dtype, np.number) or np.issubdtype(original_dtype, np.bool_):
        raise TypeError(f"Target {source} must use a real numeric dtype, got {original_dtype}")
    if image.size and not np.all(np.isfinite(image)):
        raise ValueError(f"Target {source} contains NaN or Inf values")

    mode = normalization["mode"]
    if mode == "integer_dtype_max":
        if not np.issubdtype(original_dtype, np.integer):
            raise ValueError(
                f"Target {source} uses dtype {original_dtype}, but "
                "target_normalization.mode='integer_dtype_max' requires an integer dtype"
            )
        result = image.astype(np.float32, copy=False)
        result /= float(np.iinfo(original_dtype).max)
    elif mode == "known_scale":
        result = image.astype(np.float32, copy=False) / normalization["scale"]
    elif mode == "already_normalized":
        result = image.astype(np.float32, copy=False)
    else:
        # This path is intentionally noisy and opt-in. It is useful only while
        # inspecting unknown data and must never be mistaken for a fixed protocol.
        result = image.astype(np.float32, copy=False)
        lower, upper = np.percentile(
            result,
            [normalization["lower_percentile"], normalization["upper_percentile"]],
        )
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError(
                f"Target {source} has a degenerate debug percentile range [{lower}, {upper}]"
            )
        result = np.clip((result - lower) / (upper - lower), 0.0, 1.0)

    if result.size:
        minimum = float(result.min())
        maximum = float(result.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"Target {source} is outside [0,1] after {mode} normalization "
                f"(min={minimum}, max={maximum})"
            )
    return result


def image_array_to_tensor(
    image: np.ndarray,
    target_channels: int = 1,
    tone_map: str = "none",
    tone_map_mu: float = 5000.0,
    target_normalization: dict[str, Any] | None = None,
    source: str = "image",
) -> torch.Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")

    normalization = validate_target_normalization(target_normalization)
    image = _normalize_target_array(image, normalization, source=source)

    if target_channels == 1 and image.shape[-1] != 1:
        rgb = image[..., :3]
        image = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
    elif target_channels == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] > target_channels:
        image = image[..., :target_channels]

    if tone_map == "log":
        if not np.isfinite(tone_map_mu) or tone_map_mu <= 0.0:
            raise ValueError("tone_map_mu must be finite and positive for log tone mapping")
        image = np.log1p(tone_map_mu * image) / np.log1p(tone_map_mu)
    elif tone_map not in {"none", "linear"}:
        raise ValueError(f"Unknown tone_map: {tone_map}")
    if image.size and (
        not np.all(np.isfinite(image)) or float(image.min()) < 0.0 or float(image.max()) > 1.0
    ):
        raise ValueError(f"Target {source} is non-finite or outside [0,1] after tone mapping")
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def pil_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def make_sample(
    events: np.ndarray,
    target: torch.Tensor,
    sample_id: str,
    sensor_size: tuple[int, int],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"Events must have shape Nx4 [x,y,t,p], got {events.shape}")
    event_tensor = torch.from_numpy(np.ascontiguousarray(events)).float()
    return {
        "events": event_tensor,
        "target": target,
        "sample_id": sample_id,
        "sensor_size": tuple(int(v) for v in sensor_size),
        "metadata": metadata or {},
    }
