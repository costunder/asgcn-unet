from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from asgcn_recon.metrics import structural_similarity


def _reference_gaussian_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Small unfold-based reference independent of the production convolution."""

    size = min(window_size, *prediction.shape[-2:])
    if size % 2 == 0:
        size -= 1
    coordinates = torch.arange(size, dtype=prediction.dtype)
    coordinates -= (size - 1) / 2
    kernel_1d = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    weights = torch.outer(kernel_1d, kernel_1d).flatten().view(1, 1, -1, 1)

    def patches(value: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = value.shape
        unfolded = F.unfold(value, kernel_size=size)
        return unfolded.view(batch, channels, size * size, -1)

    x = patches(prediction)
    y = patches(target)
    mu_x = (x * weights).sum(dim=2)
    mu_y = (y * weights).sum(dim=2)
    sigma_x = (x.square() * weights).sum(dim=2) - mu_x.square()
    sigma_y = (y.square() * weights).sum(dim=2) - mu_y.square()
    sigma_xy = (x * y * weights).sum(dim=2) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean().clamp(-1.0, 1.0)


def test_ssim_matches_gaussian_window_reference() -> None:
    generator = torch.Generator().manual_seed(2026)
    prediction = torch.rand((2, 2, 9, 8), generator=generator, dtype=torch.float64)
    target = torch.rand((2, 2, 9, 8), generator=generator, dtype=torch.float64)

    actual = structural_similarity(prediction, target, window_size=5)
    expected = _reference_gaussian_ssim(prediction, target, window_size=5)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("height,width", [(1, 1), (1, 7), (2, 2), (5, 8), (10, 10)])
def test_ssim_is_finite_and_exact_for_identical_small_images(height: int, width: int) -> None:
    image = torch.linspace(0.0, 1.0, height * width).reshape(1, 1, height, width)

    result = structural_similarity(image, image)

    assert torch.isfinite(result)
    torch.testing.assert_close(result, torch.ones_like(result), rtol=0.0, atol=1e-6)


def test_ssim_small_image_has_finite_gradient() -> None:
    prediction = torch.tensor([[[[0.1, 0.4], [0.8, 0.2]]]], requires_grad=True)
    target = torch.tensor([[[[0.2, 0.3], [0.7, 0.1]]]])

    loss = 1.0 - structural_similarity(prediction, target)
    loss.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ssim_promotes_mixed_precision_inputs_for_amp_training() -> None:
    prediction = torch.rand((1, 1, 8, 8), dtype=torch.bfloat16, requires_grad=True)
    target = torch.rand((1, 1, 8, 8), dtype=torch.float32)

    result = structural_similarity(prediction, target)
    result.backward()

    assert result.dtype == torch.float32
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_ssim_disables_outer_autocast_for_stable_local_statistics() -> None:
    prediction = torch.zeros((1, 1, 16, 16), requires_grad=True)
    target = torch.zeros_like(prediction)

    with torch.autocast(device_type="cpu", dtype=torch.float16):
        result = structural_similarity(prediction, target)
    result.backward()

    torch.testing.assert_close(result, torch.ones_like(result), rtol=0.0, atol=1e-6)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
