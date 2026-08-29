from __future__ import annotations

import pytest
import torch

from asgcn_recon.metrics import MetricAccumulator, temporal_consistency_error


def test_temporal_consistency_error_compares_frame_changes() -> None:
    previous_prediction = torch.zeros((1, 1, 2, 2))
    prediction = torch.full((1, 1, 2, 2), 0.5)
    previous_target = torch.zeros((1, 1, 2, 2))
    target = torch.full((1, 1, 2, 2), 0.25)

    result = temporal_consistency_error(
        prediction, previous_prediction, target, previous_target
    )

    assert float(result) == pytest.approx(0.25)


def test_metric_accumulator_supports_temporal_metric_after_first_frame() -> None:
    accumulator = MetricAccumulator()
    accumulator.update("scene-a", "a/0", {"ssim": 0.8})
    accumulator.update("scene-a", "a/1", {"ssim": 0.9, "temporal_l1": 0.1})
    accumulator.update("scene-b", "b/0", {"ssim": 0.7})
    accumulator.update("scene-b", "b/1", {"ssim": 0.6, "temporal_l1": 0.3})

    summary = accumulator.summary()

    assert summary["micro"]["temporal_l1"] == pytest.approx(0.2)
    assert summary["macro"]["temporal_l1"] == pytest.approx(0.2)
    assert summary["per_scene"]["scene-a"]["temporal_l1_frames"] == 1
