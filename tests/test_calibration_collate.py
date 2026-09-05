"""Synthetic CPU calibration-input regressions, not production performance data."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from asgcn_unet.batching import (
    PackedSampleBatch,
    move_batch,
    pack_calibration_samples,
    pack_samples,
)
from tests.test_batching import _model
from tests.test_evaluation_batches import DiagnosticDataset


@pytest.fixture(autouse=True, scope="module")
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_calibration_collate_preserves_sources_and_never_stacks_targets(monkeypatch):
    raw = DiagnosticDataset(lengths=(4, 2, 1)).values
    targets = [(sample["target"], sample["target"].clone()) for sample in raw]
    expected_events = torch.cat([sample["events"] for sample in raw])

    def forbidden_stack(*args, **kwargs):
        pytest.fail("Calibration collation must not stack unused target tensors")

    with monkeypatch.context() as context:
        context.setattr(torch, "stack", forbidden_stack)
        packed = pack_calibration_samples(raw)

    assert isinstance(packed, PackedSampleBatch)
    assert packed.targets is None
    assert packed.events.device.type == "cpu"
    assert packed.sensor_size == tuple(raw[0]["sensor_size"])
    assert packed.event_counts == tuple(sample["events"].shape[0] for sample in raw)
    torch.testing.assert_close(packed.events, expected_events, rtol=0, atol=0)
    for view, original, (target, snapshot) in zip(packed, raw, targets, strict=True):
        assert set(view) == set(original) - {"target"}
        assert view["sample_id"] == original["sample_id"]
        assert view["sensor_size"] == original["sensor_size"]
        assert view["metadata"] is original["metadata"]
        torch.testing.assert_close(view["events"], original["events"], rtol=0, atol=0)
        assert original["target"] is target
        torch.testing.assert_close(original["target"], snapshot, rtol=0, atol=0)


def test_calibration_batch_transfer_routes_only_the_packed_events(monkeypatch):
    # Trace Tensor.to on CPU; no accelerator is queried or used by this test.
    packed = pack_calibration_samples(DiagnosticDataset().values)
    transfers = []
    original_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        transfers.append(tensor)
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "to", tracked_to)
        transferred = move_batch(packed, torch.device("cpu"))
    assert len(transfers) == 1 and transfers[0] is packed.events
    assert transferred.targets is None
    assert all("target" not in sample for sample in transferred)
    torch.testing.assert_close(transferred.events, packed.events, rtol=0, atol=0)


@pytest.mark.parametrize("workers", [0, 2])
def test_calibration_collate_is_spawn_compatible_and_keeps_full_coverage(workers):
    dataset = DiagnosticDataset(lengths=(4, 2, 1))
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=workers,
        collate_fn=pack_calibration_samples, pin_memory=False, persistent_workers=False,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(814),
    )
    observed = []
    events = []
    sizes = []
    for packed in loader:
        assert isinstance(packed, PackedSampleBatch)
        assert packed.targets is None and packed.events.device.type == "cpu"
        assert all("target" not in sample for sample in packed)
        sizes.append(len(packed))
        observed.extend(sample["sample_id"] for sample in packed)
        events.append(packed.events)
    assert sizes == [2, 2, 2, 1]
    assert observed == [sample["sample_id"] for sample in dataset.values]
    torch.testing.assert_close(
        torch.cat(events), torch.cat([sample["events"] for sample in dataset.values]),
        rtol=0, atol=0,
    )
    assert all("target" in sample for sample in dataset.values)


def test_event_only_collate_preserves_calibration_maxima_counts_and_graphs():
    raw = DiagnosticDataset(lengths=(4, 2, 1)).values
    reference = _model().eval()
    event_only = copy.deepcopy(reference)
    expected = reference.calibrate_batch(pack_samples(raw))
    actual = event_only.calibrate_batch(pack_calibration_samples(raw))
    assert actual["nodes"] == expected["nodes"]
    assert actual["event_counts"] == expected["event_counts"]
    assert actual["sensor_size"] == expected["sensor_size"]
    torch.testing.assert_close(actual["edges"], expected["edges"], rtol=0, atol=0)
    assert event_only.calibration_summary() == reference.calibration_summary()
    summary = event_only.calibration_summary()
    assert summary["attempted_samples"] == len(raw)
    valid = sum(sample["events"].shape[0] > 0 for sample in raw)
    assert summary["valid_samples_per_layer"] == [valid] * len(event_only.encoder.layers)
    for actual_layer, expected_layer in zip(
        event_only.encoder.layers, reference.encoder.layers, strict=True
    ):
        torch.testing.assert_close(
            actual_layer.calibration_activation_max, expected_layer.calibration_activation_max,
            rtol=0, atol=0,
        )


def test_calibration_collate_rejects_non_cpu_events_without_accessing_cuda():
    sample = dict(DiagnosticDataset()[0])
    sample["events"] = torch.empty((1, 4), device="meta")
    with pytest.raises(ValueError, match="CPU event tensors"):
        pack_calibration_samples([sample])


def test_calibration_collate_does_not_replace_an_empty_batch_with_dummy_data():
    with pytest.raises(ValueError, match="empty sample batch"):
        pack_calibration_samples([])
