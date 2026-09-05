"""Tiny synthetic CPU validation regressions, not research quality measurements."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from asgcn_unet.batching import PackedSampleBatch
from asgcn_unet.engine import validate
from tests.test_batching import _model
from tests.test_evaluation_batches import DiagnosticDataset


@pytest.fixture(autouse=True, scope="module")
def _cpu_fixture_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _assert_same_summary(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_summary(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_same_summary(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=3e-5, abs=3e-6)
    else:
        assert actual == expected


@pytest.mark.parametrize("max_samples", [None, 6])
@pytest.mark.parametrize("score_positions", [None, {1, 3, 4, 6}])
def test_packed_validation_matches_explicit_b1_reference_with_context_and_tail(
    monkeypatch, max_samples, score_positions,
):
    dataset = DiagnosticDataset(lengths=(4, 2, 1))
    loader = DataLoader(dataset, batch_size=1, collate_fn=list)
    reference = _model(recurrent=True).eval()
    model = copy.deepcopy(reference)
    expected = validate(
        reference, loader, torch.device("cpu"), max_samples=max_samples,
        score_positions=score_positions,
    )
    visited = []
    incoming = []
    batch_sizes = []
    original = model.forward_batch

    def observe(samples, recurrent_states, **kwargs):
        assert isinstance(samples, PackedSampleBatch)
        assert samples.targets.shape[0] == len(samples)
        batch_sizes.append(len(samples))
        lanes = []
        for sample, state in zip(samples, recurrent_states, strict=True):
            visited.append(sample["sample_id"])
            incoming.append((sample["sample_id"], state is not None, sample["events"].shape[0]))
            lanes.append(sample["metadata"]["sequence_id"])
        assert len(set(lanes)) == len(lanes)
        return original(samples, recurrent_states, **kwargs)

    def forbidden_single_sample(*args, **kwargs):
        pytest.fail("Production packed validation must not call forward_sample, including a tail")

    monkeypatch.setattr(model, "forward_batch", observe)
    monkeypatch.setattr(model, "forward_sample", forbidden_single_sample)
    pauses = []
    actual = validate(
        model, loader, torch.device("cpu"), max_samples=max_samples,
        score_positions=score_positions, check_pause=lambda: pauses.append(True),
        batching_section={"batch_size": 2, "num_workers": 0},
    )
    execution = actual.pop("execution")
    assert execution is not None
    _assert_same_summary(actual, expected)
    count = len(dataset) if max_samples is None else max_samples
    expected_ids = [dataset[index]["sample_id"] for index in range(count)]
    assert sorted(visited) == sorted(expected_ids)
    assert len(visited) == len(set(visited)) == count
    assert 2 in batch_sizes and 1 in batch_sizes
    assert len(pauses) >= len(batch_sizes)
    for stream in range(3):
        actual_order = [sample_id for sample_id in visited if sample_id.startswith(f"{stream}/")]
        expected_order = [sample_id for sample_id in expected_ids if sample_id.startswith(f"{stream}/")]
        assert actual_order == expected_order
    for sample_id, has_context, events in incoming:
        frame_index = int(sample_id.split("/")[1])
        assert has_context is (frame_index > 0)
        if frame_index == 1:
            assert events == 0
