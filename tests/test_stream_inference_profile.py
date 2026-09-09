"""CPU synthetic bootstrap tests; none of these values are experiment results."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import Subset

from asgcn_unet.batching import pack_samples, sequence_key
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.stream_inference_profile import make_stream_profile_callback
from tests.test_stream_preflight import SyntheticStreams, _config


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(recurrent=True):
    values = _config()["model"]
    values["recurrent"] = recurrent
    torch.manual_seed(20260910)
    return ASGCNUNet(**values).eval()


def _runner(model, calls):
    def run_forward(packed, contexts, timing=None):
        assert timing is None
        calls.append({
            "ids": [(sequence_key(sample), sample["metadata"]["sequence_index"]) for sample in packed],
            "contexts": [None if value[0] is None else (value[0].sequence_identity, value[0].sequence_index)
                         for value in contexts],
            "shape": packed.sensor_size,
        })
        return model.forward_batch(packed, [value[0] for value in contexts], inference_mode="ann")
    return run_forward


def _reference(model, dataset, target):
    key = sequence_key(dataset[target])
    previous = None
    with torch.inference_mode():
        for index in range(target + 1):
            sample = dataset[index]
            if sequence_key(sample) != key:
                continue
            prediction, diagnostics = model.forward_batch(pack_samples([sample]), [previous], inference_mode="ann")
            previous = diagnostics[0]["recurrent_state"]
    return prediction, previous


def test_nonconsecutive_targets_bootstrap_every_predecessor_in_packed_waves():
    dataset = SyntheticStreams(frames=4)
    model = _model()
    calls = []
    callback = make_stream_profile_callback(dataset, "cpu", _runner(model, calls))
    prediction, diagnostics = callback(pack_samples([dataset[3], dataset[7]]))
    assert len(calls) == 4
    assert all(len(call["ids"]) == 2 for call in calls)
    assert [call["ids"][0][1] for call in calls] == [0, 1, 2, 3]
    assert callback.last_report["full_prefix_frames"] == 6
    assert callback.last_report["prefix_batches"] == 3
    assert callback.last_report["peak_live_states"] == 2
    assert callback.report["bootstrap_included_in_timing"] is True
    assert callback.report["steady_state_throughput"] is False
    assert prediction.shape == (2, 1, 16, 16)
    assert all(detail["recurrent_state"].sequence_index == 3 for detail in diagnostics)
    for call in calls[1:]:
        for (identity, frame), context in zip(call["ids"], call["contexts"], strict=True):
            assert context == (identity, frame - 1)


def test_unequal_prefix_lengths_keep_each_sequence_state_separate_and_match_reference():
    dataset = SyntheticStreams(frames=4)
    model = _model()
    calls = []
    callback = make_stream_profile_callback(dataset, "cpu", _runner(model, calls))
    prediction, _ = callback(pack_samples([dataset[1], dataset[7]]))
    assert [len(call["ids"]) for call in calls] == [2, 1, 1, 2]
    assert callback.last_report["full_prefix_frames"] == 4
    assert calls[-1]["contexts"][0][1] == 0
    assert calls[-1]["contexts"][1][1] == 2
    first, _ = _reference(model, dataset, 1)
    second, _ = _reference(model, dataset, 7)
    torch.testing.assert_close(prediction, torch.cat((first, second)), rtol=1e-5, atol=1e-6)


def test_each_callback_starts_fresh_and_report_contains_no_tensor_state():
    dataset = SyntheticStreams()
    calls = []
    callback = make_stream_profile_callback(dataset, torch.device("cpu"), _runner(_model(), calls))
    targets = pack_samples([dataset[2], dataset[5]])
    first, _ = callback(targets)
    split = len(calls)
    second, _ = callback(targets)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert calls[0]["contexts"] == calls[split]["contexts"] == [None, None]
    assert callback.report["calls"] == 2
    assert callback.report["full_prefix_frames"] == 8
    assert callback.report["state_reused_between_calls"] is False
    saved = callback.last_report
    saved["target_indices"][0] = -100
    assert callback.last_report["target_indices"] == [2, 5]
    def check_no_tensor(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for item in value.values():
                check_no_tensor(item)
        elif isinstance(value, list):
            for item in value:
                check_no_tensor(item)
    check_no_tensor(callback.report)


@pytest.mark.parametrize("indices", [[0, 2], [1], [0, 1, 3, 5]])
def test_subset_missing_any_required_prefix_fails_before_model_calls(indices):
    dataset = Subset(SyntheticStreams(), indices)
    with pytest.raises(ValueError, match="missing required full-sequence prefix"):
        make_stream_profile_callback(dataset, "cpu", lambda *args, **kwargs: pytest.fail("no model call"))


def test_nested_complete_subset_maps_targets_to_original_prefix():
    base = SyntheticStreams(frames=4)
    dataset = Subset(Subset(base, [0, 1, 2, 4, 5, 6]), [0, 1, 3, 4])
    calls = []
    callback = make_stream_profile_callback(dataset, "cpu", _runner(_model(), calls))
    callback(pack_samples([dataset[1], dataset[3]]))
    assert callback.last_report["target_indices"] == [1, 3]
    assert callback.last_report["full_prefix_frames"] == 2
    assert [call["ids"][0][1] for call in calls] == [0, 1]


@pytest.mark.parametrize("indices", [[0, 0], [-1], [6], [True]])
def test_invalid_subset_index_mapping_is_rejected(indices):
    with pytest.raises(ValueError, match="indices"):
        make_stream_profile_callback(Subset(SyntheticStreams(), indices), "cpu", lambda *args: None)


def test_target_batch_rejects_dependent_or_unknown_sequence_frames():
    dataset = SyntheticStreams()
    calls = []
    callback = make_stream_profile_callback(dataset, "cpu", _runner(_model(), calls))
    with pytest.raises(ValueError, match="duplicate/dependent"):
        callback(pack_samples([dataset[0], dataset[1]]))
    unknown = dataset[0]
    unknown["metadata"]["sequence_id"] = "outside-dataset"
    with pytest.raises(ValueError, match="not in"):
        callback(pack_samples([unknown]))
    assert calls == []


def test_decoder_recurrence_disabled_still_preserves_stream_graph_state():
    dataset = SyntheticStreams()
    calls = []
    callback = make_stream_profile_callback(dataset, "cpu", _runner(_model(recurrent=False), calls))
    _, diagnostics = callback(pack_samples([dataset[2], dataset[5]]))
    assert all(call["contexts"][0] is not None for call in calls[1:])
    assert all(detail["recurrent_state"].decoder is None for detail in diagnostics)
    assert all(len(detail["recurrent_state"].graph.timestamps) > 0 for detail in diagnostics)


def test_shape_changes_are_explicit_state_resets_without_mixing_shapes():
    dataset = SyntheticStreams(frames=3)
    dataset.samples[0]["sensor_size"] = (12, 12)
    original_getitem = dataset.__class__.__getitem__
    class Shapes(SyntheticStreams):
        def __getitem__(self, index):
            sample = original_getitem(self, index)
            size = tuple(self.samples[index]["sensor_size"])
            sample["sensor_size"] = size
            sample["target"] = torch.full((1, *size), 0.25)
            return sample
    shaped = Shapes()
    shaped.samples = copy.deepcopy(dataset.samples)
    calls = []
    callback = make_stream_profile_callback(shaped, "cpu", _runner(_model(), calls))
    callback(pack_samples([shaped[2], shaped[5]]))
    assert callback.last_report["shape_change_state_resets"] == 1
    lane = (("synthetic-0", ""), 1)
    next_call = next(call for call in calls if lane in call["ids"])
    assert next_call["contexts"][next_call["ids"].index(lane)] is None
    assert {call["shape"] for call in calls} == {(12, 12), (16, 16)}


def test_original_source_frame_holes_are_not_treated_as_complete_prefix():
    dataset = SyntheticStreams()
    dataset.samples[1]["sequence_index"] = 3
    with pytest.raises(ValueError, match="incomplete or unordered"):
        make_stream_profile_callback(dataset, "cpu", lambda *args: None)


def test_callback_does_not_mutate_target_samples_or_model_weights():
    dataset = SyntheticStreams()
    model = _model()
    callback = make_stream_profile_callback(dataset, "cpu", _runner(model, []))
    batch = pack_samples([dataset[2], dataset[5]])
    events, targets = batch.events.clone(), batch.targets.clone()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    callback(batch)
    torch.testing.assert_close(batch.events, events)
    torch.testing.assert_close(batch.targets, targets)
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
