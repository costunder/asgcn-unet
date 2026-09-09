"""Tiny CPU-only stream snapshot/retry tests, not model quality or performance."""

from __future__ import annotations

import copy
import io
from dataclasses import replace

import pytest
import torch
from torch import nn

from asgcn_unet import engine
from asgcn_unet.batching import pack_samples
from asgcn_unet.checkpoint import capture_training_state, restore_training_state
from asgcn_unet.evaluation_batches import evaluation_frames
from asgcn_unet.graph import EventGraph
from asgcn_unet.stream_graph import StreamGraph
from asgcn_unet.stream_state import StreamingReconstructionState, restore_stream_training_state
from asgcn_unet.training import TrainingState, forward_training_loss


@pytest.fixture(autouse=True, scope="module")
def _cpu_synthetic_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _stream(*, index=3):
    features = torch.tensor([[0.1, 0.2, 0.1, 1], [0.2, 0.2, 0.2, -1]], requires_grad=True)
    positions = torch.tensor([[0.1, 0.2, 0.1, 1], [0.2, 0.2, 0.2, 0]], dtype=torch.float64)
    graph = EventGraph(features, positions, torch.tensor([[0, 1], [1, 0]]),
                       torch.full((2, 1), 0.5, dtype=torch.float64))
    return StreamingReconstructionState(
        StreamGraph(graph, torch.zeros(2, dtype=torch.long),
                    torch.tensor([0.1, 0.2], dtype=torch.float64)),
        None, torch.full((1, 4, 2, 2), 0.1, requires_grad=True),
        0.0, 0.3, index, ("sequence-a", "synthetic.h5"), (0, 1), "a" * 64,
    )


def _store(*, independent=True):
    store = TrainingState(independent_sequences=independent)
    key = ("sequence-a", "synthetic.h5" if independent else "")
    store.values[key] = (3, (16, 16), _stream(), torch.full((1, 1, 16, 16), 0.2),
                         torch.full((1, 1, 16, 16), 0.3))
    store.last_key = key
    return store


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensor_leaves(child)


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right)
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for one, two in zip(left, right, strict=True):
            _assert_exact(one, two)
    else:
        assert type(left) is type(right) and left == right


@pytest.mark.parametrize("independent", [True, False])
def test_stream_context_round_trip_is_safe_serializable_exact_and_storage_independent(independent):
    original = _store(independent=independent)
    payload = capture_training_state(original)
    assert payload["version"] == 2
    output = io.BytesIO()
    torch.save(payload, output)
    output.seek(0)
    loaded = torch.load(output, weights_only=True)
    restored = restore_training_state(loaded, independent_sequences=independent, device="cpu")
    before = original.values[original.last_key][2].training_payload()
    saved = payload["entries"][0]["recurrent"]
    after = restored.values[restored.last_key][2].training_payload()
    _assert_exact(before, saved)
    _assert_exact(before, after)
    for source, snapshot, resumed in zip(_tensor_leaves(before), _tensor_leaves(saved),
                                          _tensor_leaves(after), strict=True):
        assert len({source.data_ptr(), snapshot.data_ptr(), resumed.data_ptr()}) == 3
        assert not snapshot.requires_grad and not resumed.requires_grad
        assert snapshot.device.type == resumed.device.type == "cpu"
    restored.values[restored.last_key][2].graph.graph.positions.fill_(42)
    assert not torch.equal(after["graph"]["positions"], saved["graph"]["positions"])
    assert torch.equal(saved["graph"]["positions"], before["graph"]["positions"])


def test_stream_context_version_one_and_live_encoder_caches_are_rejected():
    payload = capture_training_state(_store())
    payload["version"] = 1
    with pytest.raises(ValueError, match="version 2"):
        restore_training_state(payload, independent_sequences=True, device="cpu")
    store = _store()
    old = store.values[store.last_key]
    store.values[store.last_key] = (*old[:2], replace(old[2], encoder=object()), *old[3:])
    with pytest.raises(ValueError, match="raw-graph ANN"):
        capture_training_state(store)


@pytest.mark.parametrize("field,value", [
    ("sequence_index", -1), ("sequence_index", True), ("sequence_index", 4),
    ("sequence_identity", ("another", "synthetic.h5")),
    ("last_event_id", (0, -1)), ("last_event_id", None),
    ("origin_seconds", 0.4), ("watermark_seconds", 0.15),
    ("watermark_seconds", float("nan")), ("contract", "g" * 64),
    ("schema", "unknown"), ("timestamps", torch.tensor([0.2, 0.1], dtype=torch.float64)),
    ("node_batch", torch.tensor([0, 1])), ("decoder", torch.zeros(1, 0, 2, 2)),
    ("decoder", torch.full((1, 4, 2, 2), float("nan"))),
])
def test_corrupt_stream_context_rejected_before_transfer(field, value, monkeypatch):
    payload = capture_training_state(_store())
    payload["entries"][0]["recurrent"][field] = value
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid stream context must fail before device transfer")
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises((ValueError, TypeError)):
        restore_training_state(payload, independent_sequences=True, device="cpu")


@pytest.mark.parametrize("field,value", [
    ("node_features", None), ("node_features", torch.tensor(0.0)),
    ("node_features", torch.zeros(2, 3)),
    ("node_features", torch.full((2, 4), float("nan"))),
    ("positions", torch.zeros(2, 4)),
    ("edge_index", torch.tensor([0, 1])),
    ("edge_index", torch.tensor([[0, 1], [1, 2**60]])),
    ("edge_index", torch.tensor([[0, 1], [0, 0]])),
    ("edge_index", torch.tensor([[0, 1], [-1, 0]])),
    ("in_degree", torch.tensor([2, 1])),
    ("in_degree", torch.ones(2)),
    ("edge_attr", torch.full((2, 1), 1.0, dtype=torch.float64)),
])
def test_corrupt_stream_graph_rejected_before_eventgraph_allocation(field, value):
    payload = capture_training_state(_store())["entries"][0]["recurrent"]
    payload["graph"][field] = value
    with pytest.raises((ValueError, TypeError)):
        restore_stream_training_state(payload)


def test_version_two_still_checks_prediction_target_shapes_and_finiteness():
    for field, value in [("prediction", None), ("target", torch.zeros(1, 1, 15, 16)),
                          ("target", torch.full((1, 1, 16, 16), float("inf")))]:
        payload = capture_training_state(_store())
        payload["entries"][0][field] = value
        with pytest.raises(ValueError):
            restore_training_state(payload, independent_sequences=True, device="cpu")


def _sample(index=4):
    return {"events": torch.empty((0, 4), dtype=torch.float64),
            "event_ids": torch.empty((0, 2), dtype=torch.long),
            "target": torch.full((1, 16, 16), 0.3), "sensor_size": (16, 16),
            "sample_id": f"sequence-a/{index}",
            "metadata": {"sequence_id": "sequence-a", "source_file": "synthetic.h5",
                         "sequence_index": index}}


def test_amp_retry_reuses_unchanged_incoming_stream_and_commits_one_optimizer_step():
    class SyntheticLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))
            self.register_buffer("calls", torch.zeros((), dtype=torch.long))

        def forward_training_batch(self, samples, states, *, timing=None):
            self.calls.add_(1)
            prediction = self.weight.half().float().expand(len(samples), 1, 16, 16)
            next_state = states[0].clone()
            next_state.decoder = next_state.decoder + torch.rand_like(next_state.decoder)
            next_state.sequence_index += 1
            return prediction, [{"recurrent_state": next_state}]

    store = _store()
    saved = capture_training_state(store)
    samples = [_sample()]
    contexts = store.prepare(samples)
    model = SyntheticLossModel()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
    attempts = []
    def criterion(prediction, target):
        loss = prediction.mean()
        return loss, {"reconstruction": loss.detach()}
    def closure():
        _assert_exact(capture_training_state(store), saved)
        result = forward_training_loss(model, criterion, samples, contexts, batch_mode=True,
                                       amp_enabled=False, temporal_weight=0.0)
        attempts.append(result[2][1][0]["recurrent_state"].training_payload())
        return result
    payload, _, _, info = engine._training_step(
        model, optimizer, scaler, closure, optimizer_mode="adamw", max_norm=1.0,
        epoch=1, step=0, sample_id="synthetic-stream",
    )
    assert info["retries"] == 1
    assert len(attempts) == 2 and model.calls.item() == 1
    _assert_exact(attempts[0], attempts[1])
    _assert_exact(capture_training_state(store), saved)
    store.commit(samples, *payload)
    assert store.values[store.last_key][0] == 4
    assert store.values[store.last_key][2].sequence_index == 4
    assert int(optimizer.state[model.weight]["step"]) == 1


@pytest.mark.parametrize("nonfinite", [False, True])
def test_evaluation_finite_validation_supports_stream_state_without_tensor_cat(nonfinite):
    sample = _sample(index=3)
    bundle = _stream()
    if nonfinite:
        bundle.graph.graph.positions[0, 0] = float("nan")
    def run_forward(samples, contexts, timer):
        detail = {"recurrent_state": bundle, "nodes": 2, "edges": 2,
                  "isolated_nodes": 0, "isolate_ratio": 0.0, "max_degree": 1}
        return samples.targets.clone(), [detail]
    iterator = evaluation_frames([pack_samples([sample])], [[0]], device=torch.device("cpu"),
                                 run_forward=run_forward, independent_sequences=True,
                                 statistics={}, timing_steps=1, timing_warmup=0)
    if nonfinite:
        with pytest.raises(FloatingPointError, match="Nonfinite evaluation"):
            list(iterator)
    else:
        rows = list(iterator)
        assert len(rows) == 1
        assert isinstance(rows[0][5]["recurrent_state"], StreamingReconstructionState)


def test_empty_graph_with_previous_last_id_can_be_restored_after_expiration():
    payload = capture_training_state(_store())["entries"][0]["recurrent"]
    payload["graph"] = {"node_features": torch.empty(0, 4),
                        "positions": torch.empty(0, 4, dtype=torch.float64),
                        "edge_index": torch.empty(2, 0, dtype=torch.long),
                        "edge_attr": torch.empty(0, 1, dtype=torch.float64),
                        "in_degree": torch.empty(0, dtype=torch.long)}
    payload["timestamps"] = torch.empty(0, dtype=torch.float64)
    payload["node_batch"] = torch.empty(0, dtype=torch.long)
    restored = restore_stream_training_state(copy.deepcopy(payload))
    assert len(restored.graph.timestamps) == 0 and restored.last_event_id == (0, 1)
