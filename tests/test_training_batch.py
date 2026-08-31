"""Causal minibatch training regressions; generated HDF5 lives only in pytest tempdirs."""

from __future__ import annotations

import copy
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from asgcn_unet import engine
from asgcn_unet.batching import sequence_key
from asgcn_unet.data import build_dataset
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.training import TrainingState, batching_contract, forward_training_loss
from tests.fixtures import make_eventhdr


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _config(tmp_path: Path, *, batch_size: int = 2, timing_steps: int = 0) -> dict:
    train_root = tmp_path / "train"
    train_root.mkdir(parents=True)
    files = []
    for index, frames in enumerate((3, 2, 2)):
        source = make_eventhdr(tmp_path / "fixtures" / str(index), frames=frames)
        name = f"{index + 1}.h5"
        shutil.copyfile(source, train_root / name)
        files.append(name)
    val_root = tmp_path / "validation"
    make_eventhdr(val_root, frames=2)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "status": "final",
        "split_schema": "official_separate_roots_v1",
        "group_semantics": "h5_sequence_file_not_physical_scene",
        "train_files": files,
        "val_files": ["test.h5"],
    }), encoding="utf-8")
    return {
        "seed": 17,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr", "root": str(train_root), "val_root": str(val_root),
            "split_manifest": str(split), "target_channels": 1,
            "max_events": 16, "crop_size": [16, 16], "tone_map": "log",
        },
        "model": {
            "architecture_version": 2, "graph_operator": "spline", "spline_backend": "torch",
            "spline_pseudo": "distance_over_radius", "spline_is_open": True,
            "hidden_dim": 4, "graph_layers": 1, "event_sampling_factor": 1,
            "graph_radius": 2.0, "graph_position_dims": 3, "graph_chunk_size": 16,
            "spline_kernel_size": 3, "spline_degree": 1, "spline_root_weight": True,
            "raster_downsample": 4, "decoder_channels": 4, "output_channels": 1,
            "recurrent": True,
        },
        "train": {
            "epochs": 1, "batch_size": batch_size, "num_workers": 0, "amp": False,
            "batching": "independent_sequences" if batch_size > 1 else "single_frame",
            "optimizer": "adam_gc", "learning_rate": 0.0002, "grad_clip": 1.0,
            "lr_milestones": [1], "lr_gamma": 0.5, "validate_every": 1,
            "max_train_samples": None, "max_val_samples": None, "log_every": 100,
            "loss_weights": {"charbonnier": 1.0, "ssim": 0.2, "gradient": 0.1, "temporal": 0.1},
            "timing_steps": timing_steps, "timing_warmup": 0,
        },
        "output": {"run_dir": str(tmp_path / "run")},
    }


def _checkpoint(run_dir: Path) -> dict:
    return torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("timing_steps", [0, 2])
def test_real_hdf5_batch_training_covers_every_frame_and_keeps_partial_tail(
    tmp_path, monkeypatch, timing_steps
) -> None:
    config = _config(tmp_path, timing_steps=timing_steps)
    dataset = build_dataset(config["dataset"], split="train")
    expected_ids = [dataset[index]["sample_id"] for index in range(len(dataset))]
    batch_calls = []
    single_calls = []
    previous_state = {}
    original_batch = ASGCNUNet.forward_training_batch
    original_single = ASGCNUNet.forward_sample
    original_release = TrainingState.release_finished
    retained_sequence_counts = []

    def tracked_batch(model, samples, recurrent_states=None, *, timing=None):
        assert model.training
        assert len({sequence_key(sample) for sample in samples}) == len(samples)
        for sample, state in zip(samples, recurrent_states, strict=True):
            key = sequence_key(sample)
            if key in previous_state:
                torch.testing.assert_close(state, previous_state[key], rtol=0, atol=0)
                assert not state.requires_grad
            else:
                assert state is None
        # Calling forward_sample in this actual training batch would fail below.
        with (
            patch.object(model.encoder, "forward_ann", wraps=model.encoder.forward_ann) as encoder,
            patch.object(model.decoder, "forward", wraps=model.decoder.forward) as decoder,
        ):
            result = original_batch(model, samples, recurrent_states, timing=timing)
        assert encoder.call_count == decoder.call_count == 1
        assert decoder.call_args.args[0].shape[0] == len(samples)
        for sample, detail in zip(samples, result[1], strict=True):
            previous_state[sequence_key(sample)] = detail["recurrent_state"].detach().clone()
        batch_calls.append([(sample["sample_id"], sequence_key(sample), sample["metadata"]["sequence_index"]) for sample in samples])
        return result

    def tracked_single(model, sample, *args, **kwargs):
        assert not model.training, "batch training must not loop over forward_sample"
        single_calls.append((sample["metadata"]["sequence_index"], kwargs.get("recurrent_state")))
        return original_single(model, sample, *args, **kwargs)

    def tracked_release(state, samples, final_sequence_indices):
        original_release(state, samples, final_sequence_indices)
        retained_sequence_counts.append(len(state.values))
        assert len(state.values) <= 2

    monkeypatch.setattr(ASGCNUNet, "forward_training_batch", tracked_batch)
    monkeypatch.setattr(ASGCNUNet, "forward_sample", tracked_single)
    monkeypatch.setattr(TrainingState, "release_finished", tracked_release)
    engine.train(config)
    run_dir = Path(config["output"]["run_dir"])
    checkpoint = _checkpoint(run_dir)
    assert [len(batch) for batch in batch_calls] == [2, 2, 2, 1]
    assert retained_sequence_counts == [2, 1, 1, 0]
    assert Counter(item[0] for batch in batch_calls for item in batch) == Counter(expected_ids)
    sequence_order = defaultdict(list)
    for batch in batch_calls:
        for _, key, index in batch:
            sequence_order[key].append(index)
    assert sorted(sequence_order.values()) == [[0, 1], [0, 1], [0, 1, 2]]
    assert [item[0] for item in single_calls] == [0, 1]
    assert single_calls[0][1] is None and single_calls[1][1] is not None
    assert checkpoint["training_protocol"]["version"] == 6
    assert checkpoint["training_protocol"]["batching"] == batching_contract(2)
    performance = checkpoint["history"][0]["performance"]
    assert performance["frames"] == 7
    assert performance["optimizer_steps"] == 4
    assert performance["batch_size_limit"] == 2
    assert performance["includes_validation"] is False
    assert performance["frames_per_second"] > 0
    assert checkpoint["history"][0]["val"]["frames"] == 2
    assert {int(state["step"]) for state in checkpoint["optimizer"]["state"].values()} == {4}
    if timing_steps:
        report = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
        assert report["cpu_diagnostic_only"] is True
        assert report["cuda_events_measured"] is False
        assert report["window_complete"] is True
        assert report["measured_steps"] == 2
        for stage in ("dataload", "transfer", "graph", "encoder", "decoder", "loss", "backward", "optimizer"):
            assert report["stages"][stage]["host_wall"]["count"] == 2
        assert report["stages"]["gradient_check"]["host_wall"]["count"] == 4
    else:
        assert not (run_dir / "timing.json").exists()


def _sample(identity, index, *, source="first.h5", size=(2, 2), target=0.0):
    return {
        "sample_id": f"{identity}/{source}/{index}",
        "events": torch.empty(0, 4), "sensor_size": size,
        "target": torch.full((1, *size), target),
        "metadata": {"sequence_id": identity, "source_file": source, "sequence_index": index},
    }


def _commit_markers(store, samples, markers):
    prediction = torch.stack([torch.full_like(sample["target"], marker) for sample, marker in zip(samples, markers, strict=True)]).requires_grad_()
    target = torch.stack([sample["target"] for sample in samples])
    diagnostics = [{"recurrent_state": torch.full((1, 2, 1, 1), marker, requires_grad=True)} for marker in markers]
    store.commit(samples, prediction, diagnostics, target)


def test_sequence_context_follows_identity_not_batch_position_and_is_detached() -> None:
    state = TrainingState(independent_sequences=True)
    initial = [_sample("shared", 0, source="a.h5"), _sample("shared", 0, source="b.h5")]
    assert state.prepare(initial) == [(None, None, None), (None, None, None)]
    _commit_markers(state, initial, [11.0, 22.0])
    # Same scene label, different source files; swap lanes on the next call.
    following = [_sample("shared", 1, source="b.h5"), _sample("shared", 1, source="a.h5")]
    contexts = state.prepare(following)
    for context, marker in zip(contexts, (22.0, 11.0), strict=True):
        assert all(not value.requires_grad for value in context)
        assert context[0].unique().item() == context[1].unique().item() == marker
    # prepare is read-only: failed/uncommitted attempts cannot advance an index.
    repeated = state.prepare(following)
    for actual, expected in zip(repeated, contexts, strict=True):
        assert all(a is b for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("batch_size", [2, 8])
def test_committed_sequence_slices_do_not_keep_full_batch_storages_alive(batch_size) -> None:
    state = TrainingState(independent_sequences=True)
    samples = [_sample(str(index), 0) for index in range(batch_size)]
    prediction = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 1, 2, 2).requires_grad_()
    target = prediction.detach() + 0.25
    hidden = torch.arange(batch_size * 2, dtype=torch.float32).reshape(batch_size, 2, 1, 1).requires_grad_()
    diagnostics = [{"recurrent_state": hidden[index:index + 1]} for index in range(batch_size)]
    state.commit(samples, prediction, diagnostics, target)
    original_storages = {
        value.untyped_storage().data_ptr() for value in (prediction, target, hidden)
    }
    committed_storages = set()
    for index, sample in enumerate(samples):
        context = state.values[sequence_key(sample)][2:]
        for actual, expected in zip(context, (hidden[index:index + 1], prediction[index:index + 1], target[index:index + 1]), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not actual.requires_grad
            pointer = actual.untyped_storage().data_ptr()
            assert pointer not in original_storages
            assert pointer not in committed_storages
            committed_storages.add(pointer)
            assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


@pytest.mark.parametrize("index,size", [(0, (2, 2)), (2, (2, 2)), (None, (2, 2)), (1, (3, 2))])
def test_sequence_context_resets_on_rewind_gap_missing_index_or_shape_change(index, size) -> None:
    state = TrainingState(independent_sequences=True)
    _commit_markers(state, [_sample("a", 0)], [3.0])
    assert state.prepare([_sample("a", index, size=size)]) == [(None, None, None)]


def test_duplicate_streams_are_rejected_before_context_mutation() -> None:
    state = TrainingState(independent_sequences=True)
    with pytest.raises(ValueError, match="two frames of one sequence"):
        state.prepare([_sample("a", 0), _sample("a", 1)])
    assert state.values == {}


def test_only_completed_streams_release_state_and_new_lane_starts_without_context() -> None:
    store = TrainingState(independent_sequences=True)
    samples = [_sample("a", 2), _sample("b", 2)]
    _commit_markers(store, samples, [1.0, 2.0])
    ongoing = store.values[sequence_key(samples[1])]
    store.release_finished(samples, {sequence_key(samples[0]): 2, sequence_key(samples[1]): 4})
    assert set(store.values) == {sequence_key(samples[1])}
    assert store.values[sequence_key(samples[1])] is ongoing
    contexts = store.prepare([_sample("c", 0), _sample("b", 3)])
    assert contexts[0] == (None, None, None)
    assert all(a is b for a, b in zip(contexts[1], ongoing[2:], strict=True))


def test_baseline_context_matches_previous_single_trajectory_reset_formula() -> None:
    store = TrainingState(independent_sequences=False)
    previous = (None, None, None)
    context = (None, None, None)
    trajectory = [_sample("a", 0), _sample("a", 1), _sample("b", 0), _sample("a", 2), _sample("a", 4), _sample("a", 5)]
    for position, sample in enumerate(trajectory):
        identity, index, size = engine._sample_sequence_info(sample)
        expected = context if engine._continues_sequence(identity, index, size, *previous) else (None, None, None)
        actual = store.prepare([sample])[0]
        assert all(a is b for a, b in zip(actual, expected, strict=True))
        _commit_markers(store, [sample], [float(position)])
        context = tuple(store.values[store.last_key][2:])
        previous = (identity, index, size)
        assert len(store.values) == 1


class _FixedPrediction(nn.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = nn.Parameter(prediction)
        self.calls = 0

    def forward_training_batch(self, samples, recurrent_states, *, timing=None):
        self.calls += 1
        return self.prediction, [{"recurrent_state": value} for value in recurrent_states]

    def forward_sample(self, sample, *, recurrent_state=None):
        self.calls += 1
        return self.prediction, {"recurrent_state": recurrent_state}


@pytest.mark.parametrize("batch_mode", [False, True])
def test_forward_loss_matches_explicit_per_frame_temporal_mean_and_gradients(batch_mode) -> None:
    samples = [_sample("a", 1, size=(16, 16), target=0.5)]
    contexts = [(None, torch.full((1, 1, 16, 16), 0.4), torch.full((1, 1, 16, 16), 0.3))]
    prediction = torch.full((1, 1, 16, 16), 0.8)
    if batch_mode:
        samples.append(_sample("b", 0, size=(16, 16), target=0.1))
        contexts.append((None, None, None))
        prediction = torch.cat([prediction, torch.full_like(prediction, 0.2)])
    model = _FixedPrediction(prediction)
    reference = prediction.detach().clone().requires_grad_()
    criterion = ReconstructionLoss()
    loss, parts, payload = forward_training_loss(
        model, criterion, samples, contexts, batch_mode=batch_mode,
        amp_enabled=False, temporal_weight=0.7,
    )
    target = torch.stack([sample["target"] for sample in samples])
    expected, _ = criterion(reference, target)
    temporal_terms = [
        F.l1_loss(reference[i:i + 1] - context[1], target[i:i + 1] - context[2])
        if context[1] is not None else reference.new_zeros(())
        for i, context in enumerate(contexts)
    ]
    temporal = torch.stack(temporal_terms).mean()
    expected = expected + 0.7 * temporal
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(parts["temporal"], temporal, rtol=0, atol=0)
    torch.testing.assert_close(payload[2], target, rtol=0, atol=0)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.prediction.grad, reference.grad, rtol=0, atol=0)
    assert model.calls == 1


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("all_context", [False, True])
def test_temporal_full_context_avoids_index_gathers_with_equal_loss_and_gradient(
    batch_size, all_context
) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    class IndexCounter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.index.Tensor:
                self.calls += 1
            return func(*args, **(kwargs or {}))

    torch.manual_seed(7)
    samples = [_sample(str(i), 1, size=(16, 16), target=i / 16) for i in range(batch_size)]
    contexts = [
        (None, torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16))
        if all_context or i % 2 else (None, None, None)
        for i in range(batch_size)
    ]
    prediction = torch.rand(batch_size, 1, 16, 16)
    model = _FixedPrediction(prediction)
    reference = prediction.detach().clone().requires_grad_()
    criterion = ReconstructionLoss()
    counter = IndexCounter()
    with counter:
        loss, parts, _ = forward_training_loss(
            model, criterion, samples, contexts, batch_mode=batch_size > 1,
            amp_enabled=False, temporal_weight=0.7,
        )
    valid = [i for i, context in enumerate(contexts) if context[1] is not None]
    assert counter.calls == (2 if valid and not all_context else 0)
    target = torch.stack([sample["target"] for sample in samples])
    expected, _ = criterion(reference, target)
    if valid:
        temporal = F.l1_loss(
            reference[valid] - torch.cat([contexts[i][1] for i in valid]),
            target[valid] - torch.cat([contexts[i][2] for i in valid]),
        ) * (len(valid) / len(samples))
        expected = expected + 0.7 * temporal
        torch.testing.assert_close(parts["temporal"], temporal, rtol=0, atol=0)
    else:
        assert "temporal" not in parts
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(model.prediction.grad, reference.grad, rtol=0, atol=0)


def test_whole_batch_amp_retry_commits_once_without_advancing_other_streams() -> None:
    class BatchLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.25))
            self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))

        def forward_training_batch(self, samples, recurrent_states, *, timing=None):
            self.forward_count.add_(1)
            prediction = self.weight.half().float().expand(len(samples), 1, 2, 2)
            states = [state + torch.rand_like(state) for state in recurrent_states]
            return prediction, [{"recurrent_state": state} for state in states]

    model = BatchLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cpu", init_scale=65536.0)
    store = TrainingState(independent_sequences=True)
    _commit_markers(store, [_sample("a", 0), _sample("b", 0)], [1.0, 2.0])
    samples = [_sample("a", 1), _sample("b", 1)]
    contexts = store.prepare(samples)
    original_state = copy.deepcopy(store.values)
    attempts = []

    def criterion(prediction, target):
        loss = prediction.mean()
        return loss, {"reconstruction": loss.detach()}

    def closure():
        torch.testing.assert_close(store.values, original_state, rtol=0, atol=0)
        result = forward_training_loss(
            model, criterion, samples, contexts, batch_mode=True,
            amp_enabled=False, temporal_weight=0.0,
        )
        attempts.append((result[2][1], random.random(), np.random.random()))
        return result

    with patch.object(optimizer, "step", wraps=optimizer.step) as update:
        payload, _, _, info = engine._training_step(
            model, optimizer, scaler, closure, optimizer_mode="adamw", max_norm=1.0,
            epoch=1, step=0, sample_id=[sample["sample_id"] for sample in samples],
        )
    assert info["retries"] == 1
    assert update.call_count == 1
    assert model.forward_count.item() == 1
    assert len(attempts) == 2
    torch.testing.assert_close(attempts[0], attempts[1], rtol=0, atol=0)
    store.commit(samples, *payload)
    assert [entry[0] for entry in store.values.values()] == [1, 1]
    assert int(optimizer.state[model.weight]["step"]) == 1


def test_batched_epoch_resume_reproduces_uninterrupted_model_optimizer_and_rng(tmp_path) -> None:
    config = _config(tmp_path)
    config["train"]["epochs"] = 2
    config["output"]["run_dir"] = str(tmp_path / "continuous")
    engine.train(config)
    uninterrupted = _checkpoint(tmp_path / "continuous")
    config["train"]["epochs"] = 1
    config["output"]["run_dir"] = str(tmp_path / "resumed")
    engine.train(config)
    first = _checkpoint(tmp_path / "resumed")
    assert first["training_protocol"]["version"] == 6
    config["train"]["epochs"] = 2
    engine.train(config, resume_from=tmp_path / "resumed" / "last.pt")
    resumed = _checkpoint(tmp_path / "resumed")
    for field in ("model", "optimizer", "scheduler", "scaler", "val"):
        torch.testing.assert_close(resumed[field], uninterrupted[field], rtol=0, atol=0)
    assert resumed["best_model_state_sha256"] == uninterrupted["best_model_state_sha256"]
    assert resumed["rng_state"]["python"] == uninterrupted["rng_state"]["python"]
    np.testing.assert_array_equal(resumed["rng_state"]["numpy"][1], uninterrupted["rng_state"]["numpy"][1])
    torch.testing.assert_close(resumed["rng_state"]["torch"], uninterrupted["rng_state"]["torch"], rtol=0, atol=0)
    assert [entry["train_loss"] for entry in resumed["history"]] == [entry["train_loss"] for entry in uninterrupted["history"]]
    assert [(entry["performance"]["frames"], entry["performance"]["optimizer_steps"]) for entry in resumed["history"]] == [(7, 4), (7, 4)]
    assert {int(state["step"]) for state in resumed["optimizer"]["state"].values()} == {8}


@pytest.mark.parametrize("changed", ["batch_one", "batch_three", "source", "contract"])
def test_batch_resume_rejects_baseline_different_batch_source_or_contract(changed) -> None:
    config = {"seed": 17, "train": {"batch_size": 2, "batching": "independent_sequences", "amp": False}}
    protocol = engine._training_protocol(config, torch.device("cpu"))
    assert protocol["version"] == 6
    assert engine._valid_training_protocol_contract(protocol)
    altered = copy.deepcopy(protocol)
    if changed in {"batch_one", "batch_three"}:
        config["train"]["batch_size"] = 1 if changed == "batch_one" else 3
        config["train"]["batching"] = "single_frame" if changed == "batch_one" else "independent_sequences"
        altered = engine._training_protocol(config, torch.device("cpu"))
        assert altered["version"] == (5 if changed == "batch_one" else 6)
    elif changed == "source":
        altered["source"]["source_tree_sha256"] = "0" * 64
    else:
        altered["batching"]["loss"] = "mean_only_context_frames"
        assert not engine._valid_training_protocol_contract(altered)
    with pytest.raises(ValueError, match="training protocol differs"):
        engine._validate_training_protocol({"training_protocol": protocol}, altered)
