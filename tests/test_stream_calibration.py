"""Actual H5-shaped CPU smoke data only, not production calibration results."""

import copy

import pytest
import torch

from asgcn_unet import engine
from asgcn_unet.batching import sequence_key
from asgcn_unet.graph import EventGraph
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
from tests.fixtures import make_eventhdr
from tests.test_p0_engine import _eval_config, _model_config


@pytest.fixture(autouse=True)
def cpu_synthetic_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _fixture(tmp_path):
    root = tmp_path / "physical-hdr"
    make_eventhdr(root / "scene_a")
    make_eventhdr(root / "scene_b")
    model_config = _model_config()
    # Reduced dimensions belong only to this explicitly synthetic smoke test.
    model_config.update(architecture_version=3, graph_execution="event_driven", graph_layers=2,
                        hidden_dim=8, graph_radius=0.3, stream_config={
                            "window_seconds": 0.006, "time_scale_seconds": 0.01,
                            "node_time_feature": "physical_frame_offset",
                            "clock": "event_local_pending_off_v1",
                            "arrival_policy": "simultaneous_equal_timestamp",
                        })
    torch.manual_seed(104)
    model = ASGCNUNet(**model_config)
    state = model.state_dict()
    source = tmp_path / "synthetic-stream-ann.pt"
    atomic_torch_save({"checkpoint_type": "training", "epoch": 1, "model": state,
                       "model_state_sha256": engine._model_state_sha256(state),
                       "model_config": model_config, "paper_core_version": engine.PAPER_CORE_VERSION}, source)
    config = _eval_config(root, tmp_path / "unused")
    config["dataset"].update(max_events=None, event_time_contract="physical_seconds_v1",
                             timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=1.0)
    config["model"] = model_config
    config["calibration"] = {"batch_size": 2, "num_workers": 0, "persistent_workers": False}
    return config, source


@torch.no_grad()
def _manual_full_causal_maxima(config, source):
    """Independent full pairwise graph reconstruction, without stream helpers."""
    model, _ = engine.load_model_checkpoint(source, torch.device("cpu"), config["model"])
    model.eval()
    model.fold_batch_norm()
    model.reset_activation_maxima()
    dataset = engine.build_dataset(config["dataset"], split="calibration")
    retained = {}
    total = incoming = live_total = 0
    try:
        for index in range(len(dataset)):
            sample = dataset.get_topology_sample(index)
            key = sequence_key(sample)
            events = sample["events"]
            time = sample["metadata"]["stream_time"]
            height, width = sample["sensor_size"]
            x, y = events[:, 0] / (width - 1), events[:, 1] / (height - 1)
            polarity = torch.where(events[:, 3] > 0, 1., -1.)
            scale = model.stream_config["time_scale_seconds"]
            features = torch.stack((x, y, (events[:, 2] - time["interval_start_seconds"]) / scale, polarity), 1).float()
            positions = torch.stack((x, y, (events[:, 2] - time["sequence_origin_seconds"]) / scale, (polarity + 1) / 2), 1)
            timestamps = events[:, 2]
            if key in retained:
                old_x, old_pos, old_t = retained[key]
                features = torch.cat((old_x, features))
                positions = torch.cat((old_pos, positions))
                timestamps = torch.cat((old_t, timestamps))
            keep = timestamps >= time["interval_end_seconds"] - model.stream_config["window_seconds"]
            features, positions, timestamps = features[keep], positions[keep], timestamps[keep]
            retained[key] = features, positions, timestamps
            distance = torch.linalg.vector_norm(
                (positions[:, None, :3] - positions[None, :, :3]) / model.graph_radius, dim=2,
            )
            connected = (distance < 1) & ~torch.eye(len(features), dtype=torch.bool)
            edges = torch.nonzero(connected, as_tuple=False).T.contiguous()
            graph = EventGraph(features, positions, edges, distance[edges[0], edges[1], None])
            _, activations = model.encoder.forward_ann(graph, return_activations=True)
            model.encoder.update_activation_maxima(activations)
            total += 1
            incoming += len(events)
            live_total += len(features)
    finally:
        dataset.close()
    model.apply_parameter_normalization()
    return model.state_dict(), total, incoming, live_total


def test_stream_calibration_physical_lanes_match_manual_causal_graph_maxima(tmp_path, monkeypatch):
    config, source = _fixture(tmp_path)
    expected, total, incoming, live_total = _manual_full_causal_maxima(config, source)
    calls = []
    original = ASGCNUNet.calibrate_stream_batch

    def tracked(self, packed, states=None):
        assert packed.targets is None
        assert len({sequence_key(sample) for sample in packed}) == len(packed)
        assert not torch.is_autocast_enabled("cpu")
        for sample, state in zip(packed, states, strict=True):
            index = sample["metadata"]["sequence_index"]
            assert (state is None) == (index == 0)
            if state is not None:
                assert state.sequence_index + 1 == index
                assert state.encoder is None and state.decoder is None
        calls.append(len(packed))
        return original(self, packed, states)

    def static_forbidden(*args, **kwargs):
        raise AssertionError("physical streaming calibration cannot use independent static frames")

    monkeypatch.setattr(ASGCNUNet, "calibrate_stream_batch", tracked)
    monkeypatch.setattr(ASGCNUNet, "calibrate_batch", static_forbidden)
    output = tmp_path / "synthetic-stream-snn.pt"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        engine.calibrate(config, source, output, allow_unsealed_calibration=True)
    result = torch.load(output, weights_only=False)
    assert max(calls) == 2 and sum(calls) == total
    assert int(result["model"]["calibration_attempts"]) == total
    assert result["snn_calibration_samples"] == total
    assert result["snn_calibration_summary"]["minimum_valid_samples"] == total
    assert result["paper_core_version"] == engine.PAPER_CORE_VERSION
    assert result["model_config"]["architecture_version"] == 3
    assert result["model_config"]["graph_execution"] == "event_driven"
    report = result["execution_report"]
    assert report["data"]["used_ratio"] == 1
    assert report["data"]["graph_statistics"]["nodes"]["total"] == live_total > incoming
    assert report["batching"]["strategy"] == "chronological_independent_stream_shape_lanes"
    assert report["data"]["causal_context"]["completed_sequence_states_released"] is True
    assert result["calibration_performance"]["frames"] == total
    for name, value in expected.items():
        if name.endswith(("normalization_scale", "calibration_activation_max")):
            torch.testing.assert_close(result["model"][name], value, rtol=1e-5, atol=2e-6)
    engine.load_model_checkpoint(output, torch.device("cpu"), config["model"])


def test_stream_profile_uses_warm_predecessors_and_does_not_contaminate_final_pass(tmp_path):
    config, source = _fixture(tmp_path)
    reference_path = tmp_path / "synthetic-explicit.pt"
    engine.calibrate(config, source, reference_path, allow_unsealed_calibration=True)
    reference = torch.load(reference_path, weights_only=False)
    automatic = copy.deepcopy(config)
    automatic["calibration"].update(batch_size="auto", num_workers="auto", batch_candidates=[1, 2],
                                     worker_candidates=[0], profile_steps=1, profile_warmup=0,
                                     profile_debug_cpu=True)
    output = tmp_path / "synthetic-auto.pt"
    engine.calibrate(automatic, source, output, allow_unsealed_calibration=True)
    result = torch.load(output, weights_only=False)
    profile = result["calibration_batch_profile"]
    assert profile["profile_context_frames"] > 0
    assert profile["context_bootstrap_included_in_timing"] is True
    assert profile["final_pass_throughput_measurement"] is False
    assert profile["calibration"] is True and profile["cuda_measured"] is False
    assert profile["report_eligible"] is False
    assert int(result["model"]["calibration_attempts"]) == result["snn_calibration_samples"]
    for name, value in reference["model"].items():
        if name.endswith(("normalization_scale", "calibration_activation_max")):
            torch.testing.assert_close(result["model"][name], value, rtol=1e-5, atol=2e-6)


def test_stream_subset_and_existing_output_are_rejected_without_overwrite(tmp_path):
    config, source = _fixture(tmp_path)
    output = tmp_path / "synthetic-snn.pt"
    with pytest.raises(ValueError, match="every chronological"):
        engine.calibrate(config, source, output, samples=2, allow_unsealed_calibration=True)
    assert not output.exists()
    engine.calibrate(config, source, output, allow_unsealed_calibration=True)
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        engine.calibrate(config, source, output, allow_unsealed_calibration=True)
    assert output.read_bytes() == before


def test_stream_checkpoint_does_not_accept_static_input_configuration(tmp_path):
    config, source = _fixture(tmp_path)
    wrong = copy.deepcopy(config)
    wrong["model"] = _model_config()
    with pytest.raises(ValueError, match="model config|model_config|model configuration|checkpoint|static-window"):
        engine.calibrate(wrong, source, tmp_path / "forbidden.pt", allow_unsealed_calibration=True)
