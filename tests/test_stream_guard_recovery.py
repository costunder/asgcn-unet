"""Synthetic CPU recovery tests; these do not certify server memory or training."""

from __future__ import annotations

import copy
import json

import pytest

from asgcn_unet import stream_preflight
from tests.test_stream_preflight import SyntheticStreams, _config, _fixture_provenance


def _run(config, output, **kwargs):
    return stream_preflight.streaming_training_preflight(
        config, output, profile_samples=1, top_density_count=2, require_cuda=False, **kwargs,
    )


def test_small_guard_does_not_interrupt_complete_count_or_build_any_edges(monkeypatch, tmp_path):
    dataset = SyntheticStreams()
    _fixture_provenance(monkeypatch, dataset)
    config = _config()
    config["model"]["max_graph_edges"] = 8
    monkeypatch.setattr("asgcn_unet.stream_model._update", lambda *a, **k: pytest.fail("edge builder used in scan"))
    monkeypatch.setattr(stream_preflight, "_probe_stream_training", lambda *a, **k: pytest.fail("guard must block probe"))
    result = _run(config, tmp_path / "failed.json")
    assert not result["passed"]
    assert result["topology"]["scanned_samples"] == len(dataset)
    assert result["topology"]["scan_complete"]
    assert result["topology"]["max_readout_directed_edges"] == 6
    assert result["topology"]["max_prefix_union_directed_edges_upper_bound"] == 20
    assert result["topology"]["full_edge_tensors_materialized"] is False
    assert result["checks"]["complete_topology_scan"]
    assert result["checks"]["conservative_prefix_edge_guard"] is False
    assert result["failure"]["stage"] == "measured_edge_guard"
    assert result["guard_measurement"]["effective_max_graph_edges"] == 8
    assert config["model"]["max_graph_edges"] == 8


def test_explicit_measured_guard_creates_new_config_and_probes_that_exact_config(monkeypatch, tmp_path):
    import torch

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = SyntheticStreams()
        _fixture_provenance(monkeypatch, dataset)
        config = _config()
        config["model"]["max_graph_edges"] = 8
        before = copy.deepcopy(config)
        path = tmp_path / "new-config.json"
        result = _run(config, tmp_path / "profile.json", measured_guard_config_output=path, reserve_vram_mib=1024)
        assert result["passed"], result["failure"]
        assert result["report_eligible"] is False
        assert result["batch_training_probe"]["passed"] is True
        assert config == before
        derived = json.loads(path.read_text(encoding="utf-8"))
        assert derived["model"]["max_graph_edges"] == 20
        derived["model"]["max_graph_edges"] = 8
        assert derived == before
        assert result["config_provenance"]["config"]["model"]["max_graph_edges"] == 20
        assert result["input_config_provenance"]["config"]["model"]["max_graph_edges"] == 8
        assert result["guard_measurement"]["device_memory"]["measured"] is False
        assert result["guard_measurement"]["evaluation_memory_certified"] is False
    finally:
        torch.set_num_threads(original_threads)


def test_insufficient_storage_floor_stops_before_model_or_new_config(monkeypatch, tmp_path):
    dataset = SyntheticStreams()
    _fixture_provenance(monkeypatch, dataset)
    config = _config()
    config["model"]["max_graph_edges"] = 8
    monkeypatch.setattr(stream_preflight, "_cuda_memory_budget", lambda *a: {
        "measured": True, "available_after_reserve_mib": 0,
        "scope": "mocked_cpu_unit_test_not_actual_cuda_measurement",
    })
    monkeypatch.setattr(stream_preflight, "_probe_stream_training", lambda *a, **k: pytest.fail("unsafe probe"))
    path = tmp_path / "not-created.json"
    result = _run(config, tmp_path / "floor-failure.json", measured_guard_config_output=path, reserve_vram_mib=1024)
    assert result["failure"]["stage"] == "raw_graph_memory_floor"
    assert result["topology"]["scan_complete"]
    assert result["guard_measurement"]["raw_graph_storage_floor"]["bytes"] > 0
    assert not result["passed"] and not path.exists()


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_partial_scan_preserved_on_error_and_interrupt(monkeypatch, tmp_path, failure_type):
    class BrokenStreams(SyntheticStreams):
        def __getitem__(self, index):
            if index == 1:
                raise failure_type("synthetic scan interruption")
            return super().__getitem__(index)

    dataset = BrokenStreams()
    _fixture_provenance(monkeypatch, dataset)
    path = tmp_path / "partial.json"
    if failure_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            _run(_config(), path)
    else:
        _run(_config(), path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["topology"]["scanned_samples"] == 2
    assert report["topology"]["scan_complete"] is False
    assert report["topology"]["current_batch_indices"] == [1, 4]
    assert report["topology"]["samples"][0]["readout_directed_edges"] == 2
    assert report["topology"]["samples"][1] is None
    assert report["failure"]["stage"] == "count_only_topology"
    assert not report["passed"] and dataset.closed
    claimed = report.pop("commitment_sha256")
    assert claimed == stream_preflight._digest(report)


def test_config_collision_and_missing_reserve_rejected_before_device(monkeypatch, tmp_path):
    monkeypatch.setattr("asgcn_unet.utils.resolve_device", lambda *a: pytest.fail("device probe"))
    config_path = tmp_path / "existing.json"
    config_path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(_config(), tmp_path / "report.json", measured_guard_config_output=config_path, reserve_vram_mib=1024)
    assert config_path.read_text(encoding="utf-8") == "original"
    with pytest.raises(ValueError, match="reserve_vram_mib"):
        _run(_config(), tmp_path / "report.json", measured_guard_config_output=tmp_path / "new.json")
    assert not (tmp_path / "report.json").exists()


def test_storage_floor_includes_other_live_sequences_and_current_basis():
    records = [
        {"sequence_identity": ["g", lane], "sequence_index": frame,
         "readout_nodes": 3, "readout_directed_edges": 6}
        for lane in ("a", "b") for frame in (0, 1)
    ]
    result = stream_preflight._graph_storage_floor({"samples": records}, [[0, 2], [1, 3]])
    assert result["bytes"] == 2 * (6 * 24 + 3 * 72) + 2 * (6 * 56 + 3 * 72)
    assert result["batch_index"] == 1
    assert result["total_training_peak_estimate"] is False


def test_gap_with_early_predecessor_bounds_the_real_initial_watermark():
    import torch

    from asgcn_unet.batching import SequenceBatchSampler

    class GapStreams(SyntheticStreams):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if sample["metadata"]["sequence_index"] == 1:
                timing = sample["metadata"]["stream_time"]
                timing.update(interval_start_seconds=3.0, interval_end_seconds=4.0)
                # One early predecessor then a normal new event, in timestamp order.
                sample["events"][:, 2] = torch.tensor([1.2, 3.8], dtype=torch.float64)
            return sample

    dataset = GapStreams(frames=2)
    result = stream_preflight._scan_stream_topology(
        dataset, _config(), torch.device("cpu"), list(SequenceBatchSampler(dataset, 2)),
        top_density_count=2,
    )
    row = result["samples"][1]
    # The first actual arrival at 1.2 has cutoff -0.3, retaining old 0.2/0.8
    # and forming six directed edges. start(3.0)-window would wrongly lose them.
    assert row["prefix_union_nodes_upper_bound"] == 4
    assert row["prefix_union_directed_edges_upper_bound"] == 12
    assert row["readout_nodes"] == 1 and row["readout_directed_edges"] == 0


@pytest.mark.parametrize("guard", [False, True, 0, -1, 1.5])
def test_invalid_guard_is_not_silently_replaced_by_measured_value(monkeypatch, tmp_path, guard):
    monkeypatch.setattr("asgcn_unet.utils.resolve_device", lambda *a: pytest.fail("device probe"))
    config = _config()
    config["model"]["max_graph_edges"] = guard
    with pytest.raises(ValueError, match="max_graph_edges"):
        _run(config, tmp_path / "report.json", measured_guard_config_output=tmp_path / "config.json",
             reserve_vram_mib=1024)
    assert not (tmp_path / "report.json").exists()
