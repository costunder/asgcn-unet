"""Synthetic CPU diagnostic tests, not real-data or model/GPU validation.

All data below are deliberately tiny, locally generated fixtures. Resource
snapshots are mocked only for these synthetic tests; no real-data safety check
is bypassed and no experiment configuration or stored result is changed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from asgcn_unet import diagnostic_resources
from asgcn_unet.diagnostic_resources import DiagnosticResourceError

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_raw_event_graph.py"
_SPEC = importlib.util.spec_from_file_location("audit_raw_event_graph", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = audit
_SPEC.loader.exec_module(audit)


def test_import_is_stdlib_only_and_does_not_import_models_or_cuda():
    code = f"""
import sys
import runpy
runpy.run_path({str(SCRIPT)!r}, run_name='synthetic_import_check')
for name in sys.modules:
    assert name.split('.')[0] not in {{'torch', 'numpy', 'h5py'}}, name
    assert not name.startswith(('asgcn_unet.model', 'asgcn_unet.stream_model',
                                'asgcn_unet.implicit_model')), name
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_window_is_inclusive_at_both_physical_time_boundaries():
    raw = np.array([99, 100, 101, 102], dtype=np.float64)
    rows, seconds = audit.select_window_rows(
        raw, timestamp_scale_to_seconds=0.5,
        readout_seconds=51.0, window_seconds=1.0,
    )
    np.testing.assert_array_equal(rows, [1, 2, 3])
    np.testing.assert_array_equal(seconds, [50.0, 50.5, 51.0])
    assert rows.dtype == np.int64
    assert seconds.dtype == np.float64


def test_delivered_prefix_with_future_events_is_refused_not_silently_filtered():
    with pytest.raises(ValueError, match="after.*readout"):
        audit.select_window_rows(
            np.array([99.0, 100.0, 103.0]), timestamp_scale_to_seconds=0.5,
            readout_seconds=51.0, window_seconds=1.0,
        )


def test_window_keeps_duplicate_records_as_distinct_source_rows():
    rows, seconds = audit.select_window_rows(
        np.array([1.0, 1.0, 1.0]), timestamp_scale_to_seconds=1.0,
        readout_seconds=1.0, window_seconds=0.5,
    )
    np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(seconds, [1.0, 1.0, 1.0])


def test_window_has_no_8192_event_cap_or_hidden_sampling():
    # Larger than the old static-profile event cap, but only a small CPU vector.
    raw = np.arange(9001, dtype=np.float64)
    rows, seconds = audit.select_window_rows(
        raw, timestamp_scale_to_seconds=1.0,
        readout_seconds=9000.0, window_seconds=9000.0,
    )
    np.testing.assert_array_equal(rows, np.arange(9001, dtype=np.int64))
    np.testing.assert_array_equal(seconds, raw)


@pytest.mark.parametrize("raw", [[0.0, 2.0, 1.0], [0.0, np.nan], [0.0, np.inf]])
def test_full_supplied_clock_is_checked_even_outside_selected_window(raw):
    with pytest.raises((ValueError, RuntimeError)):
        audit.select_window_rows(
            np.array(raw), timestamp_scale_to_seconds=1.0,
            readout_seconds=100.0, window_seconds=1.0,
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_timestamp_scale_is_explicit_finite_and_positive(scale):
    with pytest.raises((ValueError, RuntimeError)):
        audit.select_window_rows(
            np.array([0.0, 1.0]), timestamp_scale_to_seconds=scale,
            readout_seconds=1.0, window_seconds=1.0,
        )


def test_distinct_int64_timestamps_that_collapse_in_float64_are_refused():
    with pytest.raises(ValueError):
        audit.select_window_rows(
            np.array([2**53, 2**53 + 1], dtype=np.int64),
            timestamp_scale_to_seconds=1.0, readout_seconds=float(2**53 + 4),
            window_seconds=8.0,
        )


def test_normalized_geometry_uses_sensor_axes_and_fixed_sequence_origin():
    import torch

    events = np.array([[100.0, 50.0, 1001.0, -1.0],
                       [50.0, 25.0, 1002.0, 1.0]], dtype=np.float64)
    positions = audit.normalized_positions(
        events, (101, 201), origin_seconds=1000.0, time_scale_seconds=2.0,
    )
    assert positions.dtype == torch.float64
    assert positions.device.type == "cpu"
    torch.testing.assert_close(
        positions, torch.tensor([[0.5, 0.5, 0.5, 0.0],
                                 [0.25, 0.25, 1.0, 1.0]], dtype=torch.float64),
        rtol=0, atol=0,
    )
    # The second event gets identical geometry when queried in another window.
    second = audit.normalized_positions(
        events[1:], (101, 201), origin_seconds=1000.0, time_scale_seconds=2.0,
    )
    torch.testing.assert_close(second[0], positions[1], rtol=0, atol=0)


@pytest.mark.parametrize("pair_budget", [1, 3, 100])
def test_query_uses_all_sources_strict_boundary_and_distinct_duplicate_rows(pair_budget):
    import torch

    radius = 0.25
    positions = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],  # Opposite polarity, same geometric position.
         [np.nextafter(radius, 0.0), 0.0, 0.0, 0.0],
         [radius, 0.0, 0.0, 0.0],  # Exactly on boundary: no edge.
         [np.nextafter(radius, np.inf), 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    raw_ids = np.arange(40, 45, dtype=np.int64)
    result = audit.query_neighbors(
        positions, raw_ids, [0], radius=radius, candidate_pair_budget=pair_budget,
    )
    assert len(result) == 1
    neighbors = result[0]["neighbors"]
    assert {row["node_index"] for row in neighbors} == {1, 2}
    assert {row["raw_row_id"] for row in neighbors} == {41, 42}
    assert all(row["node_index"] != 0 for row in neighbors)
    by_node = {row["node_index"]: row for row in neighbors}
    assert by_node[1]["distance_over_radius"] == 0.0
    assert by_node[2]["distance_over_radius"] < 1.0
    # Neither source was a requested destination, so this also checks against
    # erroneously building an induced graph over the selected query subset.


def test_multiple_queries_match_independent_full_source_cpu_oracle():
    import torch

    positions = torch.tensor(
        [[0, 0, 0, 0], [0.1, 0.1, 0.0, 1], [0.2, 0, 0.1, 0],
         [1, 1, 1, 1], [0, 0, 0.1, 1]], dtype=torch.float64,
    )
    raw_ids = np.array([101, 105, 112, 129, 131], dtype=np.int64)
    radius = 0.2
    queries = [2, 0]
    result = audit.query_neighbors(
        positions, raw_ids, queries, radius=radius, candidate_pair_budget=2,
    )
    assert len(result) == len(queries)
    for query, report in zip(queries, result, strict=True):
        distances = torch.linalg.vector_norm(
            (positions[:, :3] - positions[query, :3]) / radius, dim=1,
        )
        expected = {
            index for index, distance in enumerate(distances.tolist())
            if index != query and distance < 1.0
        }
        assert {row["node_index"] for row in report["neighbors"]} == expected
        for row in report["neighbors"]:
            assert row["raw_row_id"] == raw_ids[row["node_index"]]
            assert row["distance_over_radius"] == pytest.approx(
                float(distances[row["node_index"]]), rel=0, abs=1e-15,
            )


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "synthetic-report.json"
    original = b'{"original": true}\n'
    output.write_bytes(original)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        audit.save_report({"synthetic": True}, output, workspace=tmp_path)
    assert output.read_bytes() == original


def test_new_report_is_written_once_as_json(tmp_path):
    output = tmp_path / "synthetic-report.json"
    report = {"synthetic": True, "report_eligible": False, "queries": []}
    audit.save_report(report, output, workspace=tmp_path)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        audit.save_report(report, output, workspace=tmp_path)


def test_report_cannot_escape_explicit_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "outside.json"
    with pytest.raises((ValueError, RuntimeError)):
        audit.save_report({"synthetic": True}, output, workspace=workspace)
    assert not output.exists()


def test_memory_refusal_happens_before_dataset_open(monkeypatch, tmp_path):
    import h5py

    calls = []

    def refuse_synthetic_budget(**kwargs):
        calls.append(kwargs)
        raise DiagnosticResourceError("synthetic resource shortfall")

    def dataset_must_not_open(*args, **kwargs):
        pytest.fail("Dataset read happened after a failed resource preflight")

    monkeypatch.setattr(audit, "preflight", refuse_synthetic_budget)
    monkeypatch.setattr(h5py, "File", dataset_must_not_open)
    with pytest.raises(DiagnosticResourceError, match="synthetic resource shortfall"):
        audit.audit_raw_event_graph(
            source_file=tmp_path / "not-read.h5", frame_index=0,
            window_seconds=0.05, time_scale_seconds=0.05, radius=0.08,
            timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=1.0,
            cpu_threads=1, memory_budget_bytes=1024, reserve_memory_bytes=1024,
        )
    assert len(calls) == 1
    assert calls[0]["budget_bytes"] == 1024
    assert calls[0]["reserve_bytes"] == 1024


@pytest.fixture
def synthetic_resource_snapshot(monkeypatch):
    """Explicit fake allocation only for the small synthetic HDF5 tests below."""
    import torch

    def snapshot():
        return {
            "headroom_bytes": 2 * 1024**3,
            "system": {"total_bytes": 4 * 1024**3, "available_bytes": 2 * 1024**3,
                       "process_rss_bytes": 32 * 1024**2},
            "cpu": {"effective_cores": 4, "affinity_count": 4},
            "cgroup": {"memory_headroom_bytes": 2 * 1024**3},
        }

    monkeypatch.setattr(diagnostic_resources, "_snapshot", snapshot)
    # The standalone command owns its CPU pools. Restore them after the
    # in-process synthetic test so other tests do not inherit these settings.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, os.environ.get(key, "1"))
    previous_threads = torch.get_num_threads()
    yield
    torch.set_num_threads(previous_threads)


def _synthetic_hdr(path, *, stored_indices=True):
    import h5py

    with h5py.File(path, "x") as handle:
        events = handle.create_group("events")
        events.create_dataset("xs", data=np.ones(8, dtype=np.int16))
        events.create_dataset("ys", data=np.ones(8, dtype=np.int16))
        events.create_dataset("ps", data=np.zeros(8, dtype=np.bool_))
        events.create_dataset("ts", data=np.array(
            [9.0, 10.0, 10.5, 10.5, 10.75, 10.875, 11.0, 11.125], dtype=np.float64,
        ))
        images = handle.create_group("images")
        for index, (timestamp, end) in enumerate(((11.0, 5), (11.125, 7))):
            image = images.create_dataset(f"image{index:09d}", data=np.zeros((5, 7), np.uint8))
            # Different event/frame source units exercise independent scales.
            image.attrs["timestamp"] = timestamp * 2
            if stored_indices:
                image.attrs["event_idx"] = end
    return path


def _audit_synthetic(path, *, frame_index=0):
    return audit.audit_raw_event_graph(
        source_file=path, frame_index=frame_index,
        window_seconds=0.5, time_scale_seconds=1.0, radius=1.0,
        timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=0.5,
        cpu_threads=1, memory_budget_bytes=512 * 1024**2,
        reserve_memory_bytes=128 * 1024**2,
    )


def test_float_target_metadata_uses_explicit_training_reader_settings(tmp_path, synthetic_resource_snapshot):
    import h5py

    path = tmp_path / "synthetic-float-target.h5"
    with h5py.File(path, "x") as handle:
        events = handle.create_group("events")
        for name in ("xs", "ys", "ps"):
            events.create_dataset(name, data=np.ones(2, dtype=np.int16))
        events.create_dataset("ts", data=np.array([0.0, 0.125], dtype=np.float64))
        image = handle.create_group("images").create_dataset(
            "image000000000", data=np.ones((4, 4, 3), dtype=np.float32),
        )
        image.attrs["timestamp"] = 0.25
        image.attrs["event_idx"] = 2
    options = {"target_channels": 3, "target_normalization": {"mode": "known_scale", "scale": 2.0},
               "tone_map": "linear"}
    report = audit.audit_raw_event_graph(
        source_file=path, frame_index=0, window_seconds=0.5, time_scale_seconds=1.0,
        radius=1.0, timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=1.0,
        cpu_threads=1, memory_budget_bytes=512 * 1024**2, reserve_memory_bytes=128 * 1024**2,
        target_options=options,
    )
    assert report["window"]["nodes"] == 2
    assert report["target_reader_options"] == options


def test_diagnostic_calls_production_coordinate_builder(monkeypatch):
    import torch

    from asgcn_unet import stream_geometry

    original = stream_geometry.physical_node_positions
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_geometry, "physical_node_positions", observed)
    coordinates = audit.normalized_positions(np.array([[1.0, 1.0, 10.0, -1.0]]), (4, 4),
                                             origin_seconds=9.0, time_scale_seconds=2.0)
    assert len(calls) == 1
    assert torch.equal(coordinates, torch.tensor([[1 / 3, 1 / 3, 0.5, 0.0]], dtype=torch.float64))


@pytest.mark.parametrize("stored_indices", [True, False])
def test_hdr_predecessor_end_is_exclusive_no_undelivered_or_future_nodes(
    tmp_path, synthetic_resource_snapshot, stored_indices,
):
    path = _synthetic_hdr(tmp_path / "synthetic.h5", stored_indices=stored_indices)
    original = path.read_bytes()
    report = _audit_synthetic(path)
    assert report["source"]["frame_index"] == 0
    assert report["source"]["delivery_end_idx_exclusive"] == 5
    assert report["source"]["validated_timestamp_prefix_rows"] == 5
    assert report["window"]["nodes"] == 3
    assert report["window"]["first_raw_row_id"] == 2
    assert report["window"]["last_raw_row_id"] == 4
    assert report["window"]["cutoff_inclusive"] is True
    assert report["window"]["cutoff_seconds"] == 10.5
    assert report["window"]["readout_seconds"] == 11.0
    assert report["window"]["sampling_factor"] == 1
    assert report["window"]["max_events"] is None
    assert report["window"]["crop"] is None
    assert report["geometry"]["sequence_origin_seconds"] == 9.0
    trace = report["node_details"]["0"]
    assert trace["raw_row_id"] == 2
    assert trace["original_source"] == {
        "x": 1, "y": 1, "timestamp": 10.5, "polarity": False,
    }
    assert trace["preprocessed"] == {
        "x": 1.0, "y": 1.0, "timestamp_seconds": 10.5,
    }
    assert report["geometry"]["polarity_in_topology"] is False
    assert report["geometry"]["feature_parity_audited"] is False
    np.testing.assert_allclose(
        trace["normalized_topology_coordinates"], [1 / 6, 1 / 4, 1.5], rtol=0, atol=0,
    )
    assert {query["raw_row_id"] for query in report["queries"]} == {2, 3, 4}
    for query in report["queries"]:
        expected = {2, 3, 4} - {query["raw_row_id"]}
        assert {neighbor["raw_row_id"] for neighbor in query["neighbors"]} == expected
        assert query["in_degree"] == 2
        assert query["all_window_sources_checked"] == 3
        assert query["oracle_match"] is True
    # Row5's time precedes readout but its delivery is later: do not reconstruct
    # a different graph by including every timestamp <= readout independently
    # of the authoritative exclusive delivery index.
    assert report["schema"] == "asgcn_raw_graph_audit_v1"
    assert report["report_eligible"] is False
    assert report["paper_exact"] is False
    assert report["total_directed_edges"] is None
    assert path.read_bytes() == original


def test_sequence_origin_does_not_shift_between_frames_and_cuda_is_not_queried(
    monkeypatch, tmp_path, synthetic_resource_snapshot,
):
    import torch

    def cuda_must_not_be_called(*args, **kwargs):
        pytest.fail("CPU graph diagnostic attempted a CUDA operation")

    for name in ("is_available", "device_count", "_lazy_init", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, name, cuda_must_not_be_called)
    path = _synthetic_hdr(tmp_path / "synthetic.h5")
    first = _audit_synthetic(path, frame_index=0)
    second = _audit_synthetic(path, frame_index=1)
    assert first["geometry"]["sequence_origin_seconds"] == 9.0
    assert second["geometry"]["sequence_origin_seconds"] == 9.0
    assert first["window"]["cutoff_seconds"] != second["window"]["cutoff_seconds"]
    assert second["window"]["first_raw_row_id"] == 4
    assert second["window"]["last_raw_row_id"] == 6
    assert first["resources"]["cuda_queried"] is False
    assert second["resources"]["cuda_queried"] is False


def test_original_int64_timestamp_trace_is_not_reconstructed_from_float64(
    tmp_path, synthetic_resource_snapshot,
):
    import h5py

    path = tmp_path / "synthetic-integer-clock.h5"
    original_timestamp = 2**53 + 1
    with h5py.File(path, "x") as handle:
        events = handle.create_group("events")
        for name in ("xs", "ys", "ps"):
            events.create_dataset(name, data=np.zeros(2, dtype=np.int16))
        events.create_dataset("ts", data=np.array([original_timestamp] * 2, dtype=np.int64))
        image = handle.create_group("images").create_dataset(
            "image000000000", data=np.zeros((2, 2), np.uint8),
        )
        image.attrs["timestamp"] = float(2**53 + 4)
        image.attrs["event_idx"] = 2
    report = audit.audit_raw_event_graph(
        source_file=path, frame_index=0, window_seconds=8.0, time_scale_seconds=1.0,
        radius=1.0, timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=1.0,
        cpu_threads=1, memory_budget_bytes=512 * 1024**2,
        reserve_memory_bytes=128 * 1024**2,
    )
    # This is a preservation check, not a claim that float64 geometry retains
    # every bit of a large integer timestamp. JSON rawtrace must retain it.
    for node in report["node_details"].values():
        raw_timestamp = node["original_source"]["timestamp"]
        assert type(raw_timestamp) is int
        assert raw_timestamp == original_timestamp


@pytest.mark.parametrize("coordinates", [
    [], [[0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0]] * 7,
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.25, 0.0, 0.0], [3.0, 2.0, 1.0]],
])
@pytest.mark.parametrize("pair_budget", [1, 11])
def test_full_count_matches_independent_small_full_oracle(coordinates, pair_budget):
    import torch

    from asgcn_unet.implicit_radius import ImplicitRadiusIndex

    positions = torch.zeros((len(coordinates), 4), dtype=torch.float64)
    if coordinates:
        positions[:, :3] = torch.tensor(coordinates, dtype=torch.float64)
    radius = 0.25
    index = ImplicitRadiusIndex(
        positions, torch.zeros(len(positions), dtype=torch.long), batch_size=1,
        radius=radius, position_dims=3, chunk_size=3, candidate_pair_budget=pair_budget,
    )
    degrees, result = audit.count_all_degrees(index)
    # This full NxN oracle is intentionally only a tiny synthetic test fixture.
    distances = torch.linalg.vector_norm(
        (positions[:, None, :3] - positions[None, :, :3]) / radius, dim=-1,
    )
    expected_edges = distances < 1
    expected_edges.fill_diagonal_(False)
    expected = expected_edges.sum(dim=0)
    torch.testing.assert_close(degrees, expected, rtol=0, atol=0)
    assert result["nodes"] == len(positions)
    assert result["directed_edges"] == int(expected.sum())
    assert result["undirected_edges"] * 2 == result["directed_edges"]
    assert result["isolated_nodes"] == int((expected == 0).sum())
    assert result["degree_min"] == (int(expected.min()) if len(expected) else None)
    assert result["degree_max"] == (int(expected.max()) if len(expected) else None)
    assert result["degree_mean"] == (int(expected.sum()) / len(expected) if len(expected) else None)
    assert result["count_time_s"] >= 0
    assert result["all_nodes_counted"] is True
    assert result["edge_list_materialized"] is False
    assert result["full_graph_oracle_verified"] is False


def test_default_selected_query_mode_does_not_count_all_nodes(
    tmp_path, monkeypatch, synthetic_resource_snapshot,
):
    def forbidden_count(*args, **kwargs):
        pytest.fail("Default selected-query diagnostic unexpectedly counted every node")

    monkeypatch.setattr(audit, "count_all_degrees", forbidden_count)
    report = _audit_synthetic(_synthetic_hdr(tmp_path / "synthetic.h5"))
    assert report["count_all_nodes"] is False
    assert report["full_count"] is None
    assert report["total_directed_edges"] is None
    assert report["timings"]["count_time_s"] is None
    assert report["resources"]["all_node_degree_scratch_bytes"] == 0


def test_opt_in_counts_full_window_and_reuses_one_index_for_selected_oracle(
    tmp_path, monkeypatch, synthetic_resource_snapshot,
):
    from asgcn_unet.implicit_radius import ImplicitRadiusGraph, ImplicitRadiusIndex

    calls = []
    original_init = ImplicitRadiusIndex.__init__

    def observed_init(index, *args, **kwargs):
        calls.append(len(args[0]))
        original_init(index, *args, **kwargs)

    def no_graph_materialization(*args, **kwargs):
        pytest.fail("Count-only diagnostic attempted to construct a model graph")

    monkeypatch.setattr(ImplicitRadiusIndex, "__init__", observed_init)
    monkeypatch.setattr(ImplicitRadiusGraph, "from_counted_nodes", no_graph_materialization)
    path = _synthetic_hdr(tmp_path / "synthetic.h5")
    report = audit.audit_raw_event_graph(
        source_file=path, frame_index=0, window_seconds=0.5, time_scale_seconds=1.0,
        radius=1.0, timestamp_scale_to_seconds=1.0, interval_timestamp_scale_to_seconds=0.5,
        cpu_threads=1, memory_budget_bytes=512 * 1024**2,
        reserve_memory_bytes=128 * 1024**2, query_indices=[0], count_all_nodes=True,
    )
    assert calls == [3]
    assert report["count_all_nodes"] is True
    assert report["total_directed_edges"] == 6
    full = report["full_count"]
    assert (full["nodes"], full["degree_min"], full["degree_mean"], full["degree_max"]) == (3, 2, 2, 2)
    assert full["isolated_nodes"] == 0
    assert full["selected_oracle_queries_checked"] == 1
    assert full["selected_oracle_degree_sum"] == 2
    assert full["selected_oracle_degrees_match"] is True
    assert full["full_graph_oracle_verified"] is False
    assert len(report["queries"]) == 1  # No all-node neighbor JSON expansion.
    assert report["report_eligible"] is False
    assert report["paper_exact"] is False
    assert report["timings"]["count_time_s"] >= 0
    assert report["resources"]["all_node_degree_scratch_bytes"] == 3 * 64


def test_count_mode_flag_is_explicit_opt_in():
    arguments = ["--source-file", "synthetic.h5", "--frame-index", "0",
                 "--window-seconds", "0.05", "--time-scale-seconds", "0.05", "--radius", "0.08",
                 "--timestamp-scale-to-seconds", "1", "--interval-timestamp-scale-to-seconds", "1",
                 "--cpu-threads", "1", "--memory-budget-mib", "512", "--reserve-memory-mib", "128",
                 "--output", "synthetic.json"]
    assert audit.build_parser().parse_args(arguments).count_all_nodes is False
    assert audit.build_parser().parse_args([*arguments, "--count-all-nodes"]).count_all_nodes is True
