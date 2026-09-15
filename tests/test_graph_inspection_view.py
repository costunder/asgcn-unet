"""Tiny SYNTHETIC display fixtures only. No original data or GPU/model execution."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from asgcn_unet import graph_inspection_view as view


def synthetic_report():
    # Two coincident events are distinct nodes; a third node is unrelated.
    rows = [[0, 10, 2.0, 3.0, 100.0, -1, 0.2, 0.3, 0.0],
            [1, 11, 2.0, 3.0, 100.0, 1, 0.2, 0.3, 0.0],
            [2, 12, 9.0, 9.0, 100.5, 1, 0.9, 0.9, 0.5]]
    return {
        "schema": "asgcn_raw_graph_audit_v1", "synthetic_fixture": True,
        "report_eligible": False, "total_directed_edges": None,
        "window": {"nodes": 3, "sensor_size": [11, 11], "readout_seconds": 101.0,
                   "cutoff_seconds": 100.0, "window_seconds": 1.0},
        "geometry": {"radius": 0.1, "position_dims": 3, "polarity_in_topology": False,
                     "time_scale_seconds": 1.0, "sequence_origin_seconds": 100.0},
        "source": {"file_name": "SYNTHETIC.h5", "frame_index": 0,
                   "identity": "synthetic_only"},
        "point_cloud": {"schema": "asgcn_graph_point_cloud_v1", "columns": view.COLUMNS,
                        "rows": rows, "nodes": 3, "coverage": "all_window_nodes"},
        "node_details": {str(row[0]): {
            "node_index": row[0], "raw_row_id": row[1],
            "original_source": {"x": row[2], "y": row[3], "timestamp": row[4], "polarity": row[5]},
            "preprocessed": {"x": row[2], "y": row[3], "timestamp_seconds": row[4]},
            "normalized_topology_coordinates": row[6:]} for row in rows[:2]},
        "queries": [{"node_index": 0, "raw_row_id": 10, "in_degree": 1,
                     "oracle_match": True, "all_window_sources_checked": 3,
                     "neighbors": [{"node_index": 1, "raw_row_id": 11,
                                    "distance_over_radius": 0.0, "distance": 0.0}]}],
    }


@pytest.fixture
def report():
    return synthetic_report()


@pytest.fixture
def budget(monkeypatch):
    monkeypatch.setattr(view, "preflight", lambda **kwargs: {"synthetic": True})
    return {"memory_budget_bytes": 128 * 1024**2,
            "reserve_memory_bytes": 64 * 1024**2, "cpu_threads": 1}


def test_full_cloud_and_zero_length_edge_keep_distinct_source_nodes(report):
    before = copy.deepcopy(report)
    payload = view.display_payload(report)
    assert payload["nodes"] == report["point_cloud"]["rows"]
    assert payload["coverage"] == "all_window_nodes"
    assert payload["total_directed_edges"] is None
    assert payload["queries"][0]["neighbors"] == [1]
    assert report == before


def test_old_saved_report_displays_only_available_neighborhoods(report):
    del report["point_cloud"]
    payload = view.display_payload(report)
    assert payload["coverage"] == "query_neighborhoods_only"
    assert len(payload["nodes"]) == 2
    assert payload["window"]["nodes"] == 3
    assert payload["total_directed_edges"] is None


def test_boolean_source_polarities_in_old_hdf5_audit_are_valid(report):
    del report["point_cloud"]
    report["node_details"]["0"]["original_source"]["polarity"] = False
    report["node_details"]["1"]["original_source"]["polarity"] = True
    assert [row[5] for row in view.display_payload(report)["nodes"]] == [-1, 1]


def test_edge_distance_must_match_endpoints_not_just_saved_radius(report):
    report["queries"][0]["neighbors"][0].update(node_index=2, raw_row_id=12)
    with pytest.raises(ValueError, match="endpoint"):
        view.display_payload(report)


def test_large_epoch_does_not_relax_physical_window_validation(report):
    report["window"].update(readout_seconds=1e9, cutoff_seconds=1e9 - 0.5, window_seconds=0.05)
    with pytest.raises(ValueError, match="clocks"):
        view.display_payload(report)


def test_full_cloud_raw_lineage_cannot_leave_delivered_suffix(report):
    report["window"].update(first_raw_row_id=10, last_raw_row_id=12)
    report["point_cloud"]["rows"][2][1] = 13
    with pytest.raises(ValueError, match="suffix"):
        view.display_payload(report)


@pytest.mark.parametrize("mutation", [
    lambda r: r["point_cloud"]["rows"].pop(),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(0, 0),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(1, 10),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(0, True),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(1, 2**53),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(2, float("nan")),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(2, 50),
    lambda r: r["point_cloud"]["rows"][1].__setitem__(8, 8),
    lambda r: r["queries"][0].__setitem__("in_degree", 0),
    lambda r: r["queries"][0].__setitem__("raw_row_id", True),
    lambda r: r["queries"][0].__setitem__("all_window_sources_checked", True),
    lambda r: r["queries"][0]["neighbors"][0].__setitem__("raw_row_id", True),
    lambda r: r["queries"][0].__setitem__("oracle_match", False),
    lambda r: r["queries"][0].__setitem__("all_window_sources_checked", 2),
    lambda r: r["queries"][0]["neighbors"][0].__setitem__("node_index", 0),
    lambda r: r["queries"][0]["neighbors"][0].__setitem__("node_index", 4),
    lambda r: r["queries"][0]["neighbors"][0].__setitem__("distance_over_radius", 1),
    lambda r: r["queries"][0]["neighbors"].append(r["queries"][0]["neighbors"][0]),
    lambda r: r["node_details"]["1"].__setitem__("raw_row_id", 555),
    lambda r: r.__setitem__("total_directed_edges", 2),
])
def test_invalid_or_misleading_graph_data_is_refused(report, mutation):
    mutation(report)
    with pytest.raises((ValueError, TypeError)):
        view.display_payload(report)


def test_recorded_complete_edge_total_is_allowed_but_not_invented(report):
    report.update(total_directed_edges=2,
                  full_count={"all_nodes_counted": True, "directed_edges": 2, "nodes": 3})
    assert view.display_payload(report)["total_directed_edges"] == 2


def test_html_is_offline_and_data_cannot_escape_script(report, budget, tmp_path):
    attack = '</script><script>window.PWNED=true</script> __INSPECTION_JS__'
    report["source"]["file_name"] = attack
    output = view.save_html(report, tmp_path / "graph.html", workspace=tmp_path, **budget)
    html = output.read_text(encoding="utf-8")
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert 'src="https://' not in html and 'href="http' not in html
    assert "fetch(" not in html and "WebSocket(" not in html
    assert "asgcn_graph_inspection_view_v1" in html
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        view.save_html(report, output, workspace=tmp_path, **budget)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".graph-view-*.tmp"))


def test_output_cannot_escape_workspace(report, budget, tmp_path):
    with pytest.raises(ValueError, match="workspace"):
        view.save_html(report, tmp_path.parent / "escaped.html", workspace=tmp_path, **budget)


def test_budget_refusal_has_no_partial_output(report, budget, tmp_path):
    budget["memory_budget_bytes"] = 1024
    with pytest.raises(MemoryError):
        view.save_html(report, tmp_path / "graph.html", workspace=tmp_path, **budget)
    assert not list(tmp_path.iterdir())


def test_source_json_is_preserved_on_export(report, budget, tmp_path):
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    before = source.read_bytes()
    view.export_saved_report(source, tmp_path / "graph.html", workspace=tmp_path, **budget)
    assert source.read_bytes() == before


@pytest.mark.parametrize("raw", ['{"schema":1,"schema":2}', '{"nodes":NaN}'])
def test_invalid_json_refuses_before_output(raw, budget, tmp_path):
    source = tmp_path / "graph.json"
    source.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        view.export_saved_report(source, tmp_path / "graph.html", workspace=tmp_path, **budget)
    assert not (tmp_path / "graph.html").exists()


def test_resource_refusal_precedes_saved_report_read(monkeypatch, tmp_path):
    def refuse(**kwargs):
        raise RuntimeError("synthetic resource refusal")
    monkeypatch.setattr(view, "preflight", refuse)
    with pytest.raises(RuntimeError, match="resource refusal"):
        view.export_saved_report(tmp_path / "not-read.json", tmp_path / "out.html", workspace=tmp_path,
                                 memory_budget_bytes=1024, reserve_memory_bytes=1024, cpu_threads=1)


def test_script_import_does_not_load_model_or_cuda():
    script = Path(__file__).resolve().parents[1] / "scripts/export_graph_inspection.py"
    code = f"import runpy,sys; runpy.run_path({str(script)!r},run_name='synthetic'); " \
           "assert not {'torch','numpy','h5py'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_browser_fixture(report, budget, tmp_path):
    """Synthetic display stress fixture, not original events or a quality result."""
    stress = copy.deepcopy(report)
    count = 28879
    rows = [[i, i, i % 320, (i // 320) % 240, 100 + (i % 129) / 256, 1 if i % 2 else -1,
             (i % 320) / 319, ((i // 320) % 240) / 239, (i % 129) / 256] for i in range(count)]
    stress["window"].update(nodes=count, sensor_size=[240, 320])
    stress["point_cloud"].update(nodes=count, rows=rows)
    stress["source"]["file_name"] = "SYNTHETIC_BROWSER_STRESS.h5"
    stress["node_details"] = {}
    queries = []
    for target in (0, count // 2, count - 1):
        neighbors = []
        for row in rows:
            distance = math.hypot(*[(a - b) / 0.1 for a, b in zip(row[6:], rows[target][6:], strict=True)])
            if row[0] != target and distance < 1:
                neighbors.append({"node_index": row[0], "raw_row_id": row[1],
                                  "distance_over_radius": distance, "distance": distance * 0.1})
        queries.append({"node_index": target, "raw_row_id": target, "in_degree": len(neighbors),
                        "oracle_match": True, "all_window_sources_checked": count, "neighbors": neighbors})
    stress["queries"] = queries
    view.save_html(stress, tmp_path / "graph.html", workspace=tmp_path,
                   **{**budget, "memory_budget_bytes": 512 * 1024**2})
    partial = copy.deepcopy(report)
    del partial["point_cloud"]
    view.save_html(partial, tmp_path / "partial.html", workspace=tmp_path, **budget)
    empty = copy.deepcopy(report)
    empty["window"]["nodes"] = 0
    empty["point_cloud"].update(rows=[], nodes=0)
    empty.update(queries=[], node_details={})
    view.save_html(empty, tmp_path / "empty.html", workspace=tmp_path, **budget)


def test_synthetic_hdf5_to_report_to_html_uses_actual_audited_rows(monkeypatch, budget, tmp_path):
    import torch

    from tests.test_raw_graph_audit import _audit_synthetic, _synthetic_hdr, audit

    monkeypatch.setattr(audit, "preflight", lambda **kwargs: {"synthetic": True})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    threads = torch.get_num_threads()
    try:
        report = _audit_synthetic(_synthetic_hdr(tmp_path / "SYNTHETIC.h5"), include_point_cloud=True)
        report["synthetic_fixture"] = True
        payload = view.display_payload(report)
        assert payload["coverage"] == "all_window_nodes"
        assert len(payload["nodes"]) == report["window"]["nodes"]
        assert payload["nodes"] == report["point_cloud"]["rows"]
        view.save_html(report, tmp_path / "graph.html", workspace=tmp_path, **budget)
        # Backward-compatible saved reports preserve bool source polarity, too.
        del report["point_cloud"]
        view.save_html(report, tmp_path / "legacy.html", workspace=tmp_path, **budget)
    finally:
        torch.set_num_threads(threads)
