"""Synthetic CPU integration tests for actual-result diagnostic generation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from asgcn_unet.offline_viewer import ExportLimits, OfflineViewerError, _png, _PNGData
from asgcn_unet.result_visualization import (
    _check_gt,
    _encoding_plan,
    _identities,
    _input_snapshot,
    _unchanged,
    generate_result_visualizations,
)
from tests.test_result_viewer import viewer_fixture


@pytest.fixture
def artifacts(tmp_path):
    return viewer_fixture(tmp_path)


def snapshot(root: Path) -> dict:
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }


def mock_resource_gate(monkeypatch):
    calls = []

    def gate(**kwargs):
        calls.append(kwargs)
        return {
            "headroom_bytes": 8 * 1024**3,
            "passed": True,
            "fixture": "SYNTHETIC RESOURCE GATE; not live server proof",
        }

    monkeypatch.setattr("asgcn_unet.diagnostic_resources.preflight", gate)
    return calls


def run_generate(root, config, output):
    return generate_result_visualizations(
        root,
        output,
        configs={"hdr": config},
        memory_budget_bytes=256 * 1024**2,
        reserve_memory_bytes=128 * 1024**2,
        cpu_threads=2,
        display_edges=13,
        progress=lambda message: None,
    )


def test_all_real_fixture_png_graphs_connected_without_model_or_dataset_index(
    artifacts, tmp_path, monkeypatch
):
    root, config = artifacts
    before = snapshot(root)
    calls = mock_resource_gate(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "No inference, CUDA, dataset-wide index, or original full-edge construction"
        )

    monkeypatch.setattr("torch.cuda._lazy_init", forbidden)
    monkeypatch.setattr("asgcn_unet.data.EventHDRDataset.__init__", forbidden)
    monkeypatch.setattr("asgcn_unet.model.ASGCNUNet.forward_sample", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.build_event_graph", forbidden)
    output = tmp_path / "generated"
    result = run_generate(root, config, output)
    assert result["complete"] is True
    assert result["generated_frames"] == 2
    assert result["model_inference"] is False
    assert result["graph_reconstruction"] is True
    assert result["report_eligible"] is False
    assert calls and all(c["cpu_threads"] == 2 for c in calls)
    assert snapshot(root) == before
    for index in (0, 1):
        directory = output / "hdr" / f"{index:08d}"
        for name in (
            "gt.png",
            "ann.png",
            "snn_standard_if_T4.png",
            "events-xy.png",
            "graph-xyt.png",
            "graph.json",
        ):
            assert (directory / name).is_file()
        graph = json.loads((directory / "graph.json").read_text())
        assert graph["statistics"]["nodes"] > 0
        assert graph["offline_neighbor_scope"] == "included_edges_only"
        expected = next((root / "hdr" / "ann" / "predictions").glob(f"{index:08d}_*_pred.png"))
        assert (directory / "ann.png").read_bytes() == expected.read_bytes()
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "diagnostic_images" in html and "actual_directed_edges" in html
    assert "http://127.0.0.1" not in html
    embedded = json.loads(
        re.search(
            r'<script id="offline-data" type="application/json">(.*?)</script>', html, re.DOTALL
        )[1]
    )
    assert embedded["export"]["graph_reconstruction"] is True
    assert embedded["export"]["model_inference"] is False
    assert result["disk_plan"]["planned_bytes"] > 0
    assert result["retained_metadata_estimate_bytes"] > 0
    assert (output / "generation.json").is_file()
    assert not (output / "generation.failed.json").exists()


def test_resource_refusal_before_input_reads_or_output(artifacts, tmp_path, monkeypatch):
    root, config = artifacts

    def refuse(**kwargs):
        raise RuntimeError("mock measured insufficient RAM")

    monkeypatch.setattr("asgcn_unet.diagnostic_resources.preflight", refuse)
    monkeypatch.setattr(
        "asgcn_unet.result_visualization.build_payload",
        lambda *a, **k: pytest.fail("Gate must run first"),
    )
    output = tmp_path / "refused"
    with pytest.raises(RuntimeError, match="insufficient RAM"):
        run_generate(root, config, output)
    assert not output.exists()


def test_original_graph_stats_mismatch_is_failure_not_graph_free_success(
    artifacts, tmp_path, monkeypatch
):
    root, config = artifacts
    mock_resource_gate(monkeypatch)
    from asgcn_unet.diagnostic_graph import build_diagnostic_graph

    def wrong(*args, **kwargs):
        graph = build_diagnostic_graph(*args, **kwargs)
        graph["statistics"]["actual_directed_edges"] += 1
        return graph

    monkeypatch.setattr("asgcn_unet.diagnostic_graph.build_diagnostic_graph", wrong)
    output = tmp_path / "mismatch"
    with pytest.raises(OfflineViewerError, match="Graph edges"):
        run_generate(root, config, output)
    assert not (output / "index.html").exists()
    failure = json.loads((output / "generation.failed.json").read_text())
    assert failure["complete"] is False


def test_missing_source_does_not_produce_empty_html(artifacts, tmp_path, monkeypatch):
    root, config = artifacts
    mock_resource_gate(monkeypatch)

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing actual source")

    monkeypatch.setattr("asgcn_unet.diagnostic_sample.read_diagnostic_sample", missing)
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        run_generate(root, config, output)
    assert not (output / "index.html").exists()


def test_source_gt_mismatch_refuses_publish(artifacts, tmp_path, monkeypatch):
    root, config = artifacts
    mock_resource_gate(monkeypatch)
    from asgcn_unet.diagnostic_sample import read_diagnostic_sample

    def changed(*args, **kwargs):
        sample = read_diagnostic_sample(*args, **kwargs)
        sample["target"] = sample["target"] * 0
        return sample

    monkeypatch.setattr("asgcn_unet.diagnostic_sample.read_diagnostic_sample", changed)
    output = tmp_path / "different-target"
    with pytest.raises(OfflineViewerError, match="GT pixels"):
        run_generate(root, config, output)
    assert not (output / "index.html").exists()


def test_existing_output_preserved(artifacts, tmp_path):
    root, config = artifacts
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(OfflineViewerError, match="already exists"):
        run_generate(root, config, output)
    assert marker.read_text() == "preserve"


def test_selected_identity_streaming_matches_original_without_retaining_all(artifacts):
    root, _ = artifacts
    path = root / "hdr" / "ann" / "metrics.json"
    original = json.loads(path.read_text())
    report = _identities(path, frozenset({1}), ExportLimits())
    assert set(report["source_identities"]) == {1}
    expected = original["evaluation_protocol"]["evaluation_dataset"]["contract"]["sampling"][
        "selected"
    ][1]
    assert report["source_identities"][1] == expected


def test_selected_identity_tampered_small_model_contract_refused(artifacts):
    root, _ = artifacts
    path = root / "hdr" / "ann" / "metrics.json"
    report = json.loads(path.read_text())
    report["evaluation_protocol"]["model_config"]["contract"]["graph_radius"] = 0.0001
    path.write_text(json.dumps(report))
    with pytest.raises(OfflineViewerError, match="hash mismatch"):
        _identities(path, frozenset({0}), ExportLimits())


def test_full_resolution_scratch_checked_before_png_decode(artifacts, monkeypatch):
    import torch

    root, _ = artifacts
    target = next((root / "hdr" / "ann" / "predictions").glob("*_gt.png"))
    item = _png(target, ExportLimits())
    sample = {
        "sensor_size": [item["height"], item["width"]],
        "target": torch.zeros(item["channels"], item["height"], item["width"]),
    }
    monkeypatch.setattr(
        "PIL.Image.open", lambda *a, **k: pytest.fail("No decode before budget check")
    )
    with pytest.raises(OfflineViewerError, match="scratch exceeds"):
        _check_gt(sample, item, 1)


def test_changed_gt_refused_before_png_decode(artifacts, monkeypatch):
    root, _ = artifacts
    target = next((root / "hdr" / "ann" / "predictions").glob("*_gt.png"))
    item = _png(target, ExportLimits())
    target.write_bytes(target.read_bytes() + b"fixture mutation")
    monkeypatch.setattr(
        "PIL.Image.open", lambda *a, **k: pytest.fail("Changed input must not decode")
    )
    with pytest.raises(OfflineViewerError, match="GT changed"):
        _check_gt({}, item, 256 * 1024**2)


def test_png_embedding_budget_checked_without_read(monkeypatch, tmp_path):
    source = _PNGData(tmp_path / "not-read.png", (0, 0, 1024**3, 0, 0), "test")
    with pytest.raises(OfflineViewerError, match="embedding scratch"):
        _encoding_plan({"images": {"test": {"data_url": source}}}, 64 * 1024**2)


def test_disk_refusal_before_output_creation(artifacts, tmp_path, monkeypatch):
    from types import SimpleNamespace

    root, config = artifacts
    mock_resource_gate(monkeypatch)
    monkeypatch.setattr(
        "asgcn_unet.result_visualization.shutil.disk_usage", lambda p: SimpleNamespace(free=0)
    )
    output = tmp_path / "disk-refused"
    with pytest.raises(OfflineViewerError, match="Insufficient output disk"):
        run_generate(root, config, output)
    assert not output.exists()


def test_multi_pass_report_change_refused(artifacts):
    root, _ = artifacts
    files = _input_snapshot(root, 1024**2)
    path = root / "hdr" / "ann" / "metrics.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(OfflineViewerError, match="Input changed"):
        _unchanged(files)
