"""CPU-only viewer smoke tests. All images here are synthetic test fixtures."""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pytest
import torch
from PIL import Image

from asgcn_unet.data import build_dataset
from asgcn_unet.engine import (
    _dataset_sample_identity,
    _prediction_artifact_stem,
)
from asgcn_unet.graph_preview import build_graph_preview
from asgcn_unet.result_viewer import ResultViewer, ViewerError, ViewerHTTPServer
from asgcn_unet.utils import save_image
from tests.fixtures import make_eventhdr
from tests.test_viewer_protocol import make_viewer_report


def viewer_fixture(base: Path) -> tuple[Path, Path]:
    """Create explicitly non-reporting fixtures; these are NOT learned predictions."""
    data_root = base / "synthetic-smoke-data"
    make_eventhdr(data_root, frames=4)
    config = {
        "device": "cpu",
        "dataset": {
            "type": "eventhdr", "root": str(data_root), "target_channels": 1,
            "max_events": 32, "crop_size": None, "frame_stride": 1,
            "tone_map": "log", "tone_map_mu": 5000.0,
            "target_normalization": {"mode": "integer_dtype_max"},
        },
        "model": {
            "graph_radius": 0.4, "graph_position_dims": 3,
            "event_sampling_factor": 1, "graph_chunk_size": 16,
            "max_graph_edges": 20_000,
        },
        "eval": {"precision": "fp32"},
    }
    config_path = base / "viewer-smoke-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dataset = build_dataset(config["dataset"], split="eval")
    root = base / "synthetic-smoke-eval"
    try:
        for mode in ("ann", "snn_standard_if_T4"):
            run = root / "hdr" / mode
            run.mkdir(parents=True)
            rows = []
            for index in range(len(dataset)):
                sample = dataset[index]
                graph = build_graph_preview(sample, config["model"], max_graph_edges=20_000)
                rows.append({
                    "sample_id": sample["sample_id"], "scene": "synthetic-smoke-only",
                    "psnr": 11.123456789, "ssim": 0.6123456789, "rmse": 0.3,
                    "nodes": graph.payload["statistics"]["nodes"],
                    "edges": graph.payload["statistics"]["actual_directed_edges"],
                })
                # Save only the first two frames, like the production save_predictions contract.
                if index < 2:
                    stem = _prediction_artifact_stem(sample["sample_id"], index)
                    save_image(run / "predictions" / (stem + "_gt.png"), sample["target"])
                    factor = 0.8 if mode == "ann" else 0.9
                    save_image(run / "predictions" / (stem + "_pred.png"),
                               sample["target"] * factor)
            with (run / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = make_viewer_report(
                config, [_dataset_sample_identity(dataset, i) for i in range(len(dataset))],
                guard=20_000, inference_mode="ann" if mode == "ann" else "snn",
                snn_dynamics=None if mode == "ann" else "standard_if",
                simulation_steps=None if mode == "ann" else 4,
            )
            report["fixture"] = "SYNTHETIC CPU UI SMOKE TEST; NOT TRAINED MODEL RESULTS"
            (run / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    finally:
        dataset.close()
    return root, config_path


@pytest.fixture
def artifacts(tmp_path):
    return viewer_fixture(tmp_path)


def _snapshot(root: Path) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_viewer_reuses_saved_images_metrics_and_exact_graph_without_writes(artifacts, monkeypatch):
    root, config_path = artifacts
    before = _snapshot(root.parent)

    def forbidden(*args, **kwargs):
        raise AssertionError("Viewer must not initialize CUDA, infer, or write provenance caches")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr("asgcn_unet.engine._evaluation_dataset_provenance", forbidden)
    monkeypatch.setattr("asgcn_unet.model.ASGCNUNet.forward_sample", forbidden)
    monkeypatch.setattr("asgcn_unet.model.ASGCNUNet.forward_batch", forbidden)
    app = ResultViewer(root, configs={"hdr": config_path}, display_edges=3)
    try:
        catalog = app.catalog()
        assert catalog["readonly"] is True
        assert catalog["warnings"] == []
        data = catalog["datasets"][0]
        assert len(data["frames"]) == 2
        assert data["total_frames"] > 2
        frame = app.frame("hdr", 1)
        assert frame["graph_available"] is True
        assert frame["images"][0]["metrics"]["psnr"] == 11.123456789
        assert frame["images"][0]["report_eligible"] is False
        stored = app.datasets["hdr"].runs["ann"].frames[1].prediction.read_bytes()
        assert app.image("hdr", 1, "ann", "prediction") == stored
        graph = app.graph("hdr", 1)
        assert len(graph["nodes"]) == 32
        assert graph["statistics"]["displayed_edges"] <= 3
        assert graph["statistics"]["actual_directed_edges"] > 3
        assert "NOT rehashed" in graph["provenance_note"]
        neighbors = app.neighbors("hdr", 1, 0)
        assert neighbors["degree"] == len(neighbors["neighbors"])
        assert app.raw_target("hdr", 1).startswith(b"\x89PNG")
        assert app.graph("hdr", 1) is graph
        app.graph("hdr", 0)
        assert app._graph_key == ("hdr", 0)
    finally:
        app.close()
    assert app.datasets["hdr"].dataset is None
    assert _snapshot(root.parent) == before


def test_image_only_viewer_does_not_require_dataset_or_checkpoint(artifacts):
    root, _ = artifacts
    app = ResultViewer(root)
    assert app.frame("hdr", 0)["graph_available"] is False
    assert app.image("hdr", 0, "ann", "target").startswith(b"\x89PNG")
    with pytest.raises(ViewerError, match="config"):
        app.graph("hdr", 0)


@pytest.mark.parametrize("field,value", [("model", 0.2), ("transform", "none"),
                                        ("manifest", {"split": "wrong"})])
def test_graph_refuses_changed_configuration_contract(artifacts, field, value):
    root, config_path = artifacts
    if field in {"model", "transform"}:
        config = json.loads(config_path.read_text())
        if field == "model":
            config["model"]["graph_radius"] = value
        else:
            config["dataset"]["tone_map"] = value
        config_path.write_text(json.dumps(config))
    else:
        path = root / "hdr/ann/metrics.json"
        report = json.loads(path.read_text())
        report["evaluation_protocol"]["evaluation_dataset"]["contract"]["manifest"] = value
        path.write_text(json.dumps(report))
    app = ResultViewer(root, configs={"hdr": config_path})
    if field == "manifest":
        assert "ann" not in app.datasets["hdr"].runs
        assert app.warnings
        return
    with pytest.raises(ValueError):
        app.graph("hdr", 0)


def test_viewer_excludes_archives_and_exposes_incomplete_runs(artifacts):
    root, _ = artifacts
    (root / "hdr/ann.failed-example").mkdir()
    (root / "hdr/snn_literal_eq15_T8").mkdir()
    app = ResultViewer(root)
    assert len(app.catalog()["datasets"][0]["modes"]) == 2
    assert len(app.warnings) == 1
    assert "snn_literal_eq15_T8" in app.warnings[0]


def test_csv_png_identity_mismatch_is_not_silently_shown(artifacts):
    root, _ = artifacts
    path = root / "hdr/ann/frames.csv"
    path.write_text(path.read_text().replace("test/", "wrong/"), encoding="utf-8")
    # The exact sample id format belongs to the dataset; change the first CSV row robustly.
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["sample_id"] = "wrong-sample"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    app = ResultViewer(root)
    assert "ann" not in app.datasets["hdr"].runs
    assert any("identity mismatch" in warning for warning in app.warnings)


def test_saved_target_mismatch_refuses_visual_comparison(artifacts):
    root, config_path = artifacts
    app = ResultViewer(root, configs={"hdr": config_path})
    target = app.datasets["hdr"].runs["ann"].frames[0].target
    Image.fromarray(np.zeros((32, 48), dtype=np.uint8)).save(target)
    with pytest.raises(ViewerError, match="ground truths differ"):
        app.frame("hdr", 0)
    with pytest.raises(ViewerError, match="source target"):
        app.graph("hdr", 0)
    app.close()


@pytest.mark.parametrize("method,args", [
    ("frame", ("unknown", 0)), ("frame", ("hdr", -1)), ("frame", ("hdr", 999)),
    ("image", ("hdr", 0, "../../private", "target")),
    ("image", ("hdr", 0, "ann", "../../private")),
])
def test_unknown_selection_does_not_read_arbitrary_paths(artifacts, method, args):
    root, _ = artifacts
    app = ResultViewer(root)
    with pytest.raises(ViewerError):
        getattr(app, method)(*args)


def test_http_requires_private_token_local_host_and_same_origin(artifacts):
    root, _ = artifacts
    app = ResultViewer(root)
    with ViewerHTTPServer(app, port=0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            prefix = urlsplit(server.url).path

            def request(path, headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
                try:
                    connection.request("GET", path, headers=headers or {})
                    response = connection.getresponse()
                    return response.status, dict(response.getheaders()), response.read()
                finally:
                    connection.close()

            assert server.server_address[0] == "127.0.0.1"
            assert request("/api/catalog")[0] == 403
            assert request(prefix + "api/catalog", {"Host": "attacker.invalid"})[0] == 403
            assert request(prefix + "api/catalog", {"Origin": "http://attacker.invalid"})[0] == 403
            status, headers, body = request(prefix + "api/catalog")
            assert status == 200 and json.loads(body)["readonly"] is True
            assert headers["Cache-Control"] == "no-store"
            assert headers["Referrer-Policy"] == "no-referrer"
            assert request(prefix + "../../pyproject.toml")[0] == 404
            assert request("http://[", {"Host": f"127.0.0.1:{server.server_port}"})[0] == 400
            assert request(prefix + "api/frame?dataset=hdr&index=-1")[0] == 400
            assert request(prefix + "api/frame?dataset=hdr&index=0&index=1")[0] == 400
            assert request(prefix + "api/image?dataset=hdr&index=0&mode=ann&kind=target")[0] == 200
        finally:
            server.shutdown()
            thread.join(timeout=5)
            app.close()
