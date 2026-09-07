"""Offline exporter unit tests: tiny synthetic fixtures, never model results."""
from __future__ import annotations

import base64
import builtins
import csv
import hashlib
import io
import json
import re
import struct
import zlib
from pathlib import Path

import pytest

from asgcn_unet.offline_viewer import (
    ExportLimits,
    OfflineViewerError,
    _png,
    _SelectedJSON,
    _stem,
    build_payload,
    empty_payload,
    export_results_html,
    render_payload_html,
)


def png_bytes(value=80, width=3, height=2):
    """Tiny generated grayscale test image, not learned output."""
    def chunk(kind, value):
        return (struct.pack(">I", len(value)) + kind + value
                + struct.pack(">I", zlib.crc32(kind + value)))
    raw = b"".join(b"\0" + bytes([value]) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def fixture(root: Path):
    for dataset, kind in (("aid", "eventaid_r_zip"), ("hdr", "eventhdr")):
        for mode in ("ann", "snn_literal_eq15_T4"):
            run = root / dataset / mode
            predictions = run / "predictions"
            predictions.mkdir(parents=True)
            rows = []
            for index in range(3):
                sample_id = f"synthetic-only/{index}"
                rows.append({"sample_id": sample_id, "scene": "SYNTHETIC TEST ONLY",
                             "psnr": 11.123456789, "ssim": 0.6123456789})
                if index < 2:
                    stem = _stem(sample_id, index)
                    (predictions / f"{stem}_gt.png").write_bytes(png_bytes(80 + index))
                    (predictions / f"{stem}_pred.png").write_bytes(
                        png_bytes((70 if mode == "ann" else 60) + index)
                    )
            with (run / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            report = {
                "fixture": "SYNTHETIC UNIT TEST; NOT TRAINED MODEL RESULTS",
                "dataset": kind, "inference_mode": "ann" if mode == "ann" else "snn",
                "simulation_steps": None if mode == "ann" else 4,
                "snn_dynamics": None if mode == "ann" else "literal_eq15",
                "report_eligible": False,
                "quality": {"frames": 3, "micro": {"psnr": 11.123456789}, "macro": {}},
                "evaluation_protocol": {"schema": "asgcn_reporting_protocol_v1",
                                        "kind": "quality_evaluation",
                                        "protocol_sha256": "not-certified-test-fixture",
                                        "large_ignored_section": [1, 2, 3]},
            }
            (run / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def payload_from_html(path):
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script[^>]*id="offline-data"[^>]*>(.*?)</script>', html, re.DOTALL)
    assert match
    return json.loads(match[1])


def snapshot(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def test_export_all_saved_pngs_without_torch_dataset_or_server_imports(tmp_path, monkeypatch):
    root = fixture(tmp_path / "synthetic-smoke")
    before = snapshot(root)
    original_import = builtins.__import__

    def restricted(name, *args, **kwargs):
        if name.startswith(("torch", "numpy", "PIL", "asgcn_unet.engine", "asgcn_unet.data",
                            "asgcn_unet.graph", "asgcn_unet.model", "http.server")):
            raise AssertionError(f"Forbidden offline dependency: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", restricted)
    output = tmp_path / "offline-synthetic-smoke.html"
    result = export_results_html(root, output)
    assert result["saved_frames"] == 4
    assert result["graph_reconstruction"] is False
    payload = payload_from_html(output)
    assert len(payload["datasets"]) == 2
    for dataset in payload["datasets"]:
        assert dataset["total_frames"] == 3
        assert len(dataset["frames"]) == 2
        assert all(mode["saved_frames"] == 2 for mode in dataset["modes"])
        assert all(mode["protocol"]["verification"] == "stored_metadata_only"
                   for mode in dataset["modes"])
        assert dataset["frames"][0]["graph"] is None
        assert dataset["frames"][0]["images"][0]["metrics"]["psnr"] == 11.123456789
    original_pngs = {hashlib.sha256(path.read_bytes()).hexdigest(): path.read_bytes()
                     for path in root.rglob("*.png")}
    assert set(payload["images"]) == set(original_pngs)
    for digest, image in payload["images"].items():
        assert base64.b64decode(image["data_url"].split(",", 1)[1]) == original_pngs[digest]
    assert snapshot(root) == before


def test_no_output_overwrite_and_missing_data_never_falls_back(tmp_path):
    existing = tmp_path / "existing.html"
    existing.write_text("USER OUTPUT", encoding="utf-8")
    with pytest.raises(OfflineViewerError, match="already exists"):
        export_results_html(tmp_path / "missing", existing)
    assert existing.read_text() == "USER OUTPUT"
    with pytest.raises(OfflineViewerError, match="No completed"):
        export_results_html(tmp_path, tmp_path / "new.html")
    assert not (tmp_path / "new.html").exists()


def test_script_closing_text_and_placeholder_values_roundtrip(tmp_path):
    payload = empty_payload('</script><script>alert("x")</script> & 한글')
    payload["notes"] = ["__OFFLINE_JS__ __OFFLINE_CSS__ __OFFLINE_PAYLOAD__"]
    output = tmp_path / "escape.html"
    render_payload_html(payload, output)
    assert payload_from_html(output) == payload
    assert '<script>alert("x")</script>' not in output.read_text(encoding="utf-8")


def test_output_limit_rejects_without_partial_output_or_subset(tmp_path):
    root = fixture(tmp_path / "synthetic-smoke")
    with pytest.raises(OfflineViewerError, match="max_output_bytes"):
        export_results_html(root, tmp_path / "small.html", limits=ExportLimits(max_output_bytes=50))
    assert not (tmp_path / "small.html").exists()
    assert not list(tmp_path.glob(".asgcn-offline-*"))


@pytest.mark.parametrize("text", [
    '{"keep":NaN}', '{"skip":Infinity}', '{"keep":1e999}', '{"keep":1}junk',
    '{"keep":1,"keep":2}', '{"skip":{"a":1,"a":2}}', '{"keep":1,}',
    '{"skip":[1,]}', '{"skip":"\\q"}', '{"keep":"unfinished}', '[1,2]',
])
def test_stream_parser_rejects_corrupt_json_even_in_skipped_sections(text):
    with pytest.raises((OfflineViewerError, ValueError)):
        _SelectedJSON(io.StringIO(text), 1024).read({"keep": True})


def test_streaming_reader_discards_large_unselected_arrays():
    text = '{"keep":{"micro":1.25},"skip":[' + ','.join('{"a":1}' for _ in range(20000)) + ']}'
    reader = _SelectedJSON(io.StringIO(text), 1024)
    assert reader.read({"keep": {"micro": True}}) == {"keep": {"micro": 1.25}}
    assert reader.retained < 1024


def test_explicit_metadata_input_and_png_limits(tmp_path):
    root = fixture(tmp_path / "synthetic-smoke")
    with pytest.raises(OfflineViewerError, match="max_input_bytes"):
        build_payload(root, limits=ExportLimits(max_input_bytes=10))
    with pytest.raises(OfflineViewerError, match="max_metadata_bytes"):
        build_payload(root, limits=ExportLimits(max_metadata_bytes=20))
    image = next(root.rglob("*.png"))
    with pytest.raises(OfflineViewerError, match="max_png_bytes"):
        _png(image, ExportLimits(max_png_bytes=8))
    with pytest.raises(OfflineViewerError, match="max_decoded_png_bytes"):
        _png(image, ExportLimits(max_decoded_png_bytes=4))


def test_valid_crc_huge_png_header_is_rejected_without_decoding(tmp_path):
    data = bytearray(png_bytes())
    data[16:24] = struct.pack(">II", 100000, 100000)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    image = tmp_path / "huge-header-synthetic.png"
    image.write_bytes(data)
    with pytest.raises(OfflineViewerError, match="max_decoded_png_bytes"):
        _png(image, ExportLimits())


def test_atomic_publish_refuses_racing_existing_output(tmp_path, monkeypatch):
    import os

    output = tmp_path / "racing.html"
    original_link = os.link

    def racing_link(source, destination):
        Path(destination).write_text("OTHER WRITER", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        render_payload_html(empty_payload(), output)
    assert output.read_text() == "OTHER WRITER"
    assert not list(tmp_path.glob(".asgcn-offline-*"))


def test_invalid_png_checksum_is_not_embedded(tmp_path):
    data = bytearray(png_bytes())
    data[-1] ^= 1
    image = tmp_path / "bad-crc-synthetic.png"
    image.write_bytes(data)
    with pytest.raises(OfflineViewerError, match="checksum mismatch"):
        _png(image, ExportLimits())


@pytest.mark.parametrize("mutation,match", [
    ("csv", "Incomplete"), ("identity", "identity mismatch"), ("gt", "GT bytes differ"),
    ("mode", "SNN identity mismatch"), ("bool", "report_eligible"),
])
def test_artifact_mismatches_fail_explicitly(tmp_path, mutation, match):
    root = fixture(tmp_path / "synthetic-smoke")
    run = root / "aid" / "snn_literal_eq15_T4"
    if mutation == "csv":
        path = run / "frames.csv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif mutation == "identity":
        path = run / "frames.csv"
        path.write_text(path.read_text().replace("synthetic-only/0", "wrong/0"))
    elif mutation == "gt":
        next((run / "predictions").glob("*_gt.png")).write_bytes(png_bytes(11))
    else:
        path = run / "metrics.json"
        report = json.loads(path.read_text())
        report["simulation_steps" if mutation == "mode" else "report_eligible"] = 8
        path.write_text(json.dumps(report))
    with pytest.raises(OfflineViewerError, match=match):
        build_payload(root)


def graph_fixture(path):
    graph = {"nodes": [[0.0, 0.1, 0.2, -1], [0.2, 0.3, 0.4, 1]], "edges": [[0, 1]],
             "statistics": {"nodes": 2, "displayed_edges": 1, "actual_directed_edges": 2},
             "radius": 0.5, "position_dims": 3, "metadata": {},
             "provenance_note": "SYNTHETIC UNIT TEST GRAPH ONLY"}
    wrapper = {"schema": "asgcn_offline_graphs_v1", "graphs": [
        {"dataset": "aid", "index": 0, "sample_id": "synthetic-only/0", "graph": graph}
    ]}
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    return wrapper


def test_explicit_saved_graph_keeps_exact_supplied_data_and_limits_neighbor_claims(tmp_path):
    root = fixture(tmp_path / "synthetic-smoke")
    path = tmp_path / "synthetic-graph.json"
    saved = graph_fixture(path)
    payload = build_payload(root, graph_json=path)
    graph = payload["datasets"][0]["frames"][0]["graph"]
    assert graph["nodes"] == saved["graphs"][0]["graph"]["nodes"]
    assert graph["edges"] == [[0, 1]]
    assert graph["offline_neighbor_scope"] == "included_edges_only"
    assert graph["identity_verified"] is False
    assert payload["datasets"][0]["frames"][1]["graph"] is None
    saved["graphs"][0]["sample_id"] = "wrong"
    path.write_text(json.dumps(saved))
    with pytest.raises(OfflineViewerError, match="does not match"):
        build_payload(root, graph_json=path)


def test_png_content_change_after_indexing_refuses_publish(tmp_path):
    root = fixture(tmp_path / "synthetic-smoke")
    payload = build_payload(root)
    image = next(iter(payload["images"].values()))["data_url"]
    image.path.write_bytes(png_bytes(222))
    with pytest.raises(OfflineViewerError, match="PNG changed"):
        render_payload_html(payload, tmp_path / "changed.html")
    assert not (tmp_path / "changed.html").exists()
