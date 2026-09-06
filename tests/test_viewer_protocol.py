"""CPU-only synthetic commitment tests; no learned results or source evaluation."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from asgcn_unet.engine import (
    _canonical_sha256,
    _evaluation_dataset_transform_contract,
    _evaluation_manifest_contract,
    _hashed_contract,
    _public_config,
)
from asgcn_unet.viewer_protocol import (
    ViewerProtocolError,
    prepare_viewer_protocol,
    validate_viewer_protocol,
)


def make_viewer_report(
    config, identities, *, guard=None, inference_mode="ann", simulation_steps=None,
    snn_dynamics=None,
):
    """Synthetic non-reporting fixture with valid internal protocol commitments."""
    if guard is None:
        guard = config["model"]["max_graph_edges"]
    edge_guard = {
        "configured_max_graph_edges": config["model"]["max_graph_edges"],
        "requested_max_graph_edges_override": (
            guard if guard != config["model"]["max_graph_edges"] else None
        ),
        "effective_max_graph_edges": guard,
    }
    execution = {
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps,
        "snn_dynamics": snn_dynamics,
        "graph_edge_guard": edge_guard,
    }
    protocol = {
        "schema": "asgcn_reporting_protocol_v1", "kind": "quality_evaluation",
        "report_eligible": False,
        "report_ineligible_reasons": ["synthetic CPU unit-test fixture, not model results"],
        "public_config": _hashed_contract(_public_config(config)),
        "model_config": _hashed_contract(config["model"]),
        "execution": _hashed_contract(execution),
        "source": _hashed_contract({"fixture": "synthetic-only"}),
        "runtime": _hashed_contract({"device": "cpu", "fixture": "synthetic-only"}),
        "precision": _hashed_contract({"effective": "fp32"}),
        "checkpoint": {"fixture": "no checkpoint was loaded or generated"},
        "evaluation_dataset": _hashed_contract({
            "content": {"fixture": "no real source content is certified"},
            "transform": _evaluation_dataset_transform_contract(config),
            "manifest": _evaluation_manifest_contract(config),
            "coverage": {"fixture": "synthetic-only"},
            "sampling": {"selected_samples": len(identities), "selected": identities},
        }),
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    return {
        "fixture": "SYNTHETIC NON-REPORTING CPU TEST ONLY",
        "dataset": config["dataset"]["type"], "report_eligible": False,
        **copy.deepcopy(execution),
        "quality": {"frames": len(identities)}, "evaluation_protocol": protocol,
    }


def _reseal(report, *fields):
    protocol = report["evaluation_protocol"]
    for field in fields:
        protocol[field] = _hashed_contract(protocol[field]["contract"])
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = _canonical_sha256(protocol)


@pytest.fixture
def fixture():
    config = {
        "dataset": {"type": "eventaid_r_zip", "root": "synthetic-only",
                    "target_offset": 1, "tone_map": "log"},
        "model": {"max_graph_edges": 2000000, "graph_radius": 0.1},
    }
    identities = [
        {"dataset_index": index, "group": "synthetic-only",
         "sequence_id": "synthetic-only/part-001", "part_index": 1,
         "sequence_index": index, "frame_id": index,
         "event_name": f"event_upload/{index:06d}.txt",
         "target_name": f"gt_upload/{index + 1:06d}_img.jpg",
         "t0_us": index * 10000, "t1_us": (index + 1) * 10000}
        for index in range(4)
    ]
    return config, identities, make_viewer_report(config, identities, guard=7475202)


def test_prepared_summary_detaches_only_requested_identities(fixture):
    config, identities, report = fixture
    summary = prepare_viewer_protocol(report, dataset_indices=[1, 3])
    assert set(summary["identities"]) == {1, 3}
    assert "content" not in summary and "evaluation_protocol" not in summary
    result = validate_viewer_protocol(summary, config=config, dataset_index=1,
                                      current_identity=identities[1])
    assert result == {"expected_identity": identities[1], "effective_max_graph_edges": 7475202}
    report["evaluation_protocol"]["model_config"]["contract"]["graph_radius"] = 99
    identities[1]["part_index"] = 99
    assert summary["model"]["graph_radius"] == 0.1
    assert summary["identities"][1]["part_index"] == 1
    result["expected_identity"]["part_index"] = 77
    assert summary["identities"][1]["part_index"] == 1


@pytest.mark.parametrize("field", [
    "public_config", "model_config", "execution", "source", "runtime", "precision",
    "evaluation_dataset",
])
def test_rejects_each_inner_hash_even_with_a_resealed_outer_protocol(fixture, field):
    _, _, report = fixture
    report["evaluation_protocol"][field]["sha256"] = "0" * 64
    _reseal(report)
    with pytest.raises(ViewerProtocolError, match=field + r"\.sha256"):
        prepare_viewer_protocol(report, dataset_indices=[0])


def test_rejects_outer_protocol_hash_mismatch(fixture):
    _, _, report = fixture
    report["evaluation_protocol"]["checkpoint"]["fixture"] = "altered"
    with pytest.raises(ViewerProtocolError, match="protocol_sha256"):
        prepare_viewer_protocol(report, dataset_indices=[0])


@pytest.mark.parametrize("field,value", [("schema", "unknown"), ("kind", "benchmark")])
def test_rejects_other_protocol_types(fixture, field, value):
    _, _, report = fixture
    report["evaluation_protocol"][field] = value
    _reseal(report)
    with pytest.raises(ViewerProtocolError):
        prepare_viewer_protocol(report, dataset_indices=[0])


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", 4])
def test_rejects_invalid_requested_indices(fixture, value):
    _, _, report = fixture
    with pytest.raises(ViewerProtocolError):
        prepare_viewer_protocol(report, dataset_indices=[value])


def test_rejects_duplicate_requested_indices_and_accepts_no_saved_pngs(fixture):
    _, _, report = fixture
    with pytest.raises(ViewerProtocolError, match="Duplicate"):
        prepare_viewer_protocol(report, dataset_indices=[0, 0])
    assert prepare_viewer_protocol(report, dataset_indices=[])["identities"] == {}


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", 2])
def test_rejects_bad_or_reordered_selected_indices(fixture, value):
    _, _, report = fixture
    data = report["evaluation_protocol"]["evaluation_dataset"]["contract"]
    data["sampling"]["selected"][1]["dataset_index"] = value
    _reseal(report, "evaluation_dataset")
    with pytest.raises(ViewerProtocolError, match="dataset"):
        prepare_viewer_protocol(report, dataset_indices=[0])


@pytest.mark.parametrize("field", ["frames", "selected_samples", "selected"])
def test_rejects_sampling_count_mismatch(fixture, field):
    _, _, report = fixture
    sampling = report["evaluation_protocol"]["evaluation_dataset"]["contract"]["sampling"]
    if field == "frames":
        report["quality"]["frames"] = 3
    elif field == "selected_samples":
        sampling[field] = 3
    else:
        sampling[field].pop()
    _reseal(report, "evaluation_dataset")
    with pytest.raises(ViewerProtocolError):
        prepare_viewer_protocol(report, dataset_indices=[0])


def test_rejects_uncommitted_top_level_graph_guard(fixture):
    _, _, report = fixture
    report["graph_edge_guard"]["effective_max_graph_edges"] = 9999999
    with pytest.raises(ViewerProtocolError, match="graph_edge_guard"):
        prepare_viewer_protocol(report, dataset_indices=[0])


@pytest.mark.parametrize("key,value", [
    ("part_index", 2), ("sequence_id", "synthetic-only/part-002"),
    ("event_name", "event_upload/000099.txt"), ("target_name", "gt_upload/000099_img.jpg"),
    ("t0_us", 123), ("sequence_index", True),
])
def test_rejects_changed_current_identity_including_boolean_integer_confusion(fixture, key, value):
    config, identities, report = fixture
    summary = prepare_viewer_protocol(report, dataset_indices=[1])
    current = copy.deepcopy(identities[1])
    current[key] = value
    with pytest.raises(ViewerProtocolError, match="Current source sample identity"):
        validate_viewer_protocol(summary, config=config, dataset_index=1, current_identity=current)


@pytest.mark.parametrize("section,key,value", [
    ("model", "graph_radius", 0.2), ("dataset", "target_offset", 2),
    ("dataset", "tone_map", "none"),
])
def test_rejects_model_or_transform_change(fixture, section, key, value):
    config, _, report = fixture
    summary = prepare_viewer_protocol(report, dataset_indices=[1])
    config[section][key] = value
    with pytest.raises(ViewerProtocolError):
        validate_viewer_protocol(summary, config=config, dataset_index=1)


def test_manifest_content_is_compared_read_only(fixture, tmp_path):
    config, identities, _ = fixture
    manifest = tmp_path / "synthetic-manifest.json"
    manifest.write_text(json.dumps({"fixture": "synthetic-only", "files": []}), encoding="utf-8")
    config["dataset"]["file_manifest"] = str(manifest)
    report = make_viewer_report(config, identities)
    summary = prepare_viewer_protocol(report, dataset_indices=[0])
    before = manifest.read_bytes()
    validate_viewer_protocol(summary, config=config, dataset_index=0)
    assert manifest.read_bytes() == before
    manifest.write_text(json.dumps({"fixture": "changed", "files": []}), encoding="utf-8")
    with pytest.raises(ViewerProtocolError, match="manifest"):
        validate_viewer_protocol(summary, config=config, dataset_index=0)


def test_validation_has_no_cuda_checkpoint_data_or_provenance_writes(fixture, monkeypatch):
    config, identities, report = fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only protocol inspection invoked a forbidden runtime path")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr("asgcn_unet.engine.build_dataset", forbidden)
    monkeypatch.setattr("asgcn_unet.engine._evaluation_dataset_provenance", forbidden)
    monkeypatch.setattr("asgcn_unet.engine._dataset_content_fingerprint", forbidden)
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", forbidden)
    summary = prepare_viewer_protocol(report, dataset_indices=[1])
    validate_viewer_protocol(summary, config=config, dataset_index=1, current_identity=identities[1])
