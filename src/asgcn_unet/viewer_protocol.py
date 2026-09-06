"""Read-only commitments for saved-result visualization, not a new evaluation.

Prepare once while the original report is available, then retain only the small
contracts and explicitly requested saved-frame identities. No checkpoint, CUDA,
dataset construction, source-content hashing, or provenance-cache writer is used.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

_HASHED_FIELDS = (
    "public_config", "model_config", "execution", "source", "runtime", "precision",
    "evaluation_dataset",
)
_SUMMARY_SCHEMA = "asgcn_viewer_protocol_summary_v1"


class ViewerProtocolError(ValueError):
    """A saved commitment or requested source identity is inconsistent."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ViewerProtocolError("Viewer contract is not canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ViewerProtocolError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ViewerProtocolError(f"{name} must be an integer >= {minimum}")
    return value


def _same(name: str, actual: Any, expected: Any) -> None:
    # JSON equality also distinguishes bool/int and int/float identities, unlike
    # Python dict equality; saved JSON and the current loader use the same types.
    if _canonical(actual) != _canonical(expected):
        raise ViewerProtocolError(f"{name} differs from the saved evaluation")


def _hashed(name: str, value: Any) -> dict[str, Any]:
    value = _mapping(value, name)
    contract = _mapping(value.get("contract"), f"{name}.contract")
    if value.get("sha256") != _sha256(contract):
        raise ViewerProtocolError(f"{name}.sha256 mismatch")
    return contract


def prepare_viewer_protocol(
    report: dict[str, Any], *, dataset_indices: list[int],
) -> dict[str, Any]:
    """Validate report commitments once and detach a bounded saved-frame summary.

    Internal hashes detect inconsistent/corrupt metadata; they do not authenticate
    the author or prove that current source files equal their historical contents.
    """
    report = _mapping(report, "report")
    protocol = _mapping(report.get("evaluation_protocol"), "evaluation_protocol")
    if protocol.get("schema") != "asgcn_reporting_protocol_v1":
        raise ViewerProtocolError("Unsupported evaluation protocol schema")
    if protocol.get("kind") != "quality_evaluation":
        raise ViewerProtocolError("Viewer requires quality-evaluation artifacts")
    committed = dict(protocol)
    recorded_hash = committed.pop("protocol_sha256", None)
    if recorded_hash != _sha256(committed):
        raise ViewerProtocolError("evaluation_protocol.protocol_sha256 mismatch")
    contracts = {name: _hashed(name, protocol.get(name)) for name in _HASHED_FIELDS}
    execution = contracts["execution"]
    for name in ("inference_mode", "simulation_steps", "snn_dynamics", "graph_edge_guard"):
        _same(name, report.get(name), execution.get(name))
    _same("report_eligible", report.get("report_eligible"), protocol.get("report_eligible"))
    guard_contract = _mapping(execution.get("graph_edge_guard"), "graph_edge_guard")
    guard = _integer(guard_contract.get("effective_max_graph_edges"),
                     "effective_max_graph_edges", 1)
    data = contracts["evaluation_dataset"]
    transform = _mapping(data.get("transform"), "evaluation_dataset.transform")
    manifest = _mapping(data.get("manifest"), "evaluation_dataset.manifest")
    _same("dataset type", report.get("dataset"), transform.get("type"))
    total = _integer(_mapping(report.get("quality"), "quality").get("frames"),
                     "quality.frames", 1)
    sampling = _mapping(data.get("sampling"), "evaluation_dataset.sampling")
    if _integer(sampling.get("selected_samples"), "sampling.selected_samples", 1) != total:
        raise ViewerProtocolError("Sampling count differs from quality.frames")
    selected = sampling.get("selected")
    if not isinstance(selected, list) or len(selected) != total:
        raise ViewerProtocolError("Sampling identities differ from quality.frames")
    if not isinstance(dataset_indices, list):
        raise ViewerProtocolError("dataset_indices must be a list")
    wanted = set()
    for index in dataset_indices:
        _integer(index, "dataset index")
        if index >= total or index in wanted:
            raise ViewerProtocolError("Duplicate or out-of-range requested dataset index")
        wanted.add(index)
    identities = {}
    for position, item in enumerate(selected):
        identity = _mapping(item, "selected sample identity")
        index = _integer(identity.get("dataset_index"), "selected dataset_index")
        if index != position:
            raise ViewerProtocolError("Selected identities are not in original dataset-index order")
        if not isinstance(identity.get("group"), str) or not identity["group"]:
            raise ViewerProtocolError("Selected sample identity has no group")
        if index in wanted:
            identities[index] = copy.deepcopy(identity)
    return {
        "schema": _SUMMARY_SCHEMA,
        "protocol_sha256": recorded_hash,
        "model": copy.deepcopy(contracts["model_config"]),
        "transform": copy.deepcopy(transform),
        "manifest": copy.deepcopy(manifest),
        "effective_max_graph_edges": guard,
        "identities": identities,
    }


def validate_viewer_protocol(
    summary: dict[str, Any], *, config: dict[str, Any], dataset_index: int,
    current_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match a prepared summary against the current config and optional source.

    Manifest helpers only read declared JSON manifests. Source identity comparison
    is full-dictionary equality, including sequence/part/member/timestamp fields.
    """
    from .engine import (
        _evaluation_dataset_transform_contract,
        _evaluation_manifest_contract,
    )

    summary = _mapping(summary, "viewer protocol summary")
    if summary.get("schema") != _SUMMARY_SCHEMA:
        raise ViewerProtocolError("Expected a prepared viewer protocol summary")
    config = _mapping(config, "config")
    _mapping(config.get("dataset"), "config.dataset")
    model = _mapping(config.get("model"), "config.model")
    _same("Config model", model, summary.get("model"))
    _same("Dataset transform", _evaluation_dataset_transform_contract(config),
          summary.get("transform"))
    _same("Dataset manifest", _evaluation_manifest_contract(config), summary.get("manifest"))
    _integer(dataset_index, "dataset index")
    identities = _mapping(summary.get("identities"), "summary.identities")
    if dataset_index not in identities:
        raise ViewerProtocolError("Dataset index has no prepared saved-frame identity")
    identity = _mapping(identities[dataset_index], "saved sample identity")
    if current_identity is not None:
        _mapping(current_identity, "current sample identity")
        _same("Current source sample identity", current_identity, identity)
    return {
        "expected_identity": copy.deepcopy(identity),
        "effective_max_graph_edges": _integer(summary.get("effective_max_graph_edges"),
                                                "effective_max_graph_edges", 1),
    }
