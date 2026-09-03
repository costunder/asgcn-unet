from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import asgcn_unet.topology_scan as scan_module
from asgcn_unet.cli import build_parser


class _Dataset:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.closed = False

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "events": torch.zeros((index + 1, 4)),
            "target": torch.zeros((1, 2, 2)),
            "sensor_size": (2, 2),
            "sample_id": f"sample-{index}",
            "metadata": {"scene": "scene", "sequence_index": index},
        }

    def close(self) -> None:
        self.closed = True


def _config() -> dict[str, Any]:
    return {
        "seed": 2026,
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": "unused",
            "target_normalization": {"mode": "integer_dtype_max"},
        },
        "model": {
            "event_sampling_factor": 1,
            "graph_radius": 0.08,
            "graph_position_dims": 3,
            "graph_chunk_size": 512,
            "max_graph_edges": 2_000_000,
        },
    }


def _record(index: int, edges: int) -> dict[str, Any]:
    nodes = index + 1
    return {
        "dataset_index": index,
        "sample_id": f"sample-{index}",
        "scene": "scene",
        "sequence_index": index,
        "raw_events": nodes,
        "cropped_events": nodes,
        "retained_events": nodes,
        "model_sampled_events": nodes,
        "candidate_directed_edges": edges,
        "actual_directed_edges": edges,
        "directed_edge_density": 0.0,
        "max_degree": 0,
        "isolated_nodes": nodes,
        "isolate_ratio": 1.0,
    }


@pytest.fixture
def isolated_scan(monkeypatch: pytest.MonkeyPatch):
    datasets: list[_Dataset] = []

    def build(*args, **kwargs):
        dataset = _Dataset()
        datasets.append(dataset)
        return dataset

    monkeypatch.setattr(scan_module, "build_dataset", build)
    monkeypatch.setattr(scan_module, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(scan_module, "_dataset_content_fingerprint", lambda dataset, cache: {})
    monkeypatch.setattr(scan_module, "_dataset_index_contract", lambda dataset: {"samples": 5})
    monkeypatch.setattr(scan_module, "_evaluation_dataset_transform_contract", lambda config: {})
    monkeypatch.setattr(scan_module, "_evaluation_manifest_contract", lambda config: {})
    monkeypatch.setattr(scan_module, "_dataset_coverage_summary", lambda dataset, config: {})
    monkeypatch.setattr(scan_module, "_topology_contract", lambda config, device: {"v": 1})
    return datasets


def test_cli_accepts_tail_scan_and_resume_options() -> None:
    args = build_parser().parse_args(
        [
            "scan-eval-topology",
            "--config", "configs/aid-fast.json",
            "--output", "runs/aid-topology.json",
            "--start-index", "32843",
            "--known-prefix-max-edges", "4000000",
            "--resume",
        ]
    )
    assert args.start_index == 32_843
    assert args.known_prefix_max_edges == 4_000_000
    assert args.resume is True


def test_tail_scan_combines_exact_max_with_known_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_scan
) -> None:
    edges = {2: 3_000_000, 3: 5_250_000, 4: 4_500_000}
    monkeypatch.setattr(
        scan_module,
        "_sample_record",
        lambda sample, model, index: _record(index, edges[index]),
    )
    output = tmp_path / "topology.json"
    result = scan_module.scan_evaluation_topology(
        _config(), output, start_index=2, known_prefix_max_edges=4_000_000
    )
    assert result["tail_max_sample"]["dataset_index"] == 3
    assert result["global_max_sample"]["dataset_index"] == 3
    assert result["global_max_actual_directed_edges"] == 5_250_000
    assert result["global_max_is_exact"] is True
    assert result["global_edge_guard_upper_bound"] == 5_250_000
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert isolated_scan[-1].closed is True


def test_tail_below_prefix_bound_reports_only_a_safe_global_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_scan
) -> None:
    monkeypatch.setattr(
        scan_module,
        "_sample_record",
        lambda sample, model, index: _record(index, 3_000_000 + index),
    )
    result = scan_module.scan_evaluation_topology(
        _config(), tmp_path / "bounded.json",
        start_index=2, known_prefix_max_edges=4_000_000,
    )
    assert result["global_max_is_exact"] is False
    assert result["global_max_sample"] is None
    assert result["global_max_actual_directed_edges"] is None
    assert result["global_edge_guard_upper_bound"] == 4_000_000


def test_interrupted_scan_resumes_only_with_identical_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_scan
) -> None:
    output = tmp_path / "resume.json"
    calls: list[int] = []

    def interrupt(sample, model, index):
        calls.append(index)
        if index == 3:
            raise KeyboardInterrupt
        return _record(index, index * 100)

    monkeypatch.setattr(scan_module, "_sample_record", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scan_module.scan_evaluation_topology(
            _config(), output, start_index=2, known_prefix_max_edges=200
        )
    assert not output.exists()
    assert calls == [2, 3]

    monkeypatch.setattr(
        scan_module,
        "_sample_record",
        lambda sample, model, index: _record(index, index * 100),
    )
    result = scan_module.scan_evaluation_topology(
        _config(), output, start_index=2, known_prefix_max_edges=200, resume=True
    )
    assert result["scanned_samples"] == 3
    assert result["tail_max_sample"]["dataset_index"] == 4

    with pytest.raises(FileExistsError, match="output already exists"):
        scan_module.scan_evaluation_topology(
            _config(), output, start_index=2, known_prefix_max_edges=200, resume=True
        )


@pytest.mark.parametrize(
    ("start", "bound", "message"),
    [
        (1, None, "requires known_prefix"),
        (0, 4_000_000, "only with a nonzero"),
    ],
)
def test_prefix_bound_contract_is_explicit(
    tmp_path: Path, start: int, bound: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scan_module.scan_evaluation_topology(
            _config(), tmp_path / "invalid.json",
            start_index=start, known_prefix_max_edges=bound,
        )
