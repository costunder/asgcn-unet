"""Generate real saved-result PNGs and actual event graphs, without a web server.

Predictions are copied from completed evaluations to preserve recurrent context.
This is source visualization, not a new quality evaluation or checkpoint inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .offline_viewer import (
    ExportLimits,
    OfflineViewerError,
    _BudgetCSVLines,
    _inside,
    _png,
    _PNGData,
    _RetainedMetadataBudget,
    _SelectedJSON,
    _signature,
    build_payload,
    render_payload_html,
)

_MIB = 1024 * 1024


@dataclass(frozen=True)
class _SelectedIndices:
    indices: frozenset[int]


class _IdentityReader(_SelectedJSON):
    """Parse all syntax but retain only the identities with already-saved PNGs."""

    def value(self, selection: Any, depth: int = 0) -> Any:
        if not isinstance(selection, _SelectedIndices):
            return super().value(selection, depth)
        if depth > 64:
            raise OfflineViewerError("Diagnostic JSON nesting exceeds safety guard")
        self._space()
        if self._take() != "[":
            raise OfflineViewerError("Expected selected sample identity array")
        self._space()
        items, count = {}, 0
        if self._peek() == "]":
            self._take()
            return {"count": count, "items": items}
        while True:
            value = super().value(count in selection.indices, depth + 1)
            if count in selection.indices:
                items[count] = value
            count += 1
            self._space()
            delimiter = self._take()
            if delimiter == "]":
                return {"count": count, "items": items}
            if delimiter != ",":
                raise OfflineViewerError("Invalid selected identity array separator")


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise OfflineViewerError(f"{context} differs from saved evaluation")


def _hashed(value: Any, name: str) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
        raise OfflineViewerError(f"Missing saved {name} contract")
    digest = hashlib.sha256(_canonical(value["contract"]).encode("utf-8")).hexdigest()
    if digest != value.get("sha256"):
        raise OfflineViewerError(f"Saved {name} hash mismatch")
    return value["contract"]


def _identities(path: Path, indices: frozenset[int], limits: ExportLimits) -> dict:
    selection = {
        "dataset": True,
        "inference_mode": True,
        "simulation_steps": True,
        "snn_dynamics": True,
        "graph_edge_guard": True,
        "quality": {"frames": True},
        "evaluation_protocol": {
            "schema": True,
            "kind": True,
            "protocol_sha256": True,
            "model_config": True,
            "execution": True,
            "evaluation_dataset": {
                "contract": {
                    "transform": True,
                    "manifest": True,
                    "sampling": {"selected_samples": True, "selected": _SelectedIndices(indices)},
                }
            },
        },
    }
    before = _signature(path)
    if before[2] > limits.max_input_bytes:
        raise OfflineViewerError("Report exceeds explicit input byte guard")
    with path.open(encoding="utf-8") as handle:
        report = _IdentityReader(handle, limits.max_metadata_bytes).read(selection)
    if _signature(path) != before:
        raise OfflineViewerError("Report changed while reading selected source identities")
    protocol = report.get("evaluation_protocol", {})
    if (
        protocol.get("schema") != "asgcn_reporting_protocol_v1"
        or protocol.get("kind") != "quality_evaluation"
    ):
        raise OfflineViewerError("Actual graph export requires a saved quality evaluation protocol")
    report["model_contract"] = _hashed(protocol.get("model_config"), "model_config")
    execution = _hashed(protocol.get("execution"), "execution")
    for key in ("inference_mode", "simulation_steps", "snn_dynamics", "graph_edge_guard"):
        _same(report.get(key), execution.get(key), key)
    data = protocol.get("evaluation_dataset", {}).get("contract", {})
    sampling = data.get("sampling", {})
    selected = sampling.get("selected", {})
    total = report.get("quality", {}).get("frames")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 1
        or selected.get("count") != total
        or sampling.get("selected_samples") != total
    ):
        raise OfflineViewerError("Selected identities do not cover the recorded evaluation frames")
    items = selected.get("items", {})
    if set(items) != set(indices):
        raise OfflineViewerError("Saved prediction has no source identity")
    for index, identity in items.items():
        if (
            not isinstance(identity, dict)
            or type(identity.get("dataset_index")) is not int
            or identity["dataset_index"] != index
            or not isinstance(identity.get("group"), str)
        ):
            raise OfflineViewerError("Malformed selected source identity")
    report["source_identities"] = items
    report["data_contract"] = data
    return report


def _csv_graph_stats(path: Path, indices: frozenset[int], *, memory_budget_bytes: int) -> dict:
    before, result = _signature(path), {}
    budget = _RetainedMetadataBudget(memory_budget_bytes)
    with path.open(encoding="utf-8", newline="") as handle:
        lines = _BudgetCSVLines(handle, budget, ExportLimits())
        lines.begin_record()
        reader = csv.DictReader(lines)
        if not reader.fieldnames or not {"nodes", "edges"}.issubset(reader.fieldnames):
            raise OfflineViewerError("Missing saved graph statistics CSV header")
        index = -1
        while True:
            lines.begin_record()
            try:
                row = next(reader)
            except StopIteration:
                break
            index += 1
            if index not in indices:
                continue
            values = {}
            for name in ("nodes", "edges"):
                raw = row.get(name)
                if not isinstance(raw, str) or not raw.isdecimal():
                    raise OfflineViewerError(f"Missing recorded graph {name} at frame {index}")
                values[name] = int(raw)
            budget.charge(4096, "retaining selected graph statistics")
            result[index] = values
    if _signature(path) != before or set(result) != set(indices):
        raise OfflineViewerError("CSV changed or lacks selected graph statistics")
    return result


def _target_array(target):
    import numpy as np

    array = target.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim != 3 or array.shape[0] not in (1, 3):
        raise OfflineViewerError("Actual target must be CHW with one/three channels")
    array = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
    return (array * 255 + 0.5).astype(np.uint8)


def _save_png(path: Path, array) -> None:
    from PIL import Image

    with path.open("xb") as handle:
        Image.fromarray(array).save(handle, format="PNG")


def _copy_png(item: dict, output: Path) -> None:
    stored = item.get("data_url")
    if not isinstance(stored, _PNGData):
        raise OfflineViewerError("Expected verified saved evaluation PNG")
    if _signature(stored.path) != stored.signature:
        raise OfflineViewerError("Saved PNG changed before copy")
    digest = hashlib.sha256()
    with stored.path.open("rb") as source, output.open("xb") as target:
        while data := source.read(65536):
            digest.update(data)
            target.write(data)
    if digest.hexdigest() != item["sha256"] or _signature(stored.path) != stored.signature:
        raise OfflineViewerError("Saved PNG changed during copy")


def _retained_estimate(value: Any) -> int:
    # Stream the estimate too: do not construct another complete JSON string.
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return 4096 + sum(len(part) for part in encoder.iterencode(value)) * 8


def _encoding_plan(payload: dict, scratch_budget: int) -> int:
    largest = 0
    for item in payload["images"].values():
        data = item["data_url"]
        if not isinstance(data, _PNGData):
            raise OfflineViewerError("Expected verified PNG file before embedding")
        # Raw bytes, base64 bytes/string, JSON escaping and UTF-8 encoder overlap.
        largest = max(largest, data.signature[2] * 16 + _MIB)
    if largest > scratch_budget:
        raise OfflineViewerError("PNG embedding scratch exceeds memory budget; no resize made")
    return largest


def _check_gt(sample: dict, item: dict, scratch_budget: int) -> None:
    import numpy as np
    from PIL import Image

    source = item["data_url"]
    if not isinstance(source, _PNGData):
        raise OfflineViewerError("Saved GT has no verified artifact source")
    if _signature(source.path) != source.signature:
        raise OfflineViewerError("Saved GT changed before decoding")
    h, w = map(int, sample["sensor_size"])
    channels = item["channels"]
    if (item["height"], item["width"]) != (h, w) or tuple(sample["target"].shape) != (
        channels,
        h,
        w,
    ):
        raise OfflineViewerError("Saved GT and actual source target shape differ")
    # Check BEFORE either full-resolution conversion or decoder allocation.
    if h * w * 64 + 8 * _MIB > scratch_budget:
        raise OfflineViewerError("Full-resolution visualization scratch exceeds memory budget")
    with Image.open(source.path) as image:
        if image.size != (w, h) or image.mode != ("L" if channels == 1 else "RGB"):
            raise OfflineViewerError("Saved GT header changed before decoding")
        saved = np.asarray(image)
    if _signature(source.path) != source.signature:
        raise OfflineViewerError("Saved GT changed during decoding")
    if not np.array_equal(_target_array(sample["target"]), saved):
        raise OfflineViewerError("Actual source GT pixels differ from saved evaluation GT")


def _input_snapshot(root: Path, budget: int) -> dict:
    """Capture compact reports, not predictions or dataset-wide source trees."""
    result, used = {}, 0
    for key in ("aid", "hdr"):
        directory = root / key
        if not directory.is_dir():
            continue
        for mode in directory.iterdir():
            if not mode.is_dir() or (mode.name != "ann" and not mode.name.startswith("snn_")):
                continue
            if ".failed-" in mode.name or ".incomplete-" in mode.name:
                continue
            for name in ("metrics.json", "frames.csv", "benchmark.json"):
                path = mode / name
                if path.exists():
                    path = _inside(path, root)
                    used += 1024 + len(str(path)) * 8
                    if used > budget:
                        raise OfflineViewerError("Input snapshot exceeds metadata budget")
                    result[path] = _signature(path)
    return result


def _unchanged(snapshot: dict) -> None:
    for path, signature in snapshot.items():
        if _signature(path) != signature:
            raise OfflineViewerError(f"Input changed during visualization generation: {path}")


def _output_space_plan(
    payload: dict, contracts: dict, limits: ExportLimits, destination: Path, display_edges: int
) -> dict:
    """Conservative disk planning; reject whole export, never discard frames."""
    planned = limits.max_output_bytes + _MIB
    for dataset in payload["datasets"]:
        recorded = contracts[dataset["id"]][0]["stats"]
        for frame in dataset["frames"]:
            ids = [frame["images"][0]["target"]] + [item["prediction"] for item in frame["images"]]
            for key in ids:
                data = payload["images"][key]["data_url"]
                if not isinstance(data, _PNGData):
                    raise OfflineViewerError("Expected original PNG file in disk plan")
                planned += data.signature[2]
            target = payload["images"][ids[0]]
            # Lossless RGB plot encoders plus PNG chunk/container overhead.
            planned += target["width"] * target["height"] * 4 + 960 * 720 * 4 + 65536
            planned += recorded[frame["index"]]["nodes"] * 256 + display_edges * 64 + 65536
    free = shutil.disk_usage(destination.parent).free
    if free < planned:
        raise OfflineViewerError(
            f"Insufficient output disk space: {free:,} bytes free, {planned:,} planned. "
            "No existing result was removed and no frame subset was made."
        )
    return {
        "free_bytes": free,
        "planned_bytes": planned,
        "scope": "all copied PNGs, diagnostic PNG/JSON, maximum allowed HTML and report overhead",
    }


def _visualization_pngs(sample: dict, graph: dict, directory: Path) -> None:
    """Create honest event occupancy and 3-D graph plots; never model predictions."""
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = map(int, sample["sensor_size"])
    nodes = np.asarray(graph["nodes"], dtype=np.float32).reshape(-1, 4)
    positive = np.zeros((height, width), dtype=np.uint32)
    negative = np.zeros_like(positive)
    if len(nodes):
        xs = np.clip((nodes[:, 0] * max(width - 1, 1)).round().astype(np.int64), 0, width - 1)
        ys = np.clip((nodes[:, 1] * max(height - 1, 1)).round().astype(np.int64), 0, height - 1)
        for mask, array in ((nodes[:, 3] > 0, positive), (nodes[:, 3] <= 0, negative)):
            np.add.at(array, (ys[mask], xs[mask]), 1)
    event_image = np.zeros((height, width, 3), dtype=np.uint8)
    event_image[positive > 0] = [90, 220, 190]
    event_image[negative > 0] = [240, 140, 110]
    event_image[(positive > 0) & (negative > 0)] = [245, 235, 190]
    _save_png(directory / "events-xy.png", event_image)
    # Plot canvas size is presentation only; no source/model resolution changes.
    canvas = Image.new("RGB", (960, 720), (12, 17, 25))
    draw = ImageDraw.Draw(canvas)
    positions = []
    for x, y, t, _ in nodes:
        positions.append(
            (
                480 + float(x - 0.5) * 480 + float(t - 0.5) * 220,
                350 + float(y - 0.5) * 420 - float(t - 0.5) * 150,
            )
        )
    for source, target in graph["edges"]:
        draw.line((positions[source], positions[target]), fill=(48, 62, 79), width=1)
    for point, node in zip(positions, nodes):
        x, y = point
        draw.ellipse(
            (x - 2, y - 2, x + 2, y + 2), fill=(90, 220, 190) if node[3] > 0 else (240, 140, 110)
        )
    draw.text(
        (25, 18),
        (
            f"ACTUAL EVENT NODES - encoder={graph['encoder_kind']} / no graph by architecture"
            if graph.get("topology_kind") == "no_graph"
            else "ACTUAL EVENT GRAPH - CPU reconstruction / display-edge subset"
        ),
        fill=(220, 230, 240),
    )
    draw.text(
        (25, 680),
        f"nodes={len(nodes)}  actual edges={graph['statistics']['actual_directed_edges']}"
        f"  drawn edges={len(graph['edges'])}  x/y/t normalized",
        fill=(180, 196, 215),
    )
    with (directory / "graph-xyt.png").open("xb") as handle:
        canvas.save(handle, format="PNG")


def generate_result_visualizations(
    eval_root: str | Path,
    output_dir: str | Path,
    *,
    configs: dict[str, str | Path],
    memory_budget_bytes: int,
    reserve_memory_bytes: int,
    cpu_threads: int,
    display_edges: int = 5000,
    limits: ExportLimits | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """One command creates copied model PNGs, real graph files, and an offline report.

    All already-saved prediction frames are processed. Missing real inputs or
    graph/statistics disagreement fails explicitly, never a graph-free fallback.
    Explicit identity/pointwise architectures visualize their real normalized
    event nodes with zero edges; they do not acquire a graph for display.
    """
    from .diagnostic_resources import preflight

    limits = limits or ExportLimits()
    log = progress or (lambda message: print(message, file=sys.stderr, flush=True))
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise OfflineViewerError(
            f"Output directory already exists; choose a new one: {destination}"
        )
    if not destination.parent.is_dir():
        raise OfflineViewerError("Output parent directory must already exist")
    resource_report = preflight(
        budget_bytes=memory_budget_bytes,
        reserve_bytes=reserve_memory_bytes,
        cpu_threads=cpu_threads,
    )
    chunk_budget = memory_budget_bytes // 4
    source_budget = memory_budget_bytes // 2
    if chunk_budget < 16 * _MIB:
        raise OfflineViewerError(
            "Diagnostic memory budget must leave at least 16 MiB for each working area"
        )
    limits = replace(
        limits,
        max_metadata_bytes=min(limits.max_metadata_bytes, chunk_budget // 32),
        max_png_bytes=min(limits.max_png_bytes, (chunk_budget - _MIB) // 16),
    )
    log(
        f"Resource preflight: RAM headroom={resource_report['headroom_bytes'] / _MIB:.0f} MiB; "
        f"planning budget={memory_budget_bytes / _MIB:.0f} MiB; "
        f"reserve={reserve_memory_bytes / _MIB:.0f} MiB; CPU threads={cpu_threads}. "
        "Snapshot checks are not hard memory isolation."
    )
    root = Path(eval_root).expanduser().resolve(strict=True)
    input_snapshot = _input_snapshot(root, chunk_budget // 16)
    snapshot_estimate = sum(1024 + len(str(path)) * 8 for path in input_snapshot)
    log("Reading completed evaluation PNG/CSV/metrics (no model inference).")
    payload = build_payload(
        root, limits=limits, retained_budget_bytes=chunk_budget - snapshot_estimate
    )
    _unchanged(input_snapshot)
    retained_bytes = snapshot_estimate + payload["export"]["retained_metadata_estimate_bytes"]
    _encoding_plan(payload, chunk_budget)
    if not all(dataset["frames"] for dataset in payload["datasets"]):
        raise OfflineViewerError(
            "A dataset has no saved prediction PNGs; cannot create actual comparisons"
        )
    # Imports below occur only after the initial resource gate.
    import torch

    from .diagnostic_graph import build_diagnostic_graph
    from .diagnostic_sample import read_diagnostic_sample
    from .engine import _evaluation_dataset_transform_contract, _evaluation_manifest_contract
    from .utils import load_json, resolve_experiment_paths

    source_contracts = {}
    configuration = {}
    for dataset in payload["datasets"]:
        key = dataset["id"]
        if key not in configs:
            raise OfflineViewerError(f"Missing actual evaluation config for {key}")
        config_path = Path(configs[key]).expanduser().resolve(strict=True)
        input_snapshot[config_path] = _signature(config_path)
        if config_path.stat().st_size > limits.max_metadata_bytes:
            raise OfflineViewerError("Config exceeds metadata byte guard")
        config = resolve_experiment_paths(load_json(config_path), config_path)
        from .stream_input import reject_streaming_frame_diagnostic

        reject_streaming_frame_diagnostic(config["model"], config["dataset"])
        for name in ("file_manifest", "split_manifest"):
            if (
                config["dataset"].get(name)
                and Path(config["dataset"][name]).stat().st_size > limits.max_metadata_bytes
            ):
                raise OfflineViewerError("Manifest exceeds metadata byte guard")
            if config["dataset"].get(name):
                manifest_path = Path(config["dataset"][name])
                input_snapshot[manifest_path] = _signature(manifest_path)
        retained_bytes += _retained_estimate(config) + 16384
        if retained_bytes > chunk_budget:
            raise OfflineViewerError("Retained configuration metadata exceeds memory budget")
        configuration[key] = config
        indices = frozenset(frame["index"] for frame in dataset["frames"])
        reference = None
        source_contracts[key] = []
        for mode in dataset["modes"]:
            log(f"Checking saved source identities: {key}/{mode['id']}")
            directory = root / key / mode["id"]
            identity_limits = replace(
                limits,
                max_metadata_bytes=min(limits.max_metadata_bytes, (chunk_budget - _MIB) // 8192),
            )
            report = _identities(
                _inside(directory / "metrics.json", root), indices, identity_limits
            )
            _same(report["model_contract"], config["model"], "Model config")
            _same(
                report["data_contract"].get("transform"),
                _evaluation_dataset_transform_contract(config),
                "Dataset transform",
            )
            _same(
                report["data_contract"].get("manifest"),
                _evaluation_manifest_contract(config),
                "Dataset manifest",
            )
            if reference is not None:
                _same(report["source_identities"], reference, "Source identities between modes")
            reference = report["source_identities"]
            contract = {
                "mode": mode["id"],
                "identities": reference,
                "stats": _csv_graph_stats(
                    _inside(directory / "frames.csv", root),
                    indices,
                    memory_budget_bytes=chunk_budget,
                ),
                "guard": report["graph_edge_guard"],
                "protocol_sha256": report["evaluation_protocol"]["protocol_sha256"],
            }
            retained_bytes += _retained_estimate(contract)
            if retained_bytes > chunk_budget:
                raise OfflineViewerError("Retained source identity metadata exceeds memory budget")
            source_contracts[key].append(contract)
            del report
    _unchanged(input_snapshot)
    disk_plan = _output_space_plan(payload, source_contracts, limits, destination, display_edges)
    scratch_estimates = []
    topology_kinds = set()
    before_threads = torch.get_num_threads()
    created = False
    generated_frames = 0
    try:
        preflight(
            budget_bytes=memory_budget_bytes,
            reserve_bytes=reserve_memory_bytes,
            cpu_threads=cpu_threads,
        )
        destination.mkdir(mode=0o700)
        created = True
        torch.set_num_threads(cpu_threads)
        for dataset in payload["datasets"]:
            key, config = dataset["id"], configuration[dataset["id"]]
            (destination / key).mkdir()
            for frame in dataset["frames"]:
                index = frame["index"]
                log(
                    f"Generating actual event graph and PNGs: {key} index={index} {frame['sample_id']}"
                )
                preflight(
                    budget_bytes=memory_budget_bytes,
                    reserve_bytes=reserve_memory_bytes,
                    cpu_threads=cpu_threads,
                )
                identity = source_contracts[key][0]["identities"][index]

                def reserve(amount: int, purpose: str) -> None:
                    if amount > source_budget:
                        raise OfflineViewerError(
                            f"{purpose} estimate exceeds source working-memory budget"
                        )
                    preflight(
                        budget_bytes=memory_budget_bytes,
                        reserve_bytes=reserve_memory_bytes,
                        cpu_threads=cpu_threads,
                    )

                sample = read_diagnostic_sample(
                    config, identity, memory_budget_bytes=source_budget, reserve=reserve
                )
                if sample["sample_id"] != frame["sample_id"]:
                    raise OfflineViewerError(
                        "Current source sample_id differs from saved prediction"
                    )
                target_item = payload["images"][frame["images"][0]["target"]]
                _check_gt(sample, target_item, chunk_budget)
                graph = build_diagnostic_graph(
                    sample,
                    config["model"],
                    memory_budget_bytes=chunk_budget,
                    display_edges=display_edges,
                )
                for contract in source_contracts[key]:
                    recorded = contract["stats"][index]
                    _same(graph["statistics"]["nodes"], recorded["nodes"], "Graph nodes")
                    _same(
                        graph["statistics"]["actual_directed_edges"],
                        recorded["edges"],
                        "Graph edges",
                    )
                topology_kind = graph.get("topology_kind", "radius_graph")
                topology_kinds.add(topology_kind)
                graph["offline_neighbor_scope"] = (
                    "complete_no_graph" if topology_kind == "no_graph" else "included_edges_only"
                )
                graph["source_identity"] = identity
                graph["identity_verified"] = False
                graph["saved_protocol_commitment_verified"] = False
                graph["saved_dataset_commitment_verified"] = False
                graph["source_binding_checks"] = [
                    "model/transform/manifest",
                    "selected source window",
                    "sample_id",
                    "saved GT pixels",
                    "all-mode node/edge counts",
                ]
                graph["provenance_note"] = (
                    (
                        "Actual current normalized event nodes for the explicit "
                        f"{graph['encoder_kind']} no-graph architecture; zero edges are not a "
                        "failure fallback, and no radius connections were computed. "
                        if topology_kind == "no_graph"
                        else "Actual current events, exact CPU radius-graph reconstruction. "
                    )
                    + "Saved model/data "
                    "semantics, selected source, GT pixels and all-mode node/edge counts matched. "
                    "Full source-file hashes, full saved protocol/dataset commitments and historical "
                    "GPU tensor equality were NOT verified. "
                    "All input nodes and full topology statistics retained. "
                    "Predictions are original full-evaluation PNGs, not reset-state reinference."
                )
                retained_bytes += _retained_estimate(graph) + 32768
                if retained_bytes > chunk_budget:
                    raise OfflineViewerError(
                        "Retained graph display metadata exceeds memory budget; no subset made"
                    )
                frame["graph"] = graph
                directory = destination / key / f"{index:08d}"
                directory.mkdir()
                with (directory / "graph.json").open("x", encoding="utf-8") as handle:
                    json.dump(
                        graph, handle, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                    )
                _copy_png(target_item, directory / "gt.png")
                for entry in frame["images"]:
                    _copy_png(
                        payload["images"][entry["prediction"]], directory / (entry["mode"] + ".png")
                    )
                _visualization_pngs(sample, graph, directory)
                # Expose PNG diagnostics in the offline UI as well as explicit files.
                frame["diagnostic_images"] = {}
                for name, filename in (("events", "events-xy.png"), ("graph", "graph-xyt.png")):
                    info = _png(directory / filename, limits)
                    payload["images"].setdefault(info["sha256"], info)
                    frame["diagnostic_images"][name] = info["sha256"]
                generated_frames += 1
                scratch_estimates.append(
                    {
                        "dataset": key,
                        "index": index,
                        "source": sample.get("metadata", {}).get("diagnostic_source_read"),
                        "graph": graph.get("memory_plan"),
                    }
                )
                del sample
        payload["title"] = "실제 복원 결과 · 시공간 그래프"
        payload["notes"] = [
            "실제 평가의 GT·복원 PNG와 동일 프레임의 실제 이벤트 노드·입력 연결 구조입니다.",
            "재학습·재추론 없이 기존 평가의 복원 PNG를 보존했습니다. 모든 저장 PNG 프레임의 입력을 시각화했습니다.",
            "identity/pointwise 비교군은 설계상 그래프가 없습니다(no_graph). 실제 정규화 이벤트 노드만 표시하며, 0개 엣지는 그래프 실패나 축소 결과가 아닙니다.",
            "입력 생성은 현재 원본의 CPU 진단입니다. 과거 GPU 텐서의 bitwise 일치나 전체 데이터 파일 hash를 재검증하지 않았습니다.",
            "저장된 모델·실행 설정 hash는 확인했으나, 전체 평가 protocol·dataset 계약의 hash는 재검증하지 않았습니다.",
            "PNG로 저장되지 않은 전체 평가 프레임을 추가 추론하지 않습니다. 그래프 생성 범위는 모든 저장 PNG 프레임입니다.",
            "이웃 조회는 HTML/JSON에 포함된 표시용 엣지만 대상으로 합니다. 전체 엣지 수는 정확히 계산한 통계입니다.",
            "품질 수치는 저장된 float 평가값입니다. 이 시각화 생성 자체는 새로운 품질 평가가 아닙니다.",
        ]
        _unchanged(input_snapshot)
        embedding_scratch = _encoding_plan(payload, chunk_budget)
        preflight(
            budget_bytes=memory_budget_bytes,
            reserve_bytes=reserve_memory_bytes,
            cpu_threads=cpu_threads,
        )
        payload["export"].update(
            {
                "graph_reconstruction": "radius_graph" in topology_kinds,
                "input_topology_visualization": True,
                "topology_kinds": sorted(topology_kinds),
                "model_inference": False,
                "scope": "all_saved_prediction_frames",
                "retained_metadata_estimate_bytes": retained_bytes,
            }
        )
        summary = render_payload_html(payload, destination / "index.html", limits=limits)
        summary.update(
            {
                "schema": "asgcn_result_visualizations_v1",
                "complete": True,
                "graph_reconstruction": "radius_graph" in topology_kinds,
                "input_topology_visualization": True,
                "topology_kinds": sorted(topology_kinds),
                "model_inference": False,
                "report_eligible": False,
                "scope": "all_saved_prediction_frames",
                "generated_frames": generated_frames,
                "resources": resource_report,
                "working_memory_budget_bytes": chunk_budget,
                "scratch_estimates": scratch_estimates,
                "disk_plan": disk_plan,
                "retained_metadata_estimate_bytes": retained_bytes,
                "embedding_scratch_estimate_bytes": embedding_scratch,
                "input_root": str(root),
                "output_dir": str(destination),
            }
        )
        with (destination / "generation.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        log("Completed actual PNG + graph generation: " + str(destination / "index.html"))
        return summary
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        IndexError,
        MemoryError,
    ) as error:
        if created:
            failure = {
                "complete": False,
                "generated_frames": generated_frames,
                "error": str(error),
                "original_inputs_changed": False,
            }
            try:
                with (destination / "generation.failed.json").open("x", encoding="utf-8") as handle:
                    json.dump(failure, handle, ensure_ascii=False, indent=2)
            except OSError as write_error:
                log(f"Could not record failure note: {write_error}")
        raise
    finally:
        torch.set_num_threads(before_threads)


def unique_output_directory(eval_root: str | Path) -> Path:
    root = Path(eval_root).expanduser().resolve(strict=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root.parent / f"visualization-{stamp}-{uuid.uuid4().hex[:8]}"
