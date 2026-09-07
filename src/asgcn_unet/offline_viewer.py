"""Self-contained HTML export: stdlib only, no engine/GPU/server/graph construction.

Large report sections are streamed past. PNGs are preserved byte-for-byte and
encoded individually while writing, not collected as Base64 strings in RAM.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import re
import struct
import tempfile
import zlib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, TextIO

_MIB = 1024 * 1024
_MODE = re.compile(r"ann|snn_(?:literal_eq15|standard_if)_T[1-9][0-9]*")
_STEM = re.compile(r"(?P<index>[0-9]{8,})_.+_[0-9a-f]{12}_pred\.png")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_DATASETS = {"aid": ("EventAid-R", "eventaid_r_zip"), "hdr": ("EventHDR", "eventhdr")}
_METRICS = ("psnr", "ssim", "rmse", "temporal_l1")
_REPORT_SELECTION = {
    "dataset": True, "inference_mode": True, "simulation_steps": True,
    "snn_dynamics": True, "report_eligible": True, "report_ineligible_reasons": True,
    "quality": {"frames": True, "micro": True, "macro": True},
    "evaluation_protocol": {"schema": True, "kind": True, "protocol_sha256": True},
}
_BENCH_SELECTION = {
    "mean_ms": True, "fps": True, "peak_gpu_memory_mb": True,
    "report_eligible": True, "io_excluded": True, "inference_mode": True,
    "simulation_steps": True, "snn_dynamics": True,
}


class OfflineViewerError(ValueError):
    """Explicit artifact or resource failure; never fallback data."""


@dataclass(frozen=True)
class ExportLimits:
    """Byte guards reject the whole export; they never subset model/data."""
    max_input_bytes: int = 2048 * _MIB
    max_output_bytes: int = 256 * _MIB
    max_metadata_bytes: int = 8 * _MIB
    max_graph_bytes: int = 16 * _MIB
    max_png_bytes: int = 32 * _MIB
    max_decoded_png_bytes: int = 128 * _MIB

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise OfflineViewerError(f"{key} must be a positive integer")


def _metadata_cost(value: Any) -> int:
    """Conservative Python-container estimate without JSON-copy allocation."""
    if isinstance(value, dict):
        return 1024 + sum(256 + _metadata_cost(key) + _metadata_cost(item)
                          for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 512 + sum(64 + _metadata_cost(item) for item in value)
    if isinstance(value, str):
        return 256 + len(value) * 4
    if value is None or isinstance(value, (bool, int, float)):
        return 128
    raise OfflineViewerError(f"Unsupported metadata estimate type: {type(value).__name__}")


class _RetainedMetadataBudget:
    """Charge before collection growth; estimates are not an OS memory limit.

    Temporary candidates and parsed report containers are conservatively never
    refunded. Consequently the estimate may overstate final payload storage.
    """

    def __init__(self, limit: int | None):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                  or limit < 1):
            raise OfflineViewerError("retained_budget_bytes must be a positive integer or None")
        self.limit, self.retained, self.peak = limit, 0, 0

    def scratch(self, amount: int, stage: str) -> None:
        planned = self.retained + amount
        if self.limit is not None and planned > self.limit:
            raise OfflineViewerError(
                f"Retained metadata budget exceeded before {stage}: "
                f"{planned:,} estimated bytes > {self.limit:,} bytes; no subset made"
            )
        self.peak = max(self.peak, planned)

    def charge(self, amount: int, stage: str) -> None:
        self.scratch(amount, stage)
        self.retained += amount

    def read_selected(self, path: Path, selection: dict, limits: ExportLimits) -> dict:
        parsing_limits = limits
        if self.limit is not None:
            # The existing strict parser retains selected containers as well
            # as duplicate-key sets at up to 64 nesting levels. String parsing
            # temporarily holds character lists. Reserve 128 bytes per parser
            # accounting unit per level (8192 total), plus buffered-text space.
            available = self.limit - self.retained - _MIB
            units = min(limits.max_metadata_bytes, available // 8192)
            if units < 1:
                self.scratch(_MIB + 8192, "JSON parser buffers")
            self.scratch(_MIB + units * 8192, "JSON parser transient allocations")
            parsing_limits = replace(limits, max_metadata_bytes=units)
        try:
            result = _read_selected(path, selection, parsing_limits)
        except OfflineViewerError as error:
            if self.limit is not None and "max_metadata_bytes" in str(error):
                raise OfflineViewerError(
                    "Remaining metadata budget cannot accommodate the required JSON parser "
                    f"state (parser accounting allowance={parsing_limits.max_metadata_bytes:,}); "
                    "no required field was omitted and no subset made"
                ) from error
            raise
        self.charge(_metadata_cost(result), "retaining selected report metadata")
        return result


class _BudgetCSVLines:
    """Bound the complete logical CSV record, including quoted newlines.

    Never changes csv.field_size_limit or another process-wide setting.
    The stdlib's existing field limit remains an additional independent guard.
    """

    def __init__(self, handle: TextIO, budget: _RetainedMetadataBudget, limits: ExportLimits):
        self.handle, self.budget, self.limits = handle, budget, limits
        self.count, self.limit = 0, 0

    def begin_record(self) -> None:
        available = self.budget.limit - self.budget.retained - _MIB
        self.limit = min(self.limits.max_metadata_bytes, available // 128)
        self.count = 0
        if self.limit < 1:
            self.budget.scratch(_MIB + 128, "CSV record buffers")
        self.budget.scratch(_MIB + self.limit * 128, "CSV record transient allocations")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.handle.readline(self.limit - self.count + 1)
        if not line:
            raise StopIteration
        self.count += len(line)
        if self.count > self.limit:
            raise OfflineViewerError(
                "CSV logical record exceeds remaining metadata budget before full parsing; "
                "no subset made"
            )
        return line


def _iter_artifact_directory(path: Path, *, predictions: bool = False):
    """Stream directory entries; Path.glob/iterdir may materialize scandir lists."""
    if predictions and not path.exists():
        return
    with os.scandir(path) as entries:
        for entry in entries:
            if predictions and not entry.name.endswith("_pred.png"):
                continue
            yield Path(entry.path)


class _SelectedJSON:
    """Strict selective JSON reader with a 64-KiB input buffer.

    Large unselected arrays, including all dataset sampling identities, are
    parsed without constructing lists/dictionaries for their contents.
    """
    def __init__(self, handle: TextIO, retained_limit: int):
        self.handle, self.buffer, self.position = handle, "", 0
        self.retained_limit, self.retained = retained_limit, 0

    def _peek(self) -> str:
        if self.position == len(self.buffer):
            self.buffer, self.position = self.handle.read(65536), 0
        return self.buffer[self.position:self.position + 1]

    def _take(self) -> str:
        char = self._peek()
        if char:
            self.position += 1
        return char

    def _space(self) -> None:
        while self._peek() and self._peek() in " \t\r\n":
            self.position += 1

    def _charge(self, size: int) -> None:
        self.retained += size
        if self.retained > self.retained_limit:
            raise OfflineViewerError("Compact metadata exceeded max_metadata_bytes; no subset made")

    def _string(self, keep: bool) -> str | None:
        if self._take() != '"':
            raise OfflineViewerError("Expected a JSON string")
        chars = ['"'] if keep else None
        length = 0
        while True:
            char = self._take()
            if not char or ord(char) < 32:
                raise OfflineViewerError("Unterminated/invalid JSON string")
            if chars is not None:
                chars.append(char)
            length += 1
            if keep and length > self.retained_limit:
                raise OfflineViewerError("JSON scalar exceeded max_metadata_bytes")
            if char == '"':
                return json.loads("".join(chars)) if chars is not None else None
            if char == "\\":
                escaped = self._take()
                if not escaped or escaped not in '"\\/bfnrtu':
                    raise OfflineViewerError("Invalid JSON escape")
                if chars is not None:
                    chars.append(escaped)
                if escaped == "u":
                    digits = "".join(self._take() for _ in range(4))
                    if len(digits) != 4 or any(c not in "0123456789abcdefABCDEF" for c in digits):
                        raise OfflineViewerError("Invalid JSON unicode escape")
                    if chars is not None:
                        chars.append(digits)

    def value(self, selection: bool | dict, depth: int = 0) -> Any:
        if depth > 64:
            raise OfflineViewerError("JSON nesting exceeds explicit safety guard (64)")
        self._space()
        char, keep = self._peek(), selection is not False
        if char == "{":
            self._take()
            result, seen, key_bytes = ({} if keep else None), set(), 0
            self._space()
            if self._peek() == "}":
                self._take()
                return result
            while True:
                self._space()
                key = self._string(True)
                if key in seen:
                    raise OfflineViewerError(f"Duplicate JSON key: {key!r}")
                seen.add(key)
                key_bytes += len(key) + 32
                if key_bytes > self.retained_limit:
                    raise OfflineViewerError("JSON object keys exceed max_metadata_bytes")
                self._space()
                if self._take() != ":":
                    raise OfflineViewerError("Expected JSON colon")
                child = selection.get(key, False) if isinstance(selection, dict) else selection
                value = self.value(child, depth + 1)
                if child is not False:
                    self._charge(len(key) + 32)
                    result[key] = value
                self._space()
                delimiter = self._take()
                if delimiter == "}":
                    return result
                if delimiter != ",":
                    raise OfflineViewerError("Expected JSON object separator")
        if char == "[":
            self._take()
            result = [] if keep else None
            self._space()
            if self._peek() == "]":
                self._take()
                return result
            while True:
                value = self.value(selection, depth + 1)
                if keep:
                    self._charge(16)
                    result.append(value)
                self._space()
                delimiter = self._take()
                if delimiter == "]":
                    return result
                if delimiter != ",":
                    raise OfflineViewerError("Expected JSON array separator")
        if char == '"':
            result = self._string(keep)
            if keep:
                self._charge(len(result.encode("utf-8")) + 32)
            return result
        token = []
        while self._peek() and self._peek() not in " \t\r\n,]}":
            token.append(self._take())
            if len(token) > 128:
                raise OfflineViewerError("JSON numeric/literal token exceeds safety guard (128)")
        text = "".join(token)
        if text in {"null", "true", "false"}:
            value = {"null": None, "true": True, "false": False}[text]
        elif _NUMBER.fullmatch(text):
            value = json.loads(text)
            if isinstance(value, float) and not math.isfinite(value):
                raise OfflineViewerError("Non-finite JSON number")
        else:
            raise OfflineViewerError(f"Invalid JSON value: {text[:32]!r}")
        if keep:
            self._charge(32)
        return value if keep else None

    def read(self, selection: dict) -> dict:
        self._space()
        if self._peek() != "{":
            raise OfflineViewerError("Expected a JSON object")
        result = self.value(selection)
        self._space()
        if self._peek():
            raise OfflineViewerError("Trailing data after JSON object")
        return result


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise OfflineViewerError(f"Artifact is not a regular file inside selected root: {path}")
    return resolved


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read_selected(path: Path, selection: dict, limits: ExportLimits) -> dict:
    before = _signature(path)
    if before[2] > limits.max_input_bytes:
        raise OfflineViewerError(f"{path.name} exceeds max_input_bytes; no subset made")
    with path.open(encoding="utf-8") as handle:
        result = _SelectedJSON(handle, limits.max_metadata_bytes).read(selection)
    if _signature(path) != before:
        raise OfflineViewerError(f"Artifact changed while being read: {path}")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OfflineViewerError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise OfflineViewerError("Boolean numeric metric")
    result = float(value)
    if not math.isfinite(result):
        raise OfflineViewerError("Non-finite metric")
    return result


def _eligible(value: Any) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise OfflineViewerError("report_eligible must be a stored boolean or null")
    return value


def _stem(sample_id: str, index: int) -> str:
    # Same persisted identity contract as engine._prediction_artifact_stem, but
    # without importing engine/torch/the model just to display stored images.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._-")[:64] or "sample"
    return f"{index:08d}_{slug}_{hashlib.sha256(sample_id.encode('utf-8')).hexdigest()[:12]}"


@dataclass(frozen=True)
class _PNGData:
    path: Path
    signature: tuple[int, ...]
    sha256: str

    def data_url(self) -> str:
        if _signature(self.path) != self.signature:
            raise OfflineViewerError(f"PNG changed during export: {self.path}")
        data = self.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise OfflineViewerError(f"PNG content changed during export: {self.path}")
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _png(path: Path, limits: ExportLimits) -> dict:
    signature = _signature(path)
    if signature[2] > limits.max_png_bytes:
        raise OfflineViewerError(f"PNG exceeds max_png_bytes: {path}; no resize performed")
    digest = hashlib.sha256()
    width = height = None
    seen_data = False
    with path.open("rb") as handle:
        magic = handle.read(8)
        digest.update(magic)
        if magic != b"\x89PNG\r\n\x1a\n":
            raise OfflineViewerError(f"Not a PNG: {path}")
        first = True
        while True:
            header = handle.read(8)
            if len(header) != 8:
                raise OfflineViewerError(f"Truncated PNG: {path}")
            length, kind = struct.unpack(">I4s", header)
            if length > signature[2] or (first and (kind != b"IHDR" or length != 13)):
                raise OfflineViewerError(f"Invalid PNG chunk: {path}")
            digest.update(header)
            checksum, remaining, ihdr = zlib.crc32(kind), length, bytearray()
            while remaining:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    raise OfflineViewerError(f"Truncated PNG data: {path}")
                if kind == b"IHDR":
                    ihdr.extend(chunk)
                digest.update(chunk)
                checksum = zlib.crc32(chunk, checksum)
                remaining -= len(chunk)
            crc = handle.read(4)
            if len(crc) != 4 or struct.unpack(">I", crc)[0] != checksum:
                raise OfflineViewerError(f"PNG checksum mismatch: {path}")
            digest.update(crc)
            if first:
                width, height, depth, color, compression, filtering, interlace = struct.unpack(
                    ">IIBBBBB", ihdr
                )
                if not width or not height or depth != 8 or color not in {0, 2}:
                    raise OfflineViewerError(f"Expected saved evaluation 8-bit L/RGB PNG: {path}")
                if width * height * 4 > limits.max_decoded_png_bytes:
                    raise OfflineViewerError(
                        f"PNG decoded RGBA exceeds max_decoded_png_bytes: {path}; no resize made"
                    )
                if compression or filtering or interlace not in {0, 1}:
                    raise OfflineViewerError(f"Invalid PNG encoding: {path}")
                first = False
            elif kind == b"IHDR":
                raise OfflineViewerError(f"Duplicate PNG header: {path}")
            if kind == b"IDAT":
                seen_data = True
            if kind == b"IEND":
                if length or not seen_data or handle.read(1):
                    raise OfflineViewerError(f"Invalid PNG end: {path}")
                break
    if _signature(path) != signature:
        raise OfflineViewerError(f"PNG changed while being read: {path}")
    sha = digest.hexdigest()
    return {"mime": "image/png", "width": width, "height": height,
            "channels": 1 if color == 0 else 3, "sha256": sha,
            "data_url": _PNGData(path, signature, sha)}


def empty_payload(title: str = "ASGCN-U-Net offline results") -> dict:
    """Explicit data-absent template; never a failed-export fallback."""
    return {
        "schema": "asgcn_offline_results_v1", "title": title,
        "created_utc": datetime.now(timezone.utc).isoformat(), "readonly": True,
        "datasets": [], "images": {}, "warnings": [],
        "notes": [
            "Offline file: no SSH, web server, GPU, model inference, or graph reconstruction.",
            "Only existing saved PNGs are embedded. Missing predictions are not generated.",
            "GT is saved normalized/tone-mapped 8-bit evaluation data, not raw HDR radiance.",
            "Metrics are saved float results, not recalculated from these PNGs.",
            ("Stored report_eligible flags are provenance metadata, not a quality guarantee. "
             "Original protocol hashes and source data are not revalidated by this offline export."),
            ("Benchmark ms/FPS/VRAM are not full-dataset end-to-end speed or memory maxima. "
             "The stored io_excluded field describes whether benchmark I/O was excluded."),
            ("No graph is reconstructed. Optional saved graph JSON is shown only as supplied; "
             "neighbor selection is limited to included edges."),
        ],
    }


def build_payload(eval_root: str | Path, *, title: str = "ASGCN-U-Net offline results",
                  graph_json: str | Path | None = None,
                  limits: ExportLimits | None = None,
                  retained_budget_bytes: int | None = None) -> dict:
    """Read every saved comparison PNG, rejecting inconsistent/incomplete runs."""
    limits = limits or ExportLimits()
    metadata = _RetainedMetadataBudget(retained_budget_bytes)
    metadata.charge(32768 + len(title) * 8, "base payload metadata")
    root = Path(eval_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise OfflineViewerError("eval_root must contain aid/ and/or hdr/ directories")
    payload = empty_payload(title)
    images = payload["images"]
    input_paths: set[Path] = set()
    input_bytes = 0

    def register(path: Path) -> Path:
        nonlocal input_bytes
        path = _inside(path, root)
        if path not in input_paths:
            metadata.charge(2048 + len(str(path)) * 8, "registering an artifact path")
            input_bytes += path.stat().st_size
            if input_bytes > limits.max_input_bytes:
                raise OfflineViewerError("Artifacts exceed max_input_bytes; no subset made")
            input_paths.add(path)
        return path

    for key, (label, kind) in _DATASETS.items():
        base = root / key
        if not base.is_dir():
            continue
        if not base.resolve().is_relative_to(root):
            raise OfflineViewerError("Dataset directory escapes selected root")
        metadata.charge(4096, "dataset metadata containers")
        modes, frames, totals = [], {}, set()
        for directory in _iter_artifact_directory(base):
            if not directory.is_dir():
                continue
            if not _MODE.fullmatch(directory.name):
                if ".failed-" in directory.name or ".incomplete-" in directory.name:
                    metadata.charge(1024 + len(directory.name) * 8, "archive warning")
                    payload["warnings"].append(f"Archived run excluded: {key}/{directory.name}")
                continue
            mode = directory.name
            metadata.charge(4096 + len(mode) * 8, "mode and candidate containers")
            report = metadata.read_selected(register(directory / "metrics.json"),
                                            _REPORT_SELECTION, limits)
            expected = "ann" if mode == "ann" else "snn"
            if report.get("dataset") != kind or report.get("inference_mode") != expected:
                raise OfflineViewerError(f"Dataset/inference identity mismatch: {key}/{mode}")
            if expected == "snn" and mode != (
                f"snn_{report.get('snn_dynamics')}_T{report.get('simulation_steps')}"
            ):
                raise OfflineViewerError(f"SNN identity mismatch: {key}/{mode}")
            quality = report.get("quality")
            if not isinstance(quality, dict):
                raise OfflineViewerError(f"Missing quality object: {key}/{mode}")
            total = _integer(quality.get("frames"), "quality.frames", 1)
            totals.add(total)
            for average in ("micro", "macro"):
                values = quality.get(average, {})
                if not isinstance(values, dict):
                    raise OfflineViewerError(f"quality.{average} must be an object")
                quality[average] = {name: _number(value) for name, value in values.items()}
            candidates = {}
            predictions = directory / "predictions"
            if predictions.exists() and not predictions.resolve().is_relative_to(root):
                raise OfflineViewerError("Predictions directory escapes selected root")
            for path in _iter_artifact_directory(predictions, predictions=True):
                match = _STEM.fullmatch(path.name)
                if not match:
                    raise OfflineViewerError(f"Unrecognized prediction filename: {path.name}")
                index = int(match.group("index"))
                if index >= total or index in candidates:
                    raise OfflineViewerError("Duplicate/out-of-range saved dataset index")
                metadata.charge(1024, "registering a prediction candidate")
                candidates[index] = register(path)
            csv_path = register(directory / "frames.csv")
            csv_before, count = _signature(csv_path), 0
            with csv_path.open(encoding="utf-8", newline="") as handle:
                lines = (_BudgetCSVLines(handle, metadata, limits)
                         if retained_budget_bytes is not None else None)
                if lines is not None:
                    lines.begin_record()
                reader = csv.DictReader(lines if lines is not None else handle)
                if not reader.fieldnames or not {"sample_id", "scene"}.issubset(reader.fieldnames):
                    raise OfflineViewerError(f"Missing frame identity columns: {csv_path}")
                metadata.charge(_metadata_cost(reader.fieldnames), "CSV header fields")
                while True:
                    if lines is not None:
                        lines.begin_record()
                    try:
                        row = next(reader)
                    except StopIteration:
                        break
                    index = count
                    count += 1
                    if None in row or not row.get("sample_id") or row.get("scene") is None:
                        raise OfflineViewerError(f"Invalid frame row: {index}")
                    if index not in candidates:
                        continue
                    sample_id = row["sample_id"]
                    stem = _stem(sample_id, index)
                    if candidates[index].name != stem + "_pred.png":
                        raise OfflineViewerError(f"Prediction/CSV identity mismatch: {key}/{mode}/{index}")
                    if index not in frames:
                        metadata.charge(4096 + 8 * (len(sample_id) + len(row["scene"])),
                                        "registering a saved frame")
                        frames[index] = {"index": index, "sample_id": sample_id,
                                         "group": row["scene"], "images": [], "graph": None}
                    frame = frames[index]
                    if frame["sample_id"] != sample_id or frame["group"] != row["scene"]:
                        raise OfflineViewerError("Frame identities differ between modes")
                    ids = []
                    for path in (candidates[index], register(predictions / (stem + "_gt.png"))):
                        # Charge even duplicate images before hashing/constructing
                        # descriptors; overestimation avoids a late allocation check.
                        metadata.charge(4096 + len(str(path)) * 8, "image metadata descriptor")
                        metadata.scratch(_MIB, "streaming PNG hash/validation buffers")
                        info = _png(path, limits)
                        images.setdefault(info["sha256"], info)
                        ids.append(info["sha256"])
                    if any(images[ids[0]][dim] != images[ids[1]][dim]
                           for dim in ("width", "height", "channels")):
                        raise OfflineViewerError("Prediction and GT dimensions differ")
                    if frame["images"] and frame["images"][0]["target"] != ids[1]:
                        raise OfflineViewerError("Saved GT bytes differ between modes; comparison refused")
                    metadata.charge(4096, "frame/mode comparison metadata")
                    frame["images"].append({"mode": mode, "prediction": ids[0], "target": ids[1],
                                            "metrics": {name: _number(row.get(name))
                                                        for name in _METRICS},
                                            "report_eligible": _eligible(report.get("report_eligible"))})
            if _signature(csv_path) != csv_before:
                raise OfflineViewerError(f"CSV changed while being read: {csv_path}")
            if count != total:
                raise OfflineViewerError(f"Incomplete frames.csv: {count} rows, expected {total}")
            protocol = report.get("evaluation_protocol")
            if protocol is None:
                protocol = {}
            if not isinstance(protocol, dict):
                raise OfflineViewerError("evaluation_protocol must be an object")
            protocol["verification"] = "stored_metadata_only"
            benchmark = None
            if (directory / "benchmark.json").exists():
                benchmark = metadata.read_selected(register(directory / "benchmark.json"),
                                                    _BENCH_SELECTION, limits)
                for name in ("inference_mode", "simulation_steps", "snn_dynamics"):
                    if name in benchmark and benchmark[name] != report.get(name):
                        raise OfflineViewerError(f"Benchmark mode identity mismatch: {key}/{mode}")
                for name in ("mean_ms", "fps", "peak_gpu_memory_mb"):
                    benchmark[name] = _number(benchmark.get(name))
                benchmark["report_eligible"] = _eligible(benchmark.get("report_eligible"))
            modes.append({"id": mode, "label": mode, "saved_frames": len(candidates),
                          "report_eligible": _eligible(report.get("report_eligible")),
                          "quality": quality, "benchmark": benchmark, "protocol": protocol})
            if not candidates:
                metadata.charge(1024 + len(mode) * 8, "missing-PNG warning")
                payload["warnings"].append(f"{key}/{mode}: no saved PNGs; metrics only")
        if modes:
            if len(totals) != 1:
                raise OfflineViewerError(f"{key}: modes have different evaluation frame counts")
            metadata.charge(128 * (len(frames) + len(modes)), "final frame/mode ordering")
            modes.sort(key=lambda item: item["id"])
            for frame in frames.values():
                frame["images"].sort(key=lambda item: item["mode"])
            payload["datasets"].append({"id": key, "label": label, "total_frames": totals.pop(),
                                        "modes": modes,
                                        "frames": [frames[index] for index in sorted(frames)]})
    if not payload["datasets"]:
        raise OfflineViewerError("No completed aid/hdr artifacts; --empty is an explicit template only")
    if graph_json is not None:
        _attach_graphs(payload, Path(graph_json), limits, metadata_budget=metadata)
    payload["export"] = {"source": "saved evaluation artifacts", "input_bytes": input_bytes,
                         "image_files": len(images), "limits": vars(limits),
                         "graph_reconstruction": False, "model_inference": False,
                         "retained_metadata_budget_bytes": retained_budget_bytes,
                         "retained_metadata_estimate_bytes": metadata.retained,
                         "metadata_collection_peak_estimate_bytes": metadata.peak
                         if retained_budget_bytes is not None else None,
                         "metadata_budget_note": (
                             "Conservative collection estimate, not measured RSS or isolation. "
                             "Temporary candidate/report allocations are not refunded. With an "
                             "explicit budget, JSON parser and complete logical CSV records are "
                             "bounded before parsing; PNG decoding/base64 and later graph "
                             "generation require separate caller budgets."
                         )}
    return payload


def _attach_graphs(payload: dict, path: Path, limits: ExportLimits, *,
                   metadata_budget: _RetainedMetadataBudget | None = None) -> None:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > limits.max_graph_bytes:
        raise OfflineViewerError("Saved graph JSON exceeds max_graph_bytes or is not a regular file")
    graph_limits = ExportLimits(max_input_bytes=limits.max_graph_bytes,
                                max_metadata_bytes=limits.max_graph_bytes * 8)
    saved = (metadata_budget.read_selected(path, {"schema": True, "graphs": True}, graph_limits)
             if metadata_budget is not None else
             _read_selected(path, {"schema": True, "graphs": True}, graph_limits))
    if saved.get("schema") != "asgcn_offline_graphs_v1" or not isinstance(saved.get("graphs"), list):
        raise OfflineViewerError("Expected asgcn_offline_graphs_v1 with a graphs list")
    if metadata_budget is not None:
        metadata_budget.charge(1024 * sum(len(item["frames"]) for item in payload["datasets"]),
                               "graph attachment frame lookup")
    frames = {(dataset["id"], frame["index"]): frame
              for dataset in payload["datasets"] for frame in dataset["frames"]}
    for item in saved["graphs"]:
        if not isinstance(item, dict):
            raise OfflineViewerError("Saved graph entry must be an object")
        index = _integer(item.get("index"), "graph.index")
        dataset_key = item.get("dataset")
        if not isinstance(dataset_key, str):
            raise OfflineViewerError("graph.dataset must be a string")
        frame = frames.get((dataset_key, index))
        if frame is None or frame["sample_id"] != item.get("sample_id"):
            raise OfflineViewerError("Saved graph dataset/index/sample_id does not match saved frame")
        if frame["graph"] is not None:
            raise OfflineViewerError("Duplicate graph for a saved frame")
        graph = item.get("graph")
        if not isinstance(graph, dict):
            raise OfflineViewerError("Saved graph payload must be an object")
        nodes, edges, stats = (graph.get(name) for name in ("nodes", "edges", "statistics"))
        if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(stats, dict):
            raise OfflineViewerError("Graph requires nodes, edges, and statistics")
        if _integer(stats.get("nodes"), "statistics.nodes") != len(nodes):
            raise OfflineViewerError("Graph node count mismatch")
        if _integer(stats.get("displayed_edges"), "statistics.displayed_edges") != len(edges):
            raise OfflineViewerError("Graph displayed edge count mismatch")
        if _integer(stats.get("actual_directed_edges"), "statistics.actual_directed_edges") < len(edges):
            raise OfflineViewerError("Graph actual edge count is smaller than included edge count")
        for node in nodes:
            if not isinstance(node, list) or len(node) != 4:
                raise OfflineViewerError("Graph nodes must contain normalized x,y,t,polarity")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in node):
                raise OfflineViewerError("Graph node values must be finite numbers")
            if any(not 0 <= value <= 1 for value in node[:3]) or node[3] not in {-1, 1}:
                raise OfflineViewerError("Graph node coordinate/polarity range mismatch")
        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise OfflineViewerError("Graph edges must be source/destination pairs")
            if any(_integer(node, "edge node") >= len(nodes) for node in edge):
                raise OfflineViewerError("Graph edge endpoint is out of range")
        graph["offline_neighbor_scope"] = "included_edges_only"
        graph["identity_verified"] = False
        graph["offline_provenance_note"] = (
            "Copied from explicitly supplied saved JSON; dataset/index/sample_id match the CSV. "
            "Topology/source identity was not revalidated. Only included edges can be queried."
        )
        frame["graph"] = graph


class _Encoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, _PNGData):
            return value.data_url()
        return super().default(value)


def render_payload_html(payload: dict, output: str | Path, *,
                        limits: ExportLimits | None = None) -> dict:
    """Stream assets + escaped JSON to a new file; never replace existing output."""
    limits = limits or ExportLimits()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise OfflineViewerError(f"Output already exists; choose a new HTML path: {output}")
    if not output.parent.is_dir():
        raise OfflineViewerError(f"Output parent must already exist: {output.parent}")
    if not isinstance(payload, dict) or payload.get("schema") != "asgcn_offline_results_v1":
        raise OfflineViewerError("Expected asgcn_offline_results_v1 payload")
    asset_root = files("asgcn_unet").joinpath("viewer_assets")
    template = asset_root.joinpath("offline.html").read_text(encoding="utf-8")
    for token in ("__OFFLINE_CSS__", "__OFFLINE_JS__", "__OFFLINE_PAYLOAD__"):
        if template.count(token) != 1:
            raise OfflineViewerError(f"Offline template requires exactly one {token}")
    css = asset_root.joinpath("offline.css").read_text(encoding="utf-8")
    js = asset_root.joinpath("offline.js").read_text(encoding="utf-8")
    if "</style" in css.lower() or "</script" in js.lower():
        raise OfflineViewerError("Unsafe closing tag in offline assets")
    template = template.replace("__OFFLINE_CSS__", css).replace("__OFFLINE_JS__", js)
    before, after = template.split("__OFFLINE_PAYLOAD__")
    encoder = _Encoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    count, temporary = 0, None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".asgcn-offline-", suffix=".partial",
                                         dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)

            def write(text: str) -> None:
                nonlocal count
                data = text.encode("utf-8")
                count += len(data)
                if count > limits.max_output_bytes:
                    raise OfflineViewerError("HTML exceeds max_output_bytes; no subset/resize made")
                handle.write(data)

            write(before)
            for chunk in encoder.iterencode(payload):
                write(chunk.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
            write(after)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory link atomically publishes without replacing a destination,
        # including a destination created by another process during export.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)  # Only this export's own temporary file.
    return {"output": str(output), "bytes": count, "datasets": len(payload["datasets"]),
            "saved_frames": sum(len(data["frames"]) for data in payload["datasets"]),
            "embedded_images": len(payload["images"]), "offline": True,
            "graph_reconstruction": False, "model_inference": False}


def export_results_html(eval_root: str | Path, output: str | Path, *,
                        title: str = "ASGCN-U-Net offline results",
                        graph_json: str | Path | None = None,
                        limits: ExportLimits | None = None) -> dict:
    """Real saved artifact export; absence/errors never fall back to an empty demo."""
    output_path = Path(output).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise OfflineViewerError(f"Output already exists; choose a new HTML path: {output_path}")
    payload = build_payload(eval_root, title=title, graph_json=graph_json, limits=limits)
    return render_payload_html(payload, output, limits=limits)
