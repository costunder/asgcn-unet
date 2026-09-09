"""Read one recorded evaluation sample without constructing a dataset-wide index.

Only CPU source decoding is performed. The saved protocol must already have been
validated by the caller. The selected source's identity is checked again here;
this is not a whole-dataset hash/validation pass or a new model evaluation.
"""

from __future__ import annotations

import math
import os
import re
import zipfile
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import h5py
from PIL import Image

from .data.common import validate_target_normalization
from .data.eventaid_r import (
    _EVENT_RE,
    _GT_RE,
    _UPLOAD_EVENT_RE,
    _UPLOAD_GT_RE,
    EventAidRZipDataset,
)
from .data.eventhdr import EventHDRDataset, _numeric_scalar_attr
from .stream_input import (
    LEGACY_EVENT_TIME_CONTRACT,
    PHYSICAL_EVENT_TIME_CONTRACT,
    validate_event_time_contract,
)


class DiagnosticSampleError(ValueError):
    """Selected source cannot be decoded safely under the declared contract."""


class _Budget:
    def __init__(self, limit: int, reserve: Callable[[int, str], None] | None):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise DiagnosticSampleError("memory_budget_bytes must be a positive integer")
        self.limit, self.reserve, self.peak = limit, reserve, 0

    def check(self, estimate: int, stage: str) -> None:
        self.peak = max(self.peak, estimate)
        if estimate > self.limit:
            raise DiagnosticSampleError(
                f"{stage}: estimated source-reader working memory {estimate:,} bytes exceeds "
                f"the explicit {self.limit:,}-byte budget; source/model was not reduced"
            )
        if self.reserve is not None:
            self.reserve(estimate, stage)


def _integer(identity: dict, key: str, minimum: int = 0) -> int:
    value = identity.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DiagnosticSampleError(f"{key} must be an integer >= {minimum}")
    return value


def _text(identity: dict, key: str) -> str:
    value = identity.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise DiagnosticSampleError(f"{key} must be a nonempty string")
    return value


def _source(root: Path, value: str) -> Path:
    key = PurePosixPath(value.replace("\\", "/"))
    if key.is_absolute() or ".." in key.parts or ":" in value:
        raise DiagnosticSampleError("Source path must remain inside dataset.root")
    path = (root / key).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise DiagnosticSampleError("Source path escapes dataset.root or is not a file")
    return path


def _equal(value: Any, expected: Any, key: str) -> None:
    if isinstance(expected, bool) or value != expected:
        raise DiagnosticSampleError(f"Current source {key} does not match saved identity")


def _reader(cls, cfg: dict, item: dict, handle):
    # Reuse the real preprocessing implementation without its dataset-wide
    # constructor/index. The one-item reader owns this already-open source only.
    reader = cls.__new__(cls)
    reader._handles = {item["path"]: handle}
    reader._owner_pid = os.getpid()
    reader.samples = [item]
    reader.target_channels = int(cfg.get("target_channels", 1))
    reader.max_events = cfg.get("max_events", 8192)
    # This adapter is deliberately a legacy, single-window reader. The public
    # entry point rejects physical streams before opening any source: those need
    # a complete chronological prefix, which this one-item index cannot provide.
    reader.event_time_contract = LEGACY_EVENT_TIME_CONTRACT
    reader.timestamp_scale_to_seconds = None
    reader.interval_timestamp_scale_to_seconds = None
    reader.crop_size = tuple(cfg["crop_size"]) if cfg.get("crop_size") else None
    reader.tone_map = cfg.get("tone_map", "log" if cls is EventHDRDataset else "none")
    reader.tone_map_mu = float(cfg.get("tone_map_mu", 5000.0))
    reader.target_normalization = validate_target_normalization(cfg.get("target_normalization"))
    # build_dataset(..., split='eval') always disables random_crop.
    reader.random_crop = False
    reader.seed = int(cfg.get("seed", 2026))
    return reader


def _h5_index(node: h5py.Dataset, ts: h5py.Dataset, path: Path) -> tuple[int, str]:
    if "event_idx" in node.attrs:
        value = _numeric_scalar_attr(node, "event_idx", path)
        if not value.is_integer() or not 0 <= value <= len(ts):
            raise DiagnosticSampleError("Invalid selected image event_idx")
        return int(value), "stored"
    timestamp = _numeric_scalar_attr(node, "timestamp", path)
    # Match max(searchsorted(left)-1, 0) with scalar reads. Full-stream monotonicity
    # was checked during original evaluation; this diagnostic validates only the
    # requested block and its boundary, not the whole current source file.
    low, high = 0, len(ts)
    while low < high:
        middle = (low + high) // 2
        value = float(ts[middle])
        if not math.isfinite(value):
            raise DiagnosticSampleError("Non-finite timestamp at selected boundary search")
        if value < timestamp:
            low = middle + 1
        else:
            high = middle
    return max(low - 1, 0), "timestamp_predecessor_v1"


def _hdr(cfg: dict, identity: dict, budget: _Budget) -> dict:
    root = Path(cfg["root"]).expanduser().resolve(strict=True)
    source_file = _text(identity, "source_file")
    path = _source(root, source_file)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise DiagnosticSampleError("EventHDR source must be HDF5")
    group = _text(identity, "group")
    expected_group = cfg.get("file_to_scene", {}).get(source_file, source_file)
    _equal(group, expected_group, "group")
    image_key = _text(identity, "image_key")
    if re.fullmatch(r"image[0-9]+", image_key) is None:
        raise DiagnosticSampleError("Invalid EventHDR image_key")
    sequence = _integer(identity, "sequence_index")
    start, end = _integer(identity, "start_idx"), _integer(identity, "end_idx")
    if end < start:
        raise DiagnosticSampleError("EventHDR end_idx must not precede start_idx")
    stride = max(1, int(cfg.get("frame_stride", 1)))
    budget.check(8 * 1024**2, "HDF5 metadata open")
    with h5py.File(path, "r", rdcc_nbytes=1024**2) as handle:
        for name in ("events", "images"):
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise DiagnosticSampleError("External/soft HDF5 group links are not supported")
            if not isinstance(handle.get(name), h5py.Group):
                raise DiagnosticSampleError(f"Missing EventHDR {name} group")
        images = handle["images"]
        metadata_estimate = 8 * 1024**2 + len(images) * 1024
        budget.check(metadata_estimate, "Selected HDF5 image metadata")
        numbered = {}
        for key in images:
            if not key.startswith("image"):
                continue
            match = re.fullmatch(r"image([0-9]+)", key)
            if match is None or int(match[1]) in numbered:
                raise DiagnosticSampleError("Invalid/duplicate numeric HDF5 image key")
            numbered[int(match[1])] = key
        keys = [numbered[value] for value in sorted(numbered)]
        ordinal = sequence * stride
        if ordinal >= len(keys) or keys[ordinal] != image_key:
            raise DiagnosticSampleError("Selected HDF5 sequence_index/image_key mismatch")
        arrays = {}
        chunk_bytes = 0
        for name in ("xs", "ys", "ts", "ps"):
            if not isinstance(handle["events"].get(name, getlink=True), h5py.HardLink):
                raise DiagnosticSampleError("External/soft HDF5 event links are not supported")
            node = handle["events"].get(name)
            if not isinstance(node, h5py.Dataset) or node.ndim != 1:
                raise DiagnosticSampleError("Expected one-dimensional HDF5 event arrays")
            if node.dtype.kind not in ("biuf" if name == "ps" else "iuf"):
                raise DiagnosticSampleError("HDF5 event arrays must have numeric dtypes")
            arrays[name] = node
            chunk_bytes += math.prod(node.chunks or (1,)) * node.dtype.itemsize
        lengths = {len(node) for node in arrays.values()}
        if len(lengths) != 1 or end > next(iter(lengths)):
            raise DiagnosticSampleError("Selected event window is outside aligned HDF5 arrays")
        budget.check(metadata_estimate + chunk_bytes, "HDF5 event decompression chunks")
        selected = [image_key] + ([keys[ordinal - stride]] if sequence else [])
        for key in selected:
            if not isinstance(images.get(key, getlink=True), h5py.HardLink):
                raise DiagnosticSampleError("External/soft HDF5 image links are not supported")
            if not isinstance(images[key], h5py.Dataset):
                raise DiagnosticSampleError("Selected HDF5 image must be an array")
        target = images[image_key]
        timestamp = _numeric_scalar_attr(target, "timestamp", path)
        _equal(timestamp, identity.get("timestamp"), "timestamp")
        actual_end, index_source = _h5_index(target, arrays["ts"], path)
        _equal(actual_end, end, "end_idx")
        t0 = None
        actual_start = 0
        if sequence:
            previous = images[keys[ordinal - stride]]
            t0 = _numeric_scalar_attr(previous, "timestamp", path)
            actual_start, _ = _h5_index(previous, arrays["ts"], path)
            if t0 > timestamp:
                raise DiagnosticSampleError("Selected image timestamps are not ordered")
        _equal(actual_start, start, "start_idx")
        if len(target.shape) not in (2, 3) or target.dtype.kind not in "iuf":
            raise DiagnosticSampleError("Expected numeric HxW or HxWxC target array")
        pixels = math.prod(target.shape)
        target_chunks = math.prod(target.chunks or (1,)) * target.dtype.itemsize
        estimate = metadata_estimate + chunk_bytes + target_chunks
        estimate += (end - start) * 160 + pixels * (target.dtype.itemsize + 64)
        budget.check(estimate, "Selected HDF5 window and target decode")
        item = {"path": path, "scene": group, "source_file": source_file,
                "image_key": image_key, "start_idx": start, "end_idx": end,
                "event_idx_source": index_source, "t0": t0, "timestamp": timestamp,
                "sequence_index": sequence, "zero_event_interval": start == end}
        reader = _reader(EventHDRDataset, cfg, item, handle)
        try:
            return reader[0]
        finally:
            reader.close()


def _zip_directory_cost(path: Path, budget: _Budget) -> int:
    # _EndRecData reads at most the ZIP comment trailer plus fixed ZIP64 records,
    # not the central directory. Check before ZipFile materializes every ZipInfo.
    budget.check(128 * 1024, "ZIP end-directory metadata")
    with path.open("rb") as handle:
        record = zipfile._EndRecData(handle)
    if record is None:
        raise DiagnosticSampleError("Invalid ZIP end-directory record")
    entries = int(record[zipfile._ECD_ENTRIES_TOTAL])
    directory_bytes = int(record[zipfile._ECD_SIZE])
    if record[zipfile._ECD_DISK_NUMBER] or record[zipfile._ECD_DISK_START]:
        raise DiagnosticSampleError("Multi-disk ZIPs are not supported")
    estimate = 8 * 1024**2 + directory_bytes * 4 + entries * 4096
    budget.check(estimate, "Selected ZIP central directory")
    return estimate


def _aid(cfg: dict, identity: dict, budget: _Budget) -> dict:
    root = Path(cfg["root"]).expanduser().resolve(strict=True)
    group = _text(identity, "group")
    if re.fullmatch(r"R-[^/\\:]+", group) is None:
        raise DiagnosticSampleError("Invalid EventAid-R scene identity")
    path = _source(root, group + ".zip")
    frame_id = _integer(identity, "frame_id", 1)
    _equal(_integer(identity, "sequence_index"), frame_id, "sequence_index")
    event_name, target_name = _text(identity, "event_name"), _text(identity, "target_name")
    offset = cfg.get("target_offset", 1)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise DiagnosticSampleError("target_offset must be an integer")
    metadata_estimate = _zip_directory_cost(path, budget)
    with zipfile.ZipFile(path, "r") as archive:
        names = EventAidRZipDataset._validated_member_names(archive, path=path)
        upload = _UPLOAD_EVENT_RE.search(event_name.replace("\\", "/")) is not None
        event_pattern, target_pattern = (_UPLOAD_EVENT_RE, _UPLOAD_GT_RE) if upload else (
            _EVENT_RE, _GT_RE)
        events = EventAidRZipDataset._index_numbered_members(
            names, event_pattern, label="event", path=path)
        targets = EventAidRZipDataset._index_numbered_members(
            names, target_pattern, label="GT", path=path)
        _equal(events.get(frame_id), event_name, "event_name")
        _equal(targets.get(frame_id + offset), target_name, "target_name")
        opposite = (_EVENT_RE, _GT_RE) if upload else (_UPLOAD_EVENT_RE, _UPLOAD_GT_RE)
        if any(pattern.search(name.replace("\\", "/")) for name in names for pattern in opposite):
            raise DiagnosticSampleError("Mixed EventAid-R archive layouts")
        timestamp_name = EventAidRZipDataset._unique_metadata_member(
            names, "timestamps_upload.txt" if upload else "timestamps.txt", path=path)
        if timestamp_name is None:
            raise DiagnosticSampleError("Selected archive has no timestamp metadata")
        metadata_names = [timestamp_name]
        for name in ("shape.txt", "parts.txt"):
            member = EventAidRZipDataset._unique_metadata_member(names, name, path=path)
            if member is not None:
                metadata_names.append(member)
        metadata_estimate += sum(archive.getinfo(name).file_size for name in metadata_names) * 64
        budget.check(metadata_estimate, "Selected ZIP timing/shape metadata")
        timestamps = [int(value) for value in archive.read(timestamp_name).decode("utf-8").split()]
        if any(b <= a for a, b in pairwise(timestamps)):
            raise DiagnosticSampleError("Archive timestamps must be strictly increasing")
        shape = EventAidRZipDataset._read_shape(archive, names, path=path)
        event_ids = sorted(events)
        if upload:
            if len(timestamps) != len(event_ids):
                raise DiagnosticSampleError("Upload timestamp/member count mismatch")
            parts = EventAidRZipDataset._read_parts(
                archive, names, path=path, event_ids=event_ids, target_ids=sorted(targets))
            part_index = _integer(identity, "part_index")
            if part_index >= len(parts):
                raise DiagnosticSampleError("Invalid upload part_index")
            low, high = parts[part_index]
            if not (low <= frame_id < high and low <= frame_id + offset <= high):
                raise DiagnosticSampleError("Selected pair crosses an upload part boundary")
            _equal(identity.get("sequence_id"), f"{group}/part-{part_index:03d}", "sequence_id")
            row = event_ids.index(frame_id)
        else:
            if "part_index" in identity or "sequence_id" in identity:
                raise DiagnosticSampleError("Unexpected part identity in regular archive")
            row = frame_id - 1
        if row < 0 or row + 1 >= len(timestamps):
            raise DiagnosticSampleError("Selected interval is not covered by archive timestamps")
        t0, t1 = timestamps[row:row + 2]
        _equal(t0, identity.get("t0_us"), "t0_us")
        _equal(t1, identity.get("t1_us"), "t1_us")
        event_bytes = archive.getinfo(event_name).file_size
        target_bytes = archive.getinfo(target_name).file_size
        estimate = metadata_estimate + event_bytes * 40 + target_bytes * 4
        budget.check(estimate, "Selected ZIP event text and target header")
        with archive.open(target_name) as stream, Image.open(stream) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                raise DiagnosticSampleError("Target dimensions must be positive")
            if shape is not None and shape != (height, width):
                raise DiagnosticSampleError("Archive shape metadata does not match selected target")
        estimate += width * height * 128
        budget.check(estimate, "Selected ZIP event and image decode")
        item = {"path": path, "scene": group, "frame_id": frame_id,
                "event_name": event_name, "target_name": target_name, "shape": shape,
                "sequence_index": frame_id, "t0_us": t0, "t1_us": t1}
        if upload:
            item.update(part_index=part_index, sequence_id=identity["sequence_id"])
        reader = _reader(EventAidRZipDataset, cfg, item, archive)
        try:
            return reader[0]
        finally:
            reader.close()


def read_diagnostic_sample(
    config: dict,
    identity: dict,
    *,
    memory_budget_bytes: int,
    reserve: Callable[[int, str], None] | None = None,
) -> dict:
    """Read one real saved-protocol frame using unchanged evaluation preprocessing.

    ``config`` may be the full resolved experiment config or its dataset section.
    ``reserve`` may refuse each planned allocation using current host/cgroup RAM.
    Estimates are conservative planning estimates, not measured RSS or a hard OS
    allocation limit. Budget refusal never subsamples or substitutes source data.
    Existing configured max_events is applied exactly by the normal reader.
    Physical/event-driven streams are refused because their causal prefix is not
    represented by a selected single-frame source window.
    """
    if not isinstance(config, dict) or not isinstance(identity, dict):
        raise DiagnosticSampleError("config and identity must be dictionaries")
    cfg = config.get("dataset", config)
    if not isinstance(cfg, dict):
        raise DiagnosticSampleError("dataset config must be a dictionary")
    model = config.get("model", {})
    if not isinstance(model, dict):
        raise DiagnosticSampleError("model config must be a dictionary")
    time_contract = cfg.get("event_time_contract", LEGACY_EVENT_TIME_CONTRACT)
    if (time_contract == PHYSICAL_EVENT_TIME_CONTRACT
            or model.get("graph_execution") == "event_driven"
            or model.get("architecture_version") == 3):
        raise DiagnosticSampleError(
            "Single-frame diagnostics do not support physical_seconds_v1/event-driven streams; "
            "a complete chronological prefix is required. No window-normalized fallback was applied."
        )
    try:
        validate_event_time_contract(
            time_contract, cfg.get("timestamp_scale_to_seconds"), cfg.get("max_events", 8192),
            interval_timestamp_scale_to_seconds=cfg.get("interval_timestamp_scale_to_seconds"),
        )
    except (TypeError, ValueError) as error:
        raise DiagnosticSampleError(str(error)) from error
    _integer(identity, "dataset_index")
    budget = _Budget(memory_budget_bytes, reserve)
    kind = cfg.get("type")
    if kind == "eventhdr":
        sample = _hdr(cfg, identity, budget)
    elif kind == "eventaid_r_zip":
        sample = _aid(cfg, identity, budget)
    else:
        raise DiagnosticSampleError("Only EventHDR and EventAid-R diagnostics are supported")
    sample["metadata"]["diagnostic_source_read"] = {
        "scope": "one selected source window; no dataset-wide index or model inference",
        "estimated_working_memory_bytes": budget.peak,
        "memory_budget_bytes": budget.limit,
        "full_source_hash_verified": False,
    }
    return sample
