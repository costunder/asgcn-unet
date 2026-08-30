from __future__ import annotations

import os
import re
import zlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from torch.utils.data import Dataset

from .common import (
    choose_crop,
    crop_events,
    image_array_to_tensor,
    make_sample,
    normalize_polarity,
    stratified_subsample,
    uniform_cap_ratio,
    validate_target_normalization,
)

_EVENT_ARRAY_NAMES = ("xs", "ys", "ts", "ps")
_IMAGE_KEY_RE = re.compile(r"image(\d+)$")
_TIMESTAMP_CHUNK_SIZE = 1_048_576


def _recover_event_indices(
    event_ts: h5py.Dataset, frame_timestamps: np.ndarray, path: Path
) -> np.ndarray:
    """Recover missing legacy indices without loading or rewriting the event stream.

    Use the linked packager's predecessor convention with a global lower bound,
    independent of its chunk-local clamping: max(searchsorted(left) - 1, 0).
    These are NOT standard half-open timestamp boundaries. Existing attributes
    remain authoritative and are never repaired.
    """
    event_count = len(event_ts)
    if event_count == 0:
        return np.zeros(len(frame_timestamps), dtype=np.int64)
    insertion = np.full(len(frame_timestamps), event_count, dtype=np.int64)
    pending = 0
    previous_last = None
    first_timestamp = None
    for start in range(0, event_count, _TIMESTAMP_CHUNK_SIZE):
        block = np.asarray(event_ts[start : start + _TIMESTAMP_CHUNK_SIZE])
        if not np.all(np.isfinite(block)):
            raise _invalid_file(path, "events/ts timestamps must be finite to recover event_idx")
        if np.any(block[1:] < block[:-1]) or (
            previous_last is not None and block[0] < previous_last
        ):
            raise _invalid_file(
                path, "events/ts timestamps must be monotonically non-decreasing to recover event_idx"
            )
        if first_timestamp is None:
            first_timestamp = block[0]
        previous_last = block[-1]
        # Resolve each query in the first block reaching its timestamp. This also
        # chooses the first equal event when duplicate timestamps cross blocks.
        reached = int(np.searchsorted(frame_timestamps, block[-1], side="right"))
        if reached > pending:
            insertion[pending:reached] = start + np.searchsorted(
                block, frame_timestamps[pending:reached], side="left"
            )
            pending = reached
        # Validate the remaining stream even after every frame is indexed.
    if frame_timestamps[-1] < first_timestamp or frame_timestamps[0] > previous_last:
        raise _invalid_file(
            path, "image timestamps and events/ts have disjoint ranges; cannot recover event_idx"
        )
    return np.maximum(insertion - 1, 0)


def _invalid_file(path: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid EventHDR file {path}: {detail}")


def _numeric_scalar_attr(node: h5py.Dataset, name: str, path: Path) -> float:
    if name not in node.attrs:
        raise _invalid_file(path, f"images/{node.name.rsplit('/', 1)[-1]} is missing '{name}'")
    raw = np.asarray(node.attrs[name])
    if raw.size != 1 or raw.dtype.kind not in "iuf":
        raise _invalid_file(
            path,
            f"images/{node.name.rsplit('/', 1)[-1]} attribute '{name}' must be one number",
        )
    value = float(raw.reshape(-1)[0])
    if not np.isfinite(value):
        raise _invalid_file(
            path,
            f"images/{node.name.rsplit('/', 1)[-1]} attribute '{name}' must be finite",
        )
    return value


def _validate_event_values(
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    *,
    expected: int,
    height: int,
    width: int,
    source: str,
) -> None:
    arrays = {"xs": xs, "ys": ys, "ts": ts, "ps": ps}
    lengths = {name: int(values.size) for name, values in arrays.items()}
    if any(values.ndim != 1 for values in arrays.values()) or any(
        length != expected for length in lengths.values()
    ):
        raise ValueError(
            f"Invalid EventHDR event block {source}: expected {expected} values per array, "
            f"got {lengths}"
        )

    if not np.all(np.isfinite(ts)):
        raise ValueError(f"Invalid EventHDR event block {source}: timestamps must be finite")
    if ts.size > 1 and np.any(ts[1:] < ts[:-1]):
        raise ValueError(
            f"Invalid EventHDR event block {source}: timestamps must be monotonically "
            "non-decreasing"
        )

    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError(f"Invalid EventHDR event block {source}: coordinates must be finite")
    if np.any((xs < 0) | (xs >= width)) or np.any((ys < 0) | (ys >= height)):
        raise ValueError(
            f"Invalid EventHDR event block {source}: coordinates must lie within "
            f"x=[0,{width}), y=[0,{height})"
        )

    if not np.all(np.isfinite(ps)):
        raise ValueError(f"Invalid EventHDR event block {source}: polarity must be finite")
    valid_polarity = (ps == -1) | (ps == 0) | (ps == 1)
    if not np.all(valid_polarity):
        raise ValueError(
            f"Invalid EventHDR event block {source}: polarity values must be -1/1 or 0/1"
        )


class EventHDRDataset(Dataset):
    """Read the official EventHDR HDF5 structure without preprocessing copies."""

    def __init__(
        self,
        root: str | Path,
        target_channels: int = 1,
        max_events: int | None = 8192,
        crop_size: list[int] | tuple[int, int] | None = None,
        frame_stride: int = 1,
        tone_map: str = "log",
        tone_map_mu: float = 5000.0,
        target_normalization: dict[str, Any] | None = None,
        random_crop: bool = False,
        seed: int = 2026,
        allowed_files: list[str] | None = None,
        file_to_scene: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        self.frame_stride = max(1, int(frame_stride))
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
        self.target_normalization = validate_target_normalization(target_normalization)
        self.random_crop = random_crop
        self.seed = int(seed)
        self._handles: dict[Path, h5py.File] = {}
        self._owner_pid = os.getpid()
        self.zero_event_intervals = 0
        self.event_indexing: dict[str, dict[str, str | int]] = {}
        discovered = sorted([*self.root.rglob("*.h5"), *self.root.rglob("*.hdf5")])
        if not discovered:
            raise FileNotFoundError(
                f"No EventHDR .h5/.hdf5 files found under {self.root}. "
                "Place the official files in this directory or update dataset.root."
            )
        self.file_keys = {path: path.relative_to(self.root).as_posix() for path in discovered}
        key_to_path = {key: path for path, key in self.file_keys.items()}
        self.files = discovered
        if allowed_files is not None:
            allowed = [str(value).replace("\\", "/") for value in allowed_files]
            if len(allowed) != len(set(allowed)):
                raise ValueError("EventHDR allowed_files contains duplicate paths")
            missing = sorted(set(allowed) - set(key_to_path))
            if missing:
                preview = ", ".join(missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                raise FileNotFoundError(
                    f"EventHDR split requires {len(allowed)} files but {len(missing)} are "
                    f"missing under {self.root}: {preview}{suffix}"
                )
            self.files = [key_to_path[key] for key in allowed]
        selected_keys = [self.file_keys[path] for path in self.files]
        if file_to_scene is None:
            self.file_to_scene = {key: key for key in selected_keys}
        else:
            if not isinstance(file_to_scene, dict):
                raise TypeError("EventHDR file_to_scene must be a dictionary")
            normalized_mapping: dict[str, str] = {}
            for raw_key, scene_id in file_to_scene.items():
                if not isinstance(raw_key, str) or not raw_key.strip():
                    raise ValueError("EventHDR file_to_scene contains an invalid file key")
                key = raw_key.replace("\\", "/")
                if key in normalized_mapping:
                    raise ValueError(
                        f"EventHDR file_to_scene contains duplicate normalized key: {key}"
                    )
                if (
                    not isinstance(scene_id, str)
                    or not scene_id.strip()
                    or scene_id != scene_id.strip()
                ):
                    raise ValueError(f"EventHDR file_to_scene has an invalid scene ID for {key}")
                normalized_mapping[key] = scene_id
            missing_scenes = sorted(set(selected_keys) - set(normalized_mapping))
            if missing_scenes:
                raise ValueError(
                    "EventHDR file_to_scene is missing selected files: "
                    + ", ".join(missing_scenes[:8])
                    + (" ..." if len(missing_scenes) > 8 else "")
                )
            self.file_to_scene = {key: normalized_mapping[key] for key in selected_keys}
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No valid EventHDR frames found under {self.root}")

    def _build_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.files:
            source_file = self.file_keys[path]
            scene = self.file_to_scene[source_file]
            with h5py.File(path, "r") as h5:
                events_group = h5.get("events")
                images_group = h5.get("images")
                if not isinstance(events_group, h5py.Group):
                    raise _invalid_file(path, "required group 'events' is missing")
                if not isinstance(images_group, h5py.Group):
                    raise _invalid_file(path, "required group 'images' is missing")

                lengths: dict[str, int] = {}
                for name in _EVENT_ARRAY_NAMES:
                    node = events_group.get(name)
                    if not isinstance(node, h5py.Dataset):
                        raise _invalid_file(path, f"required array 'events/{name}' is missing")
                    allowed_kinds = "biuf" if name == "ps" else "iuf"
                    if node.ndim != 1 or node.dtype.kind not in allowed_kinds:
                        raise _invalid_file(
                            path, f"events/{name} must be a one-dimensional numeric array"
                        )
                    lengths[name] = len(node)
                if len(set(lengths.values())) != 1:
                    raise _invalid_file(
                        path, f"event arrays must have equal lengths, got {lengths}"
                    )
                event_count = lengths["ts"]

                numeric_image_keys: dict[int, str] = {}
                for key in images_group:
                    if not key.startswith("image"):
                        continue
                    match = _IMAGE_KEY_RE.fullmatch(key)
                    if match is None:
                        raise _invalid_file(
                            path,
                            f"images/{key} must use the numeric image<index> naming contract",
                        )
                    numeric_index = int(match.group(1))
                    previous = numeric_image_keys.get(numeric_index)
                    if previous is not None:
                        raise _invalid_file(
                            path,
                            f"images/{key} duplicates numeric image index {numeric_index} "
                            f"already used by images/{previous}",
                        )
                    numeric_image_keys[numeric_index] = key
                if not numeric_image_keys:
                    raise _invalid_file(path, "group 'images' contains no image arrays")
                image_keys = [numeric_image_keys[index] for index in sorted(numeric_image_keys)]
                frames: list[tuple[str, float, int | None]] = []
                previous_timestamp: float | None = None
                for key in image_keys:
                    node = images_group[key]
                    if not isinstance(node, h5py.Dataset):
                        raise _invalid_file(path, f"images/{key} must be an image array")
                    timestamp = _numeric_scalar_attr(node, "timestamp", path)
                    if previous_timestamp is not None and timestamp < previous_timestamp:
                        raise _invalid_file(
                            path, "image timestamps must be monotonically non-decreasing"
                        )
                    previous_timestamp = timestamp
                    end_idx = None
                    if "event_idx" in node.attrs:
                        raw_end_idx = _numeric_scalar_attr(node, "event_idx", path)
                        if not raw_end_idx.is_integer():
                            raise _invalid_file(path, f"images/{key} event_idx must be an integer")
                        end_idx = int(raw_end_idx)
                        if not 0 <= end_idx <= event_count:
                            raise _invalid_file(
                                path,
                                f"images/{key} event_idx={end_idx} is outside [0,{event_count}]",
                            )
                    frames.append((key, timestamp, end_idx))
                missing_count = sum(end is None for _, _, end in frames)
                recovered = (
                    _recover_event_indices(
                        events_group["ts"],
                        np.asarray([timestamp for _, timestamp, _ in frames], dtype=np.float64),
                        path,
                    )
                    if missing_count
                    else None
                )
                self.event_indexing[source_file] = {
                    "policy": "stored_or_timestamp_predecessor_v1",
                    "stored_images": len(frames) - missing_count,
                    "derived_images": missing_count,
                }
                selected_start_idx = 0
                selected_start_timestamp: float | None = None
                selected_sequence_index = 0
                previous_end_idx: int | None = None
                for frame_index, (key, timestamp, stored_idx) in enumerate(frames):
                    if stored_idx is None:
                        assert recovered is not None
                        end_idx = int(recovered[frame_index])
                        index_source = "timestamp_predecessor_v1"
                    else:
                        end_idx = stored_idx
                        index_source = "stored"
                    if previous_end_idx is not None and end_idx < previous_end_idx:
                        raise _invalid_file(
                            path, "image event_idx values must be monotonically non-decreasing"
                        )
                    previous_end_idx = end_idx
                    if frame_index % self.frame_stride == 0:
                        is_zero_event_interval = end_idx == selected_start_idx
                        if is_zero_event_interval:
                            self.zero_event_intervals += 1
                        samples.append(
                            {
                                "path": path,
                                "scene": scene,
                                "source_file": source_file,
                                "image_key": key,
                                "start_idx": selected_start_idx,
                                "end_idx": end_idx,
                                "event_idx_source": index_source,
                                "t0": selected_start_timestamp,
                                "timestamp": timestamp,
                                "sequence_index": selected_sequence_index,
                                "zero_event_interval": is_zero_event_interval,
                            }
                        )
                        # With frame_stride > 1, aggregate every skipped event interval
                        # into the next selected output instead of silently discarding it.
                        selected_start_idx = end_idx
                        selected_start_timestamp = timestamp
                        selected_sequence_index += 1
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_handle(self, path: Path) -> h5py.File:
        process_id = os.getpid()
        if process_id != self._owner_pid:
            # DataLoader workers must never reuse HDF5 objects inherited through fork.
            self._handles = {}
            self._owner_pid = process_id
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        h5 = self._get_handle(item["path"])
        start, end = item["start_idx"], item["end_idx"]
        xs = np.asarray(h5["events/xs"][start:end], dtype=np.float32)
        ys = np.asarray(h5["events/ys"][start:end], dtype=np.float32)
        ts = np.asarray(h5["events/ts"][start:end], dtype=np.float64)
        raw_ps = np.asarray(h5["events/ps"][start:end])
        image = np.asarray(h5["images"][item["image_key"]])
        target = image_array_to_tensor(
            image,
            self.target_channels,
            tone_map=self.tone_map,
            tone_map_mu=self.tone_map_mu,
            target_normalization=self.target_normalization,
            source=f"{item['path']}::{item['image_key']}",
        )
        height, width = target.shape[-2:]
        _validate_event_values(
            xs,
            ys,
            ts,
            raw_ps,
            expected=end - start,
            height=height,
            width=width,
            source=f"{item['path']}::{item['image_key']}",
        )
        ps = normalize_polarity(raw_ps)
        events = np.column_stack((xs, ys, ts, ps))
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1e-9)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        raw_event_count = len(events)
        # Recurrent pixels and temporal losses must refer to the same sensor ROI
        # throughout one source sequence. The crop is deterministic per file, not
        # per frame; epoch-varying sequence crops are intentionally not implemented.
        crop_identity = f"{item['scene']}\0{item['source_file']}"
        crop_seed = (self.seed + zlib.crc32(crop_identity.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(crop_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        cropped_event_count = len(events)
        dataset_sampling_ratio = uniform_cap_ratio(cropped_event_count, self.max_events)
        events = stratified_subsample(events, self.max_events)
        retained_event_count = len(events)
        sample_id = (
            f"{item['scene']}/{item['image_key']}"
            if item["scene"] == item["source_file"]
            else f"{item['scene']}/{item['source_file']}/{item['image_key']}"
        )
        t0 = item["t0"]
        t1 = item["timestamp"]
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventHDR",
                "timestamp": t1,
                "t0": t0,
                "t1": t1,
                "dt_us": round((t1 - t0) * 1_000_000) if t0 is not None else None,
                "source": str(item["path"]),
                "source_file": item["source_file"],
                "scene": item["scene"],
                "sequence_index": item["sequence_index"],
                "event_idx_source": item["event_idx_source"],
                "event_start_idx": start,
                "event_end_idx": end,
                "raw_event_count": raw_event_count,
                "cropped_event_count": cropped_event_count,
                "retained_event_count": retained_event_count,
                "dataset_sampling_ratio": dataset_sampling_ratio,
                "zero_event_interval": bool(item["zero_event_interval"]),
                "crop": {
                    "left": crop.left,
                    "top": crop.top,
                    "width": crop.width,
                    "height": crop.height,
                },
            },
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_owner_pid"] = None
        return state

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            try:
                handle.close()
            except (OSError, RuntimeError):
                pass
        handles.clear()

    def __del__(self) -> None:
        self.close()
