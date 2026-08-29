from __future__ import annotations

import io
import os
import re
import zipfile
import zlib
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .common import (
    choose_crop,
    crop_events,
    image_array_to_tensor,
    make_sample,
    normalize_polarity,
    pil_to_array,
    stratified_subsample,
    uniform_cap_ratio,
    validate_target_normalization,
)

_EVENT_RE = re.compile(r"(?:^|/)event/(\d+)\.txt$", re.IGNORECASE)
_GT_RE = re.compile(r"(?:^|/)gt/(\d+)_img\.png$", re.IGNORECASE)


def _validate_event_coordinates(
    events: np.ndarray, *, height: int, width: int, source: str
) -> None:
    if not len(events):
        return
    xs, ys = events[:, 0], events[:, 1]
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError(f"Invalid EventAid-R event block {source}: coordinates must be finite")
    if np.any((xs < 0) | (xs >= width)) or np.any((ys < 0) | (ys >= height)):
        raise ValueError(
            f"Invalid EventAid-R event block {source}: coordinates must lie within "
            f"x=[0,{width}), y=[0,{height})"
        )


class EventAidRZipDataset(Dataset):
    """Read official EventAid-R scene ZIPs directly to avoid an extracted copy."""

    def __init__(
        self,
        root: str | Path,
        target_channels: int = 1,
        max_events: int | None = 8192,
        crop_size: list[int] | tuple[int, int] | None = None,
        target_offset: int = 1,
        tone_map: str = "none",
        tone_map_mu: float = 5000.0,
        target_normalization: dict[str, Any] | None = None,
        random_crop: bool = False,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        if isinstance(target_offset, bool) or not isinstance(target_offset, Integral):
            raise TypeError("target_offset must be an integer and must not be bool")
        self.target_offset = int(target_offset)
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
        self.target_normalization = validate_target_normalization(target_normalization)
        self.random_crop = random_crop
        self.seed = int(seed)
        self._handles: dict[Path, zipfile.ZipFile] = {}
        self._owner_pid = os.getpid()
        self.zip_paths = sorted(self.root.glob("R-*.zip"))
        if not self.zip_paths:
            raise FileNotFoundError(f"No R-*.zip files found under {self.root}")
        self.samples, self.scene_info = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No paired EventAid-R samples found under {self.root}")

    @staticmethod
    def _unique_metadata_member(names: list[str], basename: str, *, path: Path) -> str | None:
        candidates = [
            name
            for name in names
            if name.replace("\\", "/").rsplit("/", 1)[-1].casefold() == basename.casefold()
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"Invalid EventAid-R scene {path}: duplicate {basename} members: "
                + ", ".join(candidates)
            )
        return candidates[0] if candidates else None

    @classmethod
    def _read_shape(
        cls, zf: zipfile.ZipFile, names: list[str], *, path: Path
    ) -> tuple[int, int] | None:
        shape_name = cls._unique_metadata_member(names, "shape.txt", path=path)
        if not shape_name:
            return None
        try:
            values = zf.read(shape_name).decode("utf-8").split()
            if len(values) != 2:
                raise ValueError("expected exactly two integer tokens")
            width, height = (int(value) for value in values)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid EventAid-R shape in {path}::{shape_name}: "
                "expected exactly two integer tokens 'width height'"
            ) from error
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid EventAid-R shape in {path}::{shape_name}: dimensions must be positive"
            )
        return height, width

    @staticmethod
    def _index_numbered_members(
        names: list[str], pattern: re.Pattern[str], *, label: str, path: Path
    ) -> dict[int, str]:
        indexed: dict[int, str] = {}
        for name in names:
            match = pattern.search(name.replace("\\", "/"))
            if match is None:
                continue
            numeric_id = int(match.group(1))
            if numeric_id in indexed:
                raise ValueError(
                    f"Invalid EventAid-R scene {path}: duplicate numeric {label} ID "
                    f"{numeric_id}: {indexed[numeric_id]}, {name}"
                )
            indexed[numeric_id] = name
        return indexed

    @staticmethod
    def _validated_member_names(zf: zipfile.ZipFile, *, path: Path) -> list[str]:
        names: list[str] = []
        casefolded: dict[str, str] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            normalized = info.filename.replace("\\", "/")
            logical_name = normalized.casefold()
            if logical_name in casefolded:
                raise ValueError(
                    f"Invalid EventAid-R scene {path}: case-insensitive duplicate ZIP "
                    f"member: {casefolded[logical_name]}, {info.filename}"
                )
            casefolded[logical_name] = info.filename
            names.append(info.filename)
        return names

    def _build_index(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        scene_info: dict[str, Any] = {}
        for path in self.zip_paths:
            with zipfile.ZipFile(path) as zf:
                names = self._validated_member_names(zf, path=path)
                events = self._index_numbered_members(names, _EVENT_RE, label="event", path=path)
                targets = self._index_numbered_members(names, _GT_RE, label="GT", path=path)
                shape = self._read_shape(zf, names, path=path)
                timestamps_name = self._unique_metadata_member(names, "timestamps.txt", path=path)
                if timestamps_name is None:
                    raise ValueError(f"Invalid EventAid-R scene {path}: timestamps.txt is missing")
                try:
                    timestamps = [
                        int(value) for value in zf.read(timestamps_name).decode("utf-8").split()
                    ]
                except (UnicodeDecodeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}"
                    ) from error
                if not timestamps:
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}: empty file"
                    )
                if any(current <= previous for previous, current in pairwise(timestamps)):
                    raise ValueError(
                        f"Invalid EventAid-R timestamps in {path}::{timestamps_name}: "
                        "values must be strictly increasing"
                    )
                event_ids = sorted(events)
                target_ids = sorted(targets)
                if not event_ids or not target_ids:
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event and GT files are required"
                    )
                if event_ids != list(range(event_ids[0], event_ids[-1] + 1)):
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event IDs are not contiguous"
                    )
                if target_ids != list(range(target_ids[0], target_ids[-1] + 1)):
                    raise ValueError(f"Invalid EventAid-R scene {path}: GT IDs are not contiguous")
                paired_ids = [
                    event_id for event_id in event_ids if event_id + self.target_offset in targets
                ]
                boundary = abs(self.target_offset)
                if self.target_offset >= 0:
                    allowed_event_gaps = set(event_ids[-boundary:]) if boundary else set()
                    allowed_target_gaps = set(target_ids[:boundary]) if boundary else set()
                else:
                    allowed_event_gaps = set(event_ids[:boundary])
                    allowed_target_gaps = set(target_ids[-boundary:])
                unpaired_events = set(event_ids) - set(paired_ids)
                paired_targets = {event_id + self.target_offset for event_id in paired_ids}
                unpaired_targets = set(target_ids) - paired_targets
                if (
                    not paired_ids
                    or unpaired_events - allowed_event_gaps
                    or unpaired_targets - allowed_target_gaps
                ):
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: event/GT pairing has internal gaps"
                    )
                if paired_ids[0] < 1 or len(timestamps) <= paired_ids[-1]:
                    raise ValueError(
                        f"Invalid EventAid-R scene {path}: timestamps.txt does not cover "
                        "every paired event interval"
                    )
                scene = path.stem
                scene_info[scene] = {"shape": shape, "frames": len(targets), "events": len(events)}
                for event_id in paired_ids:
                    target_id = event_id + self.target_offset
                    samples.append(
                        {
                            "path": path,
                            "scene": scene,
                            "frame_id": event_id,
                            "event_name": events[event_id],
                            "target_name": targets[target_id],
                            "shape": shape,
                            "sequence_index": event_id,
                            "t0_us": timestamps[event_id - 1],
                            "t1_us": timestamps[event_id],
                        }
                    )
        return samples, scene_info

    def __len__(self) -> int:
        return len(self.samples)

    def _get_handle(self, path: Path) -> zipfile.ZipFile:
        process_id = os.getpid()
        if process_id != self._owner_pid:
            # Keep one independent archive descriptor per DataLoader worker.
            self._handles = {}
            self._owner_pid = process_id
        if path not in self._handles:
            self._handles[path] = zipfile.ZipFile(path)
        return self._handles[path]

    @staticmethod
    def _read_events(
        raw: bytes,
        source: str = "event file",
        *,
        interval_t0: float,
        interval_t1: float,
    ) -> tuple[np.ndarray, dict[str, float | int | None]]:
        if not raw.strip():
            return np.empty((0, 4), dtype=np.float32), {
                "event_timestamp_min": None,
                "event_timestamp_max": None,
                "event_timestamp_span": None,
                "interval_t0": float(interval_t0),
                "interval_t1": float(interval_t1),
                "event_to_interval_span_ratio": None,
                "event_min_offset_from_t0": None,
                "outside_interval_count": 0,
                "event_count": 0,
                "strict_interval_validation": False,
            }
        try:
            rows = np.loadtxt(io.BytesIO(raw), dtype=np.float64, comments=None, ndmin=2)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                f"Invalid EventAid-R event block {source}: every token must be numeric"
            ) from error
        if rows.ndim != 2 or rows.shape[1] != 4:
            raise ValueError(
                f"Invalid EventAid-R event block {source}: expected four values per event"
            )
        # Official text columns: timestamp, x, y, polarity.
        timestamps = rows[:, 0]
        if not np.all(np.isfinite(timestamps)):
            raise ValueError(f"Invalid EventAid-R event block {source}: timestamps must be finite")
        if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError(
                f"Invalid EventAid-R event block {source}: timestamps must be monotonically "
                "non-decreasing"
            )
        event_span = float(timestamps[-1] - timestamps[0])
        interval_span = float(interval_t1 - interval_t0)
        timestamp_diagnostics: dict[str, float | int | None] = {
            "event_timestamp_min": float(timestamps[0]),
            "event_timestamp_max": float(timestamps[-1]),
            "event_timestamp_span": event_span,
            "interval_t0": float(interval_t0),
            "interval_t1": float(interval_t1),
            "event_to_interval_span_ratio": (
                event_span / interval_span if interval_span > 0 else None
            ),
            "event_min_offset_from_t0": float(timestamps[0] - interval_t0),
            "outside_interval_count": int(
                np.count_nonzero((timestamps < interval_t0) | (timestamps > interval_t1))
            ),
            "event_count": int(timestamps.size),
            # Promote this diagnostic to a hard contract only after the official
            # 14-archive scan establishes a shared timestamp basis and unit.
            "strict_interval_validation": False,
        }
        polarity = rows[:, 3]
        if not np.all(np.isfinite(polarity)):
            raise ValueError(f"Invalid EventAid-R event block {source}: polarity must be finite")
        valid_polarity = (polarity == -1) | (polarity == 0) | (polarity == 1)
        if not np.all(valid_polarity):
            raise ValueError(
                f"Invalid EventAid-R event block {source}: polarity values must be -1/1 or 0/1"
            )
        events = rows[:, [1, 2, 0, 3]]
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1.0)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        events[:, 3] = normalize_polarity(events[:, 3])
        return events, timestamp_diagnostics

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        zf = self._get_handle(item["path"])
        source = f"{item['path']}::{item['event_name']}"
        events, timestamp_diagnostics = self._read_events(
            zf.read(item["event_name"]),
            source=source,
            interval_t0=float(item["t0_us"]),
            interval_t1=float(item["t1_us"]),
        )
        with Image.open(io.BytesIO(zf.read(item["target_name"]))) as image:
            target = image_array_to_tensor(
                pil_to_array(image),
                self.target_channels,
                tone_map=self.tone_map,
                tone_map_mu=self.tone_map_mu,
                target_normalization=self.target_normalization,
                source=f"{item['path']}::{item['target_name']}",
            )
        height, width = target.shape[-2:]
        if item["shape"] and item["shape"] != (height, width):
            raise ValueError(
                f"{item['scene']} shape.txt={item['shape']} but target={(height, width)}"
            )
        _validate_event_coordinates(events, height=height, width=width, source=source)
        raw_event_count = len(events)
        # Keep the sensor ROI aligned for recurrent/temporal evaluation.
        crop_identity = f"{item['scene']}\0{item['path'].name}"
        crop_seed = (self.seed + zlib.crc32(crop_identity.encode("utf-8"))) % (2**32)
        rng = np.random.default_rng(crop_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        cropped_event_count = len(events)
        dataset_sampling_ratio = uniform_cap_ratio(cropped_event_count, self.max_events)
        events = stratified_subsample(events, self.max_events)
        retained_event_count = len(events)
        sample_id = f"{item['scene']}/{item['frame_id']:06d}"
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventAid-R",
                "scene": item["scene"],
                "sequence_index": item["sequence_index"],
                "source": str(item["path"]),
                "t0_us": item["t0_us"],
                "t1_us": item["t1_us"],
                "dt_us": (item["t1_us"] - item["t0_us"])
                if item["t0_us"] is not None and item["t1_us"] is not None
                else None,
                "raw_event_count": raw_event_count,
                "cropped_event_count": cropped_event_count,
                "retained_event_count": retained_event_count,
                "dataset_sampling_ratio": dataset_sampling_ratio,
                "event_timestamp_diagnostics": timestamp_diagnostics,
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
            except OSError:
                pass
        handles.clear()

    def __del__(self) -> None:
        self.close()
