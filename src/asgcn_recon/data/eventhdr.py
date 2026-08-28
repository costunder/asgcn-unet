from __future__ import annotations

import os
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
        random_crop: bool = False,
        seed: int = 2026,
        allowed_files: list[str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        self.frame_stride = max(1, int(frame_stride))
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
        self.random_crop = random_crop
        self.seed = int(seed)
        self._handles: dict[Path, h5py.File] = {}
        self._owner_pid = os.getpid()
        discovered = sorted([*self.root.rglob("*.h5"), *self.root.rglob("*.hdf5")])
        if not discovered:
            raise FileNotFoundError(
                f"No EventHDR .h5/.hdf5 files found under {self.root}. "
                "Place the official files in this directory or update dataset.root."
            )
        self.files = discovered
        if allowed_files is not None:
            allowed = set(allowed_files)
            present = {path.name for path in discovered}
            missing = sorted(allowed - present)
            if missing:
                preview = ", ".join(missing[:8])
                suffix = " ..." if len(missing) > 8 else ""
                raise FileNotFoundError(
                    f"EventHDR split requires {len(allowed)} files but {len(missing)} are "
                    f"missing under {self.root}: {preview}{suffix}"
                )
            self.files = [path for path in self.files if path.name in allowed]
        self.samples = self._build_index()
        if not self.samples:
            raise RuntimeError(f"No valid EventHDR frames found under {self.root}")

    def _build_index(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for path in self.files:
            with h5py.File(path, "r") as h5:
                if "events" not in h5 or "images" not in h5:
                    continue
                image_keys = sorted(k for k in h5["images"] if k.startswith("image"))
                selected_start_idx = 0
                selected_start_timestamp: float | None = None
                for frame_index, key in enumerate(image_keys):
                    node = h5["images"][key]
                    end_idx = int(node.attrs.get("event_idx", selected_start_idx))
                    timestamp = float(node.attrs.get("timestamp", frame_index))
                    if (
                        frame_index % self.frame_stride == 0
                        and end_idx > selected_start_idx
                    ):
                        samples.append(
                            {
                                "path": path,
                                "image_key": key,
                                "start_idx": selected_start_idx,
                                "end_idx": end_idx,
                                "t0": selected_start_timestamp,
                                "timestamp": timestamp,
                            }
                        )
                        # With frame_stride > 1, aggregate every skipped event interval
                        # into the next selected output instead of silently discarding it.
                        selected_start_idx = end_idx
                        selected_start_timestamp = timestamp
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
        ps = normalize_polarity(np.asarray(h5["events/ps"][start:end]))
        events = np.column_stack((xs, ys, ts, ps))
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1e-9)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        image = np.asarray(h5["images"][item["image_key"]])
        target = image_array_to_tensor(
            image,
            self.target_channels,
            tone_map=self.tone_map,
            tone_map_mu=self.tone_map_mu,
        )
        height, width = target.shape[-2:]
        scene_seed = zlib.crc32(str(item["path"]).encode("utf-8"))
        rng = np.random.default_rng(self.seed + scene_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        events = stratified_subsample(events, self.max_events)
        sample_id = f"{item['path'].stem}/{item['image_key']}"
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
                "scene": item["path"].stem,
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
