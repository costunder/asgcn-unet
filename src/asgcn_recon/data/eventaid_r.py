from __future__ import annotations

import io
import os
import re
import zipfile
import zlib
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
)

_EVENT_RE = re.compile(r"(?:^|/)event/(\d+)\.txt$", re.IGNORECASE)
_GT_RE = re.compile(r"(?:^|/)gt/(\d+)_img\.png$", re.IGNORECASE)


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
        random_crop: bool = False,
        seed: int = 2026,
    ) -> None:
        self.root = Path(root).expanduser()
        self.target_channels = int(target_channels)
        self.max_events = max_events
        self.crop_size = tuple(crop_size) if crop_size else None
        self.target_offset = int(target_offset)
        self.tone_map = tone_map
        self.tone_map_mu = float(tone_map_mu)
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
    def _read_shape(zf: zipfile.ZipFile, names: list[str]) -> tuple[int, int] | None:
        shape_name = next((name for name in names if name.lower().endswith("shape.txt")), None)
        if not shape_name:
            return None
        values = zf.read(shape_name).decode("utf-8", errors="replace").split()
        if len(values) < 2:
            return None
        width, height = int(values[0]), int(values[1])
        return height, width

    def _build_index(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        scene_info: dict[str, Any] = {}
        for path in self.zip_paths:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
                events = {int(m.group(1)): name for name in names if (m := _EVENT_RE.search(name))}
                targets = {int(m.group(1)): name for name in names if (m := _GT_RE.search(name))}
                shape = self._read_shape(zf, names)
                timestamps_name = next(
                    (name for name in names if name.lower().endswith("timestamps.txt")), None
                )
                timestamps: list[int] = []
                if timestamps_name:
                    timestamps = [
                        int(value)
                        for value in zf.read(timestamps_name).decode("utf-8").split()
                    ]
                scene = path.stem
                scene_info[scene] = {"shape": shape, "frames": len(targets), "events": len(events)}
                for event_id in sorted(events):
                    target_id = event_id + self.target_offset
                    if target_id not in targets:
                        continue
                    samples.append(
                        {
                            "path": path,
                            "scene": scene,
                            "frame_id": event_id,
                            "event_name": events[event_id],
                            "target_name": targets[target_id],
                            "shape": shape,
                            "t0_us": timestamps[event_id - 1]
                            if 0 <= event_id - 1 < len(timestamps)
                            else None,
                            "t1_us": timestamps[event_id]
                            if 0 <= event_id < len(timestamps)
                            else None,
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
    def _read_events(raw: bytes) -> np.ndarray:
        values = np.fromstring(raw.decode("ascii", errors="ignore"), sep=" ", dtype=np.float64)
        usable = (values.size // 4) * 4
        if usable == 0:
            return np.empty((0, 4), dtype=np.float32)
        rows = values[:usable].reshape(-1, 4)
        # Official text columns: timestamp, x, y, polarity.
        events = rows[:, [1, 2, 0, 3]]
        if len(events):
            time_span = max(float(events[-1, 2] - events[0, 2]), 1.0)
            events[:, 2] = (events[:, 2] - events[0, 2]) / time_span
        events = events.astype(np.float32, copy=False)
        events[:, 3] = normalize_polarity(events[:, 3])
        return events

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        zf = self._get_handle(item["path"])
        events = self._read_events(zf.read(item["event_name"]))
        with Image.open(io.BytesIO(zf.read(item["target_name"]))) as image:
            target = image_array_to_tensor(
                pil_to_array(image),
                self.target_channels,
                tone_map=self.tone_map,
                tone_map_mu=self.tone_map_mu,
            )
        height, width = target.shape[-2:]
        if item["shape"] and item["shape"] != (height, width):
            raise ValueError(
                f"{item['scene']} shape.txt={item['shape']} but target={(height, width)}"
            )
        scene_seed = zlib.crc32(item["scene"].encode("utf-8"))
        rng = np.random.default_rng(self.seed + scene_seed)
        crop = choose_crop(height, width, self.crop_size, self.random_crop, rng)
        target = target[:, crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]
        events = crop_events(events, crop)
        events = stratified_subsample(events, self.max_events)
        sample_id = f"{item['scene']}/{item['frame_id']:06d}"
        return make_sample(
            events,
            target,
            sample_id,
            (crop.height, crop.width),
            {
                "dataset": "EventAid-R",
                "scene": item["scene"],
                "source": str(item["path"]),
                "t0_us": item["t0_us"],
                "t1_us": item["t1_us"],
                "dt_us": (item["t1_us"] - item["t0_us"])
                if item["t0_us"] is not None and item["t1_us"] is not None
                else None,
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
