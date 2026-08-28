from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def make_eventhdr(root: Path, frames: int = 4) -> Path:
    """Create the smallest useful EventHDR-shaped fixture under pytest tmp_path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test.h5"
    height, width = 32, 48
    events_per_frame = 96
    total = frames * events_per_frame
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as h5:
        h5.attrs["sensor_resolution"] = np.array([height, width], dtype=np.int32)
        h5.attrs["num_events"] = total
        h5.attrs["num_imgs"] = frames
        events = h5.create_group("events")
        events.create_dataset("xs", data=rng.integers(0, width, total, dtype=np.int16))
        events.create_dataset("ys", data=rng.integers(0, height, total, dtype=np.int16))
        events.create_dataset("ts", data=np.linspace(0.0, frames * 0.002, total))
        events.create_dataset("ps", data=rng.integers(0, 2, total, dtype=np.uint8))
        images = h5.create_group("images")
        images.attrs["num_images"] = frames
        yy, xx = np.mgrid[:height, :width]
        for index in range(frames):
            image = np.clip((xx + yy + index * 8) / (width + height + frames * 8), 0, 1)
            node = images.create_dataset(
                f"image{index:09d}", data=(image * 65535).astype(np.uint16)
            )
            node.attrs["event_idx"] = (index + 1) * events_per_frame
            node.attrs["timestamp"] = (index + 1) * 0.002
            node.attrs["size"] = [height, width]
            node.attrs["type"] = "hdr"
    return path


def _png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def make_eventaid(root: Path, frames: int = 4) -> Path:
    """Create the smallest useful EventAid-R-shaped fixture under pytest tmp_path."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-test.zip"
    height, width = 32, 48
    rng = np.random.default_rng(11)
    timestamps = [1_000_000 + index * 10_000 for index in range(frames)]
    yy, xx = np.mgrid[:height, :width]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("shape.txt", f"{width} {height}\n")
        zf.writestr("timestamps.txt", "\n".join(str(value) for value in timestamps) + "\n")
        for index in range(1, frames + 1):
            image = np.clip((xx + yy + index * 6) / (width + height + frames * 6), 0, 1)
            zf.writestr(f"gt/{index:06d}_img.png", _png_bytes(image * 255))
            t0 = timestamps[index - 1]
            rows = [
                (
                    f"{timestamp} {rng.integers(0, width)} {rng.integers(0, height)} "
                    f"{rng.integers(0, 2)}"
                )
                for timestamp in np.linspace(t0, t0 + 9_500, 80, dtype=np.int64)
            ]
            zf.writestr(f"event/{index:06d}.txt", "\n".join(rows) + "\n")
    return path
