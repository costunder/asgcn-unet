from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .data import EventAidRZipDataset, EventHDRDataset
from .engine import benchmark, evaluate
from .losses import ReconstructionLoss
from .model import ASGCNReconstructor
from .utils import atomic_torch_save, save_json


def create_eventhdr_smoke(root: Path, frames: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "smoke_scene.h5"
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


def create_eventaid_smoke(root: Path, frames: int = 4) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "R-smoke.zip"
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
            t1 = t0 + 9_500
            rows = []
            for timestamp in np.linspace(t0, t1, 80, dtype=np.int64):
                rows.append(
                    f"{timestamp} {rng.integers(0, width)} {rng.integers(0, height)} "
                    f"{rng.integers(0, 2)}"
                )
            zf.writestr(f"event/{index:06d}.txt", "\n".join(rows) + "\n")
    return path


def run_smoke(workspace: Path) -> dict:
    if workspace.exists():
        shutil.rmtree(workspace)
    eventhdr_train = workspace / "EventHDR" / "train"
    eventhdr_eval = workspace / "EventHDR" / "eval"
    eventaid_root = workspace / "EventAid-R"
    create_eventhdr_smoke(eventhdr_train)
    create_eventhdr_smoke(eventhdr_eval)
    create_eventaid_smoke(eventaid_root)

    train_set = EventHDRDataset(
        eventhdr_train, max_events=64, crop_size=(32, 48), random_crop=False
    )
    external_set = EventAidRZipDataset(eventaid_root, max_events=64, target_offset=1)
    model_config = {
        "hidden_dim": 8,
        "graph_layers": 2,
        "causal_candidates": 4,
        "spatial_radius": 1.0,
        "temporal_radius": 1.0,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": True,
    }
    model = ASGCNReconstructor(**model_config)
    sample = train_set[0]
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    checkpoint_path = workspace / "smoke.pt"
    atomic_torch_save(
        {"epoch": 0, "model": model.state_dict(), "model_config": model_config}, checkpoint_path
    )
    eval_config = {
        "seed": 7,
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": str(eventaid_root),
            "target_channels": 1,
            "max_events": 64,
            "crop_size": None,
            "target_offset": 1,
            "tone_map": "none",
        },
        "model": model_config,
        "eval": {
            "num_workers": 0,
            "max_samples": 2,
            "save_predictions": 1,
            "output_dir": str(workspace / "evaluation"),
        },
    }
    evaluation = evaluate(eval_config, checkpoint_path)
    timing = benchmark(eval_config, checkpoint_path, warmup=1, steps=2)
    report = {
        "eventhdr_samples": len(train_set),
        "eventaid_samples": len(external_set),
        "prediction_shape": list(prediction.shape),
        "nodes": diagnostics["nodes"],
        "edges": diagnostics["edges"],
        "loss": float(loss.detach()),
        "evaluation_frames": evaluation["quality"]["frames"],
        "benchmark_frames": timing["frames"],
    }
    save_json(workspace / "smoke_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.workspace), indent=2))


if __name__ == "__main__":
    main()
