from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import load_json, move_sample, resolve_device


def verify_sample(
    label: str,
    dataset,
    model: ASGCNReconstructor,
    device: torch.device,
) -> dict[str, object]:
    # The second frame has a real previous-output deadline where available.
    index = 1 if len(dataset) > 1 else 0
    sample = move_sample(dataset[index], device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        prediction, diagnostics = model.forward_sample(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1_000.0
    dt_us = sample["metadata"].get("dt_us")
    return {
        "dataset": label,
        "sample_id": sample["sample_id"],
        "dataset_samples": len(dataset),
        "events": int(sample["events"].shape[0]),
        "nodes": diagnostics["nodes"],
        "edges": diagnostics["edges"],
        "target_shape": list(sample["target"].shape),
        "prediction_shape": list(prediction.shape),
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "latency_ms_single_cpu_or_gpu_run": latency_ms,
        "dt_us": dt_us,
        "realtime_factor_single_run": (
            latency_ms / (float(dt_us) / 1_000.0) if dt_us else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the model on downloaded official samples")
    parser.add_argument("--eventhdr-root", default="data/EventHDR/train")
    parser.add_argument("--eventaid-root", default="data/EventAid-R")
    parser.add_argument("--model-config", default="configs/eventhdr_train.json")
    parser.add_argument("--max-events", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = load_json(args.model_config)
    model = ASGCNReconstructor(**config["model"]).to(device).eval()
    results: list[dict[str, object]] = []

    eventhdr_root = Path(args.eventhdr_root)
    if any(eventhdr_root.glob("*.h5")):
        dataset = EventHDRDataset(
            eventhdr_root,
            max_events=args.max_events,
            crop_size=(256, 256),
            random_crop=False,
        )
        results.append(verify_sample("EventHDR", dataset, model, device))
        dataset.close()

    eventaid_root = Path(args.eventaid_root)
    if any(eventaid_root.glob("R-*.zip")):
        dataset = EventAidRZipDataset(eventaid_root, max_events=args.max_events)
        results.append(verify_sample("EventAid-R", dataset, model, device))
        dataset.close()

    if not results:
        raise FileNotFoundError("No official EventHDR H5 or EventAid-R ZIP sample was found")
    print(json.dumps({"device": str(device), "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
