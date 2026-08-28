from __future__ import annotations

import copy
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_dataset, collate_samples
from .losses import ReconstructionLoss
from .metrics import MetricAccumulator, frame_metrics, percentile
from .model import ASGCNReconstructor
from .utils import (
    atomic_torch_save,
    move_sample,
    resolve_device,
    save_image,
    save_json,
    set_seed,
    write_frame_csv,
)


def build_model(config: dict[str, Any]) -> ASGCNReconstructor:
    return ASGCNReconstructor(**config)


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    fallback_model_config: dict[str, Any],
) -> tuple[ASGCNReconstructor, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    model_config = checkpoint.get("model_config", fallback_model_config)
    model = build_model(model_config).to(device)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _data_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
):
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_samples,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = (
            True if persistent_workers is None else bool(persistent_workers)
        )
        loader_options["worker_init_fn"] = _seed_worker
        if prefetch_factor is not None:
            if int(prefetch_factor) < 1:
                raise ValueError("prefetch_factor must be at least 1")
            loader_options["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_options)


def _loader_kwargs(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "persistent_workers": section.get("persistent_workers"),
        "prefetch_factor": section.get("prefetch_factor"),
    }


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # PyTorch before the unified torch.amp API.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _validation_dataset(config: dict[str, Any]):
    data_config = copy.deepcopy(config["dataset"])
    data_config["root"] = data_config.get("val_root", data_config["root"])
    return build_dataset(data_config, split="val")


@torch.no_grad()
def validate(
    model: ASGCNReconstructor,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    current_scene = None
    recurrent_state = None
    for index, batch in enumerate(loader):
        if max_samples is not None and index >= max_samples:
            break
        if len(batch) != 1:
            raise ValueError("Stateful validation currently requires batch_size=1")
        sample = move_sample(batch[0], device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        prediction, diagnostics = model.forward_sample(sample, recurrent_state=recurrent_state)
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0)
        accumulator.update(scene, sample["sample_id"], frame_metrics(prediction, target))
    return accumulator.summary()["micro"]


def train(
    config: dict[str, Any], resume_from: str | Path | None = None
) -> Path:
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    train_config = config["train"]
    data_config = copy.deepcopy(config["dataset"])
    train_dataset = build_dataset(data_config, split="train")
    val_dataset = _validation_dataset(config)
    batch_size = int(train_config.get("batch_size", 1))
    if batch_size != 1 and config["model"].get("recurrent", True):
        raise ValueError("The recurrent experiment uses chronological batch_size=1")
    train_loader = _data_loader(
        train_dataset,
        batch_size,
        int(train_config.get("num_workers", 0)),
        device,
        shuffle=False,
        **_loader_kwargs(train_config),
    )
    val_loader = _data_loader(
        val_dataset,
        1,
        int(train_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(train_config),
    )

    resume_path = resume_from or train_config.get("resume")
    resume_checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        model, resume_checkpoint = load_model_checkpoint(
            resume_path, device, config["model"]
        )
    else:
        model = build_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 2e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-6)),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    criterion = ReconstructionLoss(train_config.get("loss_weights"))
    temporal_weight = float(train_config.get("loss_weights", {}).get("temporal", 0.0))
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", config)

    best_ssim = float("-inf")
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        if "optimizer" not in resume_checkpoint:
            raise ValueError(
                f"Checkpoint {resume_path} has model weights but no optimizer state; "
                "it cannot be used for exact training resume"
            )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        _optimizer_to(optimizer, device)
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_ssim = float(
            resume_checkpoint.get(
                "best_ssim",
                resume_checkpoint.get("val", {}).get("ssim", float("-inf")),
            )
        )
        history = list(resume_checkpoint.get("history", []))
        _restore_rng_state(resume_checkpoint.get("rng_state"))

    epochs = int(train_config.get("epochs", 40))
    validate_every = max(1, int(train_config.get("validate_every", 1)))
    max_train_samples = train_config.get("max_train_samples")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        current_scene = None
        recurrent_state = None
        previous_prediction = None
        previous_target = None
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"train {epoch:03d}/{epochs:03d}")
        for step, batch in enumerate(progress):
            if max_train_samples is not None and seen >= int(max_train_samples):
                break
            if len(batch) != 1:
                raise ValueError("Stateful training currently requires batch_size=1")
            sample = move_sample(batch[0], device)
            scene = str(sample["metadata"].get("scene", "unknown"))
            if scene != current_scene:
                recurrent_state = None
                previous_prediction = None
                previous_target = None
                current_scene = scene
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction, diagnostics = model.forward_sample(
                    sample, recurrent_state=recurrent_state
                )
                target = sample["target"].unsqueeze(0)
                loss, loss_parts = criterion(prediction, target)
                if temporal_weight > 0 and previous_prediction is not None:
                    temporal = F.l1_loss(
                        prediction - previous_prediction,
                        target - previous_target,
                    )
                    loss = loss + temporal_weight * temporal
                    loss_parts["temporal"] = float(temporal.detach().cpu())
                    loss_parts["total"] = float(loss.detach().cpu())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()

            recurrent_state = diagnostics["recurrent_state"]
            if recurrent_state is not None:
                recurrent_state = recurrent_state.detach()
            previous_prediction = prediction.detach()
            previous_target = target.detach()
            running_loss += float(loss.detach().cpu())
            seen += 1
            if step % int(train_config.get("log_every", 20)) == 0:
                progress.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}", **loss_parts)

        should_validate = epoch % validate_every == 0 or epoch == epochs
        val_metrics = (
            validate(
                model,
                val_loader,
                device,
                max_samples=train_config.get("max_val_samples"),
            )
            if should_validate
            else {}
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val": val_metrics,
        }
        history.append(record)
        save_json(run_dir / "history.json", history)
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": (
                resume_checkpoint.get("model_config", config["model"])
                if resume_checkpoint is not None
                else config["model"]
            ),
            "config": config,
            "val": val_metrics,
            "best_ssim": best_ssim,
            "history": history,
            "rng_state": _capture_rng_state(),
        }
        if val_metrics.get("ssim", float("-inf")) > best_ssim:
            best_ssim = val_metrics["ssim"]
            checkpoint["best_ssim"] = best_ssim
            atomic_torch_save(checkpoint, run_dir / "best.pt")
        atomic_torch_save(checkpoint, run_dir / "last.pt")
        print(record)
    best_path = run_dir / "best.pt"
    if best_path.is_file():
        return best_path
    if start_epoch > epochs and resume_path is not None:
        return Path(resume_path)
    return run_dir / "last.pt"


def _maybe_lpips(enabled: bool, device: torch.device):
    if not enabled:
        return None
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS requested. Install with: pip install -e '.[eval]'") from exc
    return lpips.LPIPS(net="alex").to(device).eval()


@torch.no_grad()
def evaluate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 2026)))
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    eval_config = config.get("eval", {})
    loader = _data_loader(
        dataset,
        1,
        int(eval_config.get("num_workers", 0)),
        device,
        **_loader_kwargs(eval_config),
    )
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    lpips_model = _maybe_lpips(bool(eval_config.get("lpips", False)), device)
    accumulator = MetricAccumulator()
    frame_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    realtime_factors: list[float] = []
    current_scene = None
    recurrent_state = None
    output_dir = Path(eval_config.get("output_dir", "runs/evaluation"))
    save_limit = int(eval_config.get("save_predictions", 0))
    max_samples = eval_config.get("max_samples")
    saved = 0
    for index, batch in enumerate(tqdm(loader, desc=f"evaluate-{inference_mode}")):
        if max_samples is not None and index >= int(max_samples):
            break
        sample = move_sample(batch[0], device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        prediction, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        target = sample["target"].unsqueeze(0)
        metrics = frame_metrics(prediction, target, lpips_model)
        accumulator.update(scene, sample["sample_id"], metrics)
        dt_us = sample["metadata"].get("dt_us")
        rtf = latency_ms / (float(dt_us) / 1000.0) if dt_us else None
        if rtf is not None:
            realtime_factors.append(rtf)
        row = {
            "scene": scene,
            "sample_id": sample["sample_id"],
            **metrics,
            "latency_ms": latency_ms,
            "rtf": rtf,
            "events": int(sample["events"].shape[0]),
            "nodes": diagnostics["nodes"],
            "edges": diagnostics["edges"],
        }
        frame_rows.append(row)
        latencies.append(latency_ms)
        if saved < save_limit:
            safe_name = sample["sample_id"].replace("/", "_")
            save_image(output_dir / "predictions" / f"{safe_name}_pred.png", prediction)
            save_image(output_dir / "predictions" / f"{safe_name}_gt.png", target)
            saved += 1

    quality = accumulator.summary()
    latency = _latency_summary(latencies)
    latency["deadline_miss_ratio"] = (
        sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
        if realtime_factors
        else None
    )
    latency["rtf_p95"] = percentile(realtime_factors, 0.95) if realtime_factors else None
    result = {
        "dataset": config["dataset"]["type"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "quality": quality,
        "latency": latency,
    }
    save_json(output_dir / "metrics.json", result)
    write_frame_csv(output_dir / "frames.csv", frame_rows)
    return result


def _latency_summary(latencies: list[float]) -> dict[str, float | int | None]:
    if not latencies:
        return {"frames": 0}
    mean = statistics.fmean(latencies)
    return {
        "frames": len(latencies),
        "mean_ms": mean,
        "p50_ms": percentile(latencies, 0.50),
        "p90_ms": percentile(latencies, 0.90),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "max_ms": max(latencies),
        "fps": 1000.0 / mean,
    }


@torch.no_grad()
def benchmark(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    warmup: int = 10,
    steps: int = 100,
    inference_mode: str = "ann",
    simulation_steps: int = 16,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if inference_mode == "snn" and simulation_steps < 1:
        raise ValueError("simulation_steps must be at least 1 for SNN inference")
    device = resolve_device(config.get("device", "auto"))
    dataset = build_dataset(config["dataset"], split="eval")
    model, _ = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    cuda_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    cuda_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    latencies: list[float] = []
    event_counts: list[int] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []
    firing_rates: list[float] = []
    realtime_factors: list[float] = []
    recurrent_state = None
    current_scene = None
    total = warmup + steps
    for iteration in range(total):
        if iteration > 0 and iteration % len(dataset) == 0:
            recurrent_state = None
            current_scene = None
        if iteration == warmup and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        raw = dataset[iteration % len(dataset)]  # I/O intentionally outside the timer.
        sample = move_sample(raw, device)
        scene = str(sample["metadata"].get("scene", "unknown"))
        if scene != current_scene:
            recurrent_state = None
            current_scene = scene
        if cuda_start is not None:
            cuda_start.record()
        else:
            start = time.perf_counter()
        _, diagnostics = model.forward_sample(
            sample,
            inference_mode=inference_mode,
            simulation_steps=simulation_steps,
            recurrent_state=recurrent_state,
        )
        if cuda_end is not None:
            cuda_end.record()
            cuda_end.synchronize()
            elapsed_ms = float(cuda_start.elapsed_time(cuda_end))
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        recurrent_state = diagnostics["recurrent_state"]
        if recurrent_state is not None:
            recurrent_state = recurrent_state.detach()
        if iteration >= warmup:
            latencies.append(elapsed_ms)
            event_counts.append(int(sample["events"].shape[0]))
            node_counts.append(diagnostics["nodes"])
            edge_counts.append(diagnostics["edges"])
            firing_rates.extend(
                float(value.detach().cpu())
                if torch.is_tensor(value)
                else float(value)
                for value in diagnostics["firing_rates"]
            )
            dt_us = sample["metadata"].get("dt_us")
            if dt_us:
                realtime_factors.append(elapsed_ms / (float(dt_us) / 1000.0))
    result: dict[str, Any] = {
        **_latency_summary(latencies),
        "events_per_second": sum(event_counts) / (sum(latencies) / 1000.0),
        "mean_nodes": statistics.fmean(node_counts),
        "mean_edges": statistics.fmean(edge_counts),
        "mean_firing_rate": statistics.fmean(firing_rates) if firing_rates else None,
        "deadline_miss_ratio": (
            sum(value > 1.0 for value in realtime_factors) / len(realtime_factors)
            if realtime_factors
            else None
        ),
        "rtf_p95": percentile(realtime_factors, 0.95) if realtime_factors else None,
        "inference_mode": inference_mode,
        "simulation_steps": simulation_steps if inference_mode == "snn" else None,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "peak_gpu_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == "cuda" else None
        ),
        "timer": "cuda_event" if device.type == "cuda" else "perf_counter",
        "io_excluded": True,
    }
    return result


@torch.no_grad()
def calibrate(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    samples: int = 100,
) -> Path:
    device = resolve_device(config.get("device", "auto"))
    data_config = copy.deepcopy(config["dataset"])
    # Calibration is restricted to EventHDR train, never EventAid-R.
    if data_config["type"] != "eventhdr":
        raise ValueError("SNN calibration must use EventHDR training data")
    dataset = build_dataset(data_config, split="calibration")
    model, checkpoint = load_model_checkpoint(checkpoint_path, device, config["model"])
    model.eval()
    model.fold_batch_norm()
    model.encoder.reset_thresholds()
    for index in tqdm(range(min(samples, len(dataset))), desc="calibrate-SNN"):
        sample = move_sample(dataset[index], device)
        model.calibrate_sample(sample, momentum=-1.0)
    checkpoint["model"] = model.state_dict()
    checkpoint["snn_calibration_samples"] = min(samples, len(dataset))
    checkpoint["batch_norm_folded"] = True
    output_path = Path(output_path)
    atomic_torch_save(checkpoint, output_path)
    return output_path
