"""Bounded CPU prefetch for an unchanged, resumable physical scan schedule.

Memory planning is conservative accounting, not OS memory isolation or a measured
peak. Decode/collate work may overlap consumer computation; wait time is separate.
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .batching import PackedSampleBatch, move_batch, pack_samples
from .resources import collect_runtime_resources
from .stream_input import PHYSICAL_EVENT_TIME_CONTRACT


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be measured, finite and positive")
    return value


def _cpu_tensors(value):
    if isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError("Scan dataset workers and collation must return CPU tensors only")
    if isinstance(value, dict):
        for item in value.values():
            _cpu_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _cpu_tensors(item)


class _TimedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        started = time.perf_counter()
        sample = self.dataset[index]
        _cpu_tensors(sample)
        elapsed = (time.perf_counter() - started) * 1000
        return index, sample, elapsed


@dataclass
class _TimedPacked:
    indices: list[int]
    packed: PackedSampleBatch
    decode_ms: float
    collate_ms: float
    pin_ms: float = 0.0

    def pin_memory(self):
        started = time.perf_counter()
        self.packed = self.packed.pin_memory()
        self.pin_ms = (time.perf_counter() - started) * 1000
        return self


def _collate_scan(samples):
    started = time.perf_counter()
    indices, records, elapsed = zip(*samples, strict=True)
    packed = pack_samples(list(records))
    return _TimedPacked(list(indices), packed, sum(elapsed), (time.perf_counter() - started) * 1000)


def _initialize_cpu_worker(worker_id):
    # Spawn, never fork a parent with an initialized CUDA runtime. Torch's worker
    # loop is CPU-only; cap intra-op threads, not the configured worker count.
    if torch.cuda.is_initialized():
        raise RuntimeError("A scan decoding worker unexpectedly inherited CUDA state")
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class _RemainingBatches:
    def __init__(self, batches, start):
        self.batches, self.start = batches, start

    def __iter__(self):
        for number in range(self.start, len(self.batches)):
            yield list(self.batches[number])

    def __len__(self):
        return len(self.batches) - self.start


def _sample_memory_bounds(dataset):
    """Bound full EventHDR intervals from indexed spans/image metadata, not crops.

    No event values or image pixels are read. 48 bytes/event are exact packed
    float64[4]+int64[2] tensor storage; 40 additional bytes allow Python arrival
    group entries. Transient decoding uses an explicitly conservative 256 bytes
    per raw event plus full-resolution target conversion buffers.
    """
    if getattr(dataset, "event_time_contract", None) != PHYSICAL_EVENT_TIME_CONTRACT:
        raise ValueError("Scan loader memory planning requires the physical EventHDR input contract")
    records = getattr(dataset, "samples", None)
    if not isinstance(records, list) or len(records) != len(dataset):
        raise ValueError("Scan loader requires complete indexed dataset.samples metadata")
    channels = _integer(getattr(dataset, "target_channels", None), "target_channels", 1)
    crop = getattr(dataset, "crop_size", None)
    bounds = []
    with ExitStack() as stack:
        handles = {}
        for record in records:
            start = _integer(record.get("start_idx"), "event start_idx")
            end = _integer(record.get("end_idx"), "event end_idx")
            if end < start:
                raise ValueError("Raw EventHDR interval ends before it starts")
            shape = record.get("sensor_size") or record.get("shape")
            if shape is None:
                path = Path(record["path"])
                if path not in handles:
                    handles[path] = stack.enter_context(h5py.File(path, "r"))
                image = handles[path]["images"][record["image_key"]]
                if not isinstance(image, h5py.Dataset) or image.ndim not in (2, 3):
                    raise ValueError("EventHDR memory planning requires HxW or HxWxC target metadata")
                shape = image.shape[:2]
                source_channels = image.shape[2] if image.ndim == 3 else 1
                native_bytes = image.dtype.itemsize * math.prod(image.shape)
            else:
                # Indexed geometry is also used by synthetic CPU tests; when
                # supplied it must describe the uncropped source dimensions.
                source_channels = _integer(record.get("source_channels", channels), "source_channels", 1)
                native_bytes = math.prod(shape) * source_channels * _integer(
                    record.get("source_dtype_bytes", 4), "source_dtype_bytes", 1)
            if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                raise ValueError("Source sensor_size must contain full height and width")
            height, width = (_integer(value, "source dimension", 1) for value in shape)
            cropped_height, cropped_width = height, width
            if crop is not None:
                if not isinstance(crop, (tuple, list)) or len(crop) != 2:
                    raise ValueError("crop_size must contain positive height and width")
                cropped_height = min(height, _integer(crop[0], "crop height", 1))
                cropped_width = min(width, _integer(crop[1], "crop width", 1))
            events = end - start
            target_bytes = 4 * channels * cropped_height * cropped_width
            full_target_bytes = 4 * max(channels, source_channels) * height * width
            metadata_bytes = 65536 + len(repr(record).encode("utf-8"))
            payload = events * 88 + target_bytes + metadata_bytes
            scratch = events * 256 + full_target_bytes * 4 + native_bytes + metadata_bytes
            bounds.append((payload, scratch))
    return bounds


def _make_plan(dataset, batches, start, device, train_config, resources, memory_budget_mib, reserve_memory_mib):
    workers = _integer(train_config.get("num_workers", 0), "train.num_workers")
    persistent = train_config.get("persistent_workers")
    if persistent is not None and type(persistent) is not bool:
        raise ValueError("train.persistent_workers must be a boolean or None")
    persistent = (True if persistent is None else persistent) if workers else False
    prefetch = train_config.get("prefetch_factor")
    if prefetch is not None:
        _integer(prefetch, "train.prefetch_factor", 1)
    prefetch = (2 if prefetch is None else prefetch) if workers else None
    pin = train_config.get("pin_memory", device.type == "cuda")
    if type(pin) is not bool or (pin and device.type != "cuda"):
        raise ValueError("train.pin_memory must be boolean and pinned loading requires the CUDA consumer")
    plan = {
        "schema": "asgcn_stream_scan_loader_plan_v1", "dataset_samples": len(dataset),
        "total_batches": len(batches), "start_batch": start, "remaining_batches": len(batches) - start,
        "num_workers": workers, "persistent_workers": persistent, "prefetch_factor": prefetch,
        "pin_memory": pin, "multiprocessing_context": "spawn" if workers else None,
        "worker_torch_threads": 1, "physical_batch_size": train_config["batch_size"],
        "schedule_changed": False, "cursor_scope": "consumed_batches_not_prefetched_dispatched_indices",
        "hard_memory_isolation": False, "measured_peak_memory": False,
    }
    if start == len(batches):
        return {**plan, "workers_started": False, "planned_host_bytes": 0, "resources": resources}
    if os.name == "nt" and resources.get("allocation_limits_verified") is not True:
        raise RuntimeError("Windows Job CPU/RAM allocation limits are not measured; scan loading was refused")
    cpu = _positive(resources.get("cpu", {}).get("effective_cpu_limit"), "effective CPU allocation")
    main_threads = torch.get_num_threads()
    if workers + main_threads > math.floor(cpu):
        raise RuntimeError(
            f"Scan input needs {workers} workers plus {main_threads} parent Torch threads, exceeding "
            f"the measured {cpu:g}-core allocation; no worker or batch size was reduced")
    memory = resources.get("memory", {})
    available = _positive(memory.get("effective_available_bytes"), "available RAM")
    rss = _positive(memory.get("process_rss_bytes"), "parent process RSS")
    budget = available / 2 if memory_budget_mib is None else _positive(memory_budget_mib, "memory_budget_mib") * 1024**2
    reserve = available / 4 if reserve_memory_mib is None else _positive(reserve_memory_mib, "reserve_memory_mib") * 1024**2
    bounds = _sample_memory_bounds(dataset)
    payload = max(sum(bounds[index][0] for index in batches[number]) for number in range(start, len(batches)))
    scratch = max(sum(bounds[index][1] for index in batches[number]) for number in range(start, len(batches)))
    slots = workers * prefetch if workers else 0
    copies = slots * (2 + int(pin)) + 2 + workers
    worker_baseline = workers * rss
    required = payload * copies + scratch * max(workers, 1) + worker_baseline
    plan.update({
        "resources": resources, "workers_started": False, "main_torch_threads": main_threads,
        "prefetch_slots": slots, "max_batch_payload_bound_bytes": payload,
        "max_batch_decode_scratch_estimate_bytes": scratch, "payload_copy_slots": copies,
        "worker_baseline_estimate_bytes": worker_baseline,
        "worker_baseline_scope": "conservative_parent_RSS_replication_not_measured_worker_RSS",
        "planned_host_bytes": required, "memory_budget_bytes": budget, "reserve_memory_bytes": reserve,
        "memory_scope": "input_queues_decode_collate_pin_and_worker_baselines_excludes_scanner_and_CUDA",
    })
    if required > budget or required + reserve > available:
        raise RuntimeError(
            f"Prefetched scan input plan requires {required / 1024**2:.1f} MiB with "
            f"{reserve / 1024**2:.1f} MiB reserve; budget={budget / 1024**2:.1f} MiB, "
            f"available={available / 1024**2:.1f} MiB. No worker, prefetch, batch or data was reduced")
    return plan


class ScanBatchIterator:
    """Use as a context manager; cleanup affects only this DataLoader's workers."""

    def __init__(self, dataset, batches, start_batch, device, train_config, *, seed, resource_snapshot=None,
                 memory_budget_mib=None, reserve_memory_mib=None):
        self.device = torch.device(device)
        _integer(seed, "scan loader seed")
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        physical = _integer(train_config.get("batch_size"), "train.batch_size", 1)
        self.batches = tuple(tuple(batch) for batch in batches)
        _integer(start_batch, "start_batch")
        if start_batch > len(self.batches):
            raise ValueError("start_batch exceeds the complete schedule")
        seen = set()
        for batch in self.batches:
            if not batch or len(batch) > physical:
                raise ValueError("Scan schedule contains an empty/oversized physical batch")
            for index in batch:
                _integer(index, "dataset index")
                if index >= len(dataset) or index in seen:
                    raise ValueError("Scan schedule has an out-of-range or duplicate dataset index")
                seen.add(index)
        if seen != set(range(len(dataset))):
            raise ValueError("Scan schedule must cover the entire dataset exactly once")
        resources = collect_runtime_resources(include_cuda=False) if resource_snapshot is None else resource_snapshot
        self.plan = _make_plan(dataset, self.batches, start_batch, self.device, train_config,
                               resources, memory_budget_mib, reserve_memory_mib)
        self.plan.update({
            "isolated_loader_seed": seed,
            "rng_scope": "private_loader_generator_only_dataset_random_operations_are_not_isolated",
        })
        self.dataset, self.start, self.cursor = dataset, start_batch, start_batch
        self.loader, self.iterator, self.closed = None, None, False

    def __enter__(self):
        if self.closed or self.iterator is not None:
            raise RuntimeError("Scan loader cannot be reopened or entered twice")
        if self.start == len(self.batches):
            self.iterator = iter(())
            return self
        options = {
            "dataset": _TimedDataset(self.dataset), "batch_sampler": _RemainingBatches(self.batches, self.start),
            "collate_fn": _collate_scan, "num_workers": self.plan["num_workers"],
            "pin_memory": self.plan["pin_memory"],
            "generator": self.generator,
        }
        if self.plan["num_workers"]:
            options.update(persistent_workers=self.plan["persistent_workers"],
                           prefetch_factor=self.plan["prefetch_factor"], multiprocessing_context="spawn",
                           worker_init_fn=_initialize_cpu_worker)
        self.loader = DataLoader(**options)
        self.iterator = iter(self.loader)
        self.plan["workers_started"] = bool(self.plan["num_workers"])
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None or self.closed:
            raise RuntimeError("Scan batches require an open context manager")
        started = time.perf_counter()
        value = next(self.iterator)
        loaded = time.perf_counter()
        expected = list(self.batches[self.cursor])
        if value.indices != expected:
            raise RuntimeError("DataLoader changed the committed scan schedule order")
        tensor_bytes = sum(tensor.numel() * tensor.element_size() for tensor in
                           (value.packed.events, value.packed.event_ids, value.packed.targets)
                           if tensor is not None)
        if tensor_bytes > self.plan["max_batch_payload_bound_bytes"]:
            raise RuntimeError("Decoded input exceeds its metadata-derived memory plan; no batch was committed")
        resources = collect_runtime_resources(include_cuda=False)
        available = _positive(resources.get("memory", {}).get("effective_available_bytes"), "live available RAM")
        if available < self.plan["reserve_memory_bytes"]:
            raise RuntimeError("Prefetched scan input crossed its live RAM reserve; no batch was committed")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transfer_started = time.perf_counter()
        packed = move_batch(value.packed, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        transferred = time.perf_counter()
        self.cursor += 1
        return expected, packed, {
            "execution": "bounded_cpu_dataloader_prefetch_then_one_packed_transfer",
            "producer_location": "worker_process" if self.plan["num_workers"] else "main_process",
            "actual_packed_tensor_bytes": tensor_bytes,
            "dataset_indices": expected, "physical_batch_size": len(packed),
            "event_counts": list(packed.event_counts), "events_shape": list(packed.events.shape),
            "target_shape": None if packed.targets is None else list(packed.targets.shape),
            "pin_memory": self.plan["pin_memory"], "non_blocking_transfer": self.device.type == "cuda",
            "dataset_loading_ms": value.decode_ms, "cpu_collate_and_pin_ms": value.collate_ms + value.pin_ms,
            "worker_decode_ms": value.decode_ms, "worker_collate_ms": value.collate_ms,
            "pin_memory_ms": value.pin_ms, "prefetch_wait_ms": (loaded - started) * 1000,
            "host_to_device_ms": (transferred - transfer_started) * 1000,
            "total_input_ms": (transferred - started) * 1000,
            "timing_scope": "consumer_wait_and_transfer_wall_time_worker_decode_collate_overlap_not_additive",
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Framework shutdown signals/joins only workers owned by this iterator;
        # never enumerate or signal user/parent/server processes.
        shutdown = getattr(self.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        self.iterator, self.loader = None, None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def iter_scan_batches(dataset, batches, start_batch, device, train_config, *, seed, **options):
    """Return a context-managed packed iterator with a pre-spawn resource plan."""
    return ScanBatchIterator(dataset, batches, start_batch, device, train_config, seed=seed, **options)
