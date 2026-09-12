"""Synthetic CPU-only scan loading tests; no production dataset or GPU runs."""

from __future__ import annotations

import copy
import os

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from asgcn_unet import stream_scan_loader as loading
from asgcn_unet.stream_input import PHYSICAL_EVENT_TIME_CONTRACT


class TinyPhysicalDataset(Dataset):
    event_time_contract = PHYSICAL_EVENT_TIME_CONTRACT
    target_channels = 1
    crop_size = None

    def __init__(self):
        self.samples = [
            {"start_idx": index * 3, "end_idx": index * 3 + index % 3,
             "sensor_size": (2, 3), "sequence_index": index, "scene": str(index % 2)}
            for index in range(6)
        ]
        self.accessed = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self.accessed.append(index)
        count = index % 3
        return {
            "events": torch.full((count, 4), float(index), dtype=torch.float64),
            "event_ids": torch.full((count, 2), index, dtype=torch.long),
            "target": torch.full((1, 2, 3), float(index), dtype=torch.float32),
            "sensor_size": (2, 3), "sample_id": str(index),
            "metadata": {"index": index, "worker_pid": os.getpid(),
                         "torch_threads": torch.get_num_threads(), "cuda_initialized": torch.cuda.is_initialized()},
        }


@pytest.fixture(autouse=True)
def single_cpu_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(loading, "collect_runtime_resources", lambda **kwargs: _resources())
    yield
    torch.set_num_threads(previous)


def _resources():
    return {
        "synthetic_cpu_test_only": True, "allocation_limits_verified": True,
        "cpu": {"effective_cpu_limit": 8},
        "memory": {"effective_available_bytes": 1024**3, "process_rss_bytes": 8 * 1024**2},
    }


def _config(workers=0):
    return {"batch_size": 2, "num_workers": workers, "persistent_workers": True,
            "prefetch_factor": 2, "pin_memory": False}


def _loader(dataset=None, *, start=0, workers=0, batches=None, **options):
    return loading.iter_scan_batches(
        TinyPhysicalDataset() if dataset is None else dataset,
        [[0, 3], [1, 4], [2, 5]] if batches is None else batches,
        start, torch.device("cpu"), _config(workers), seed=2026, resource_snapshot=_resources(), **options,
    )


def test_schedule_cursor_and_all_events_targets_are_preserved():
    dataset = TinyPhysicalDataset()
    with _loader(dataset, start=1) as iterator:
        assert iterator.plan["start_batch"] == 1 and iterator.plan["schedule_changed"] is False
        actual = list(iterator)
    assert [indices for indices, _, _ in actual] == [[1, 4], [2, 5]]
    assert dataset.accessed == [1, 4, 2, 5]
    for indices, batch, timing in actual:
        assert batch.event_counts == tuple(index % 3 for index in indices)
        for index, sample in zip(indices, batch, strict=True):
            assert torch.all(sample["events"] == index)
            assert torch.all(sample["event_ids"] == index)
            assert torch.all(sample["target"] == index)
        assert timing["prefetch_wait_ms"] >= 0 and timing["worker_decode_ms"] >= 0
        assert timing["cpu_collate_and_pin_ms"] >= 0 and timing["host_to_device_ms"] >= 0
        assert "not_additive" in timing["timing_scope"]
    assert iterator.closed and iterator.iterator is None


def test_exact_completed_cursor_does_not_start_a_dataloader(monkeypatch):
    monkeypatch.setattr(loading, "DataLoader", lambda **kwargs: pytest.fail("completed cursor cannot spawn"))
    with _loader(start=3, workers=4) as iterator:
        assert list(iterator) == []
        assert iterator.plan["planned_host_bytes"] == 0
        assert iterator.plan["workers_started"] is False


@pytest.mark.parametrize("schedule", [[[0, 3], [1, 4]], [[0, 3], [1, 4], [2, 2]],
                                      [[0, 3], [1, 4], [2, 8]], [[], [0, 1, 2, 3, 4, 5]],
                                      [[False, 3], [1, 4], [2, 5]]])
def test_partial_duplicate_or_rebatched_schedule_is_rejected(schedule):
    with pytest.raises(ValueError):
        _loader(batches=schedule)


@pytest.mark.parametrize("start", [-1, 4, True])
def test_invalid_cursor_is_not_silently_reset(start):
    with pytest.raises(ValueError):
        _loader(start=start)


def test_worker_prefetch_and_pinning_options_are_preserved_without_spawning():
    iterator = _loader(workers=3)
    assert iterator.plan["num_workers"] == 3
    assert iterator.plan["persistent_workers"] is True
    assert iterator.plan["prefetch_factor"] == 2 and iterator.plan["prefetch_slots"] == 6
    assert iterator.plan["multiprocessing_context"] == "spawn" and iterator.plan["pin_memory"] is False
    assert iterator.plan["worker_baseline_estimate_bytes"] == 3 * 8 * 1024**2
    assert iterator.plan["measured_peak_memory"] is False


def test_actual_cpu_allocation_refuses_oversubscription_without_reducing_workers():
    with pytest.raises(RuntimeError, match="no worker or batch size was reduced"):
        _loader(workers=8)


def test_input_memory_budget_refusal_happens_before_decode_or_spawn(monkeypatch):
    dataset = TinyPhysicalDataset()
    monkeypatch.setattr(loading, "DataLoader", lambda **kwargs: pytest.fail("cannot spawn above budget"))
    with pytest.raises(RuntimeError, match="No worker, prefetch, batch or data was reduced"):
        _loader(dataset, workers=2, memory_budget_mib=0.01)
    assert dataset.accessed == []


def test_live_ram_reserve_failure_never_advances_consumed_cursor(monkeypatch):
    with _loader() as iterator:
        constrained = _resources()
        constrained["memory"]["effective_available_bytes"] = 1
        monkeypatch.setattr(loading, "collect_runtime_resources", lambda **kwargs: constrained)
        with pytest.raises(RuntimeError, match="live RAM reserve"):
            next(iterator)
        assert iterator.cursor == 0


def test_no_cuda_is_queried_for_cpu_scan(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU scan must not initialize/query CUDA"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("CPU scan cannot synchronize CUDA"))
    with _loader() as iterator:
        assert len(list(iterator)) == 3


def test_full_resolution_hdf5_metadata_is_used_without_decoding_pixels(tmp_path):
    path = tmp_path / "synthetic.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images/frame", data=np.zeros((8, 9, 3), dtype=np.uint16))
    dataset = TinyPhysicalDataset()
    dataset.crop_size = (2, 3)
    for record in dataset.samples:
        record.pop("sensor_size")
        record.update(path=path, image_key="frame")
    bounds = loading._sample_memory_bounds(dataset)
    assert dataset.accessed == []
    # Even a cropped output retains full-resolution decoding conversion scratch.
    assert bounds[0][1] >= 8 * 9 * 3 * (4 * 4 + 2)


def test_missing_size_contract_fails_instead_of_sampling_a_small_frame():
    dataset = TinyPhysicalDataset()
    dataset.samples[2].pop("end_idx")
    with pytest.raises(ValueError, match="end_idx"):
        _loader(dataset)
    assert dataset.accessed == []


def test_normal_context_cleanup_closes_its_own_worker_even_on_consumer_error():
    dataset = TinyPhysicalDataset()
    with pytest.raises(RuntimeError, match="synthetic consumer failure"), _loader(dataset, workers=1) as iterator:
        workers = list(iterator.iterator._workers)
        indices, packed, _ = next(iterator)
        assert indices == [0, 3]
        assert packed[0]["metadata"]["worker_pid"] != os.getpid()
        assert packed[0]["metadata"]["torch_threads"] == 1
        assert packed[0]["metadata"]["cuda_initialized"] is False
        assert iterator.cursor == 1  # dispatched prefetch must not advance this cursor
        raise RuntimeError("synthetic consumer failure")
    assert all(not worker.is_alive() for worker in workers)
    assert iterator.closed
    assert dataset.accessed == []  # decoding occurred only in the spawned CPU process


@pytest.mark.parametrize("key,value", [("num_workers", True), ("persistent_workers", "true"),
                                       ("prefetch_factor", 0), ("pin_memory", "true")])
def test_loader_flags_are_validated_without_coercion(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError):
        loading.iter_scan_batches(TinyPhysicalDataset(), [[0, 3], [1, 4], [2, 5]], 0,
                                  "cpu", config, seed=2026, resource_snapshot=_resources())


def test_config_and_full_schedule_remain_unchanged():
    config, schedule = _config(), [[0, 3], [1, 4], [2, 5]]
    originals = copy.deepcopy((config, schedule))
    with loading.iter_scan_batches(TinyPhysicalDataset(), schedule, 1, "cpu", config,
                                   seed=2026, resource_snapshot=_resources()) as iterator:
        list(iterator)
    assert (config, schedule) == originals


def test_windows_unmeasured_job_limits_are_not_assumed_safe(monkeypatch):
    monkeypatch.setattr(loading.os, "name", "nt")
    snapshot = _resources()
    snapshot.pop("allocation_limits_verified")
    with pytest.raises(RuntimeError, match="Windows Job"):
        loading.iter_scan_batches(TinyPhysicalDataset(), [[0, 3], [1, 4], [2, 5]], 0,
                                  "cpu", _config(), seed=2026, resource_snapshot=snapshot)


@pytest.mark.parametrize("workers", [0, 1])
def test_loader_base_seed_does_not_consume_parent_torch_rng(workers):
    before = torch.random.get_rng_state().clone()
    with _loader(workers=workers) as iterator:
        assert iterator.plan["isolated_loader_seed"] == 2026
        assert len(list(iterator)) == 3
    assert torch.equal(torch.random.get_rng_state(), before)


def test_loader_seed_has_no_implicit_default():
    with pytest.raises(TypeError, match="seed"):
        loading.iter_scan_batches(TinyPhysicalDataset(), [[0, 3], [1, 4], [2, 5]], 0,
                                  "cpu", _config(), resource_snapshot=_resources())
