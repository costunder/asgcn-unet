"""Explicit CPU unit diagnostics for inference scheduling; no real-data claims."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from asgcn_unet import inference_profile
from asgcn_unet.batching import SequenceBatchSampler, pack_samples
from asgcn_unet.inference_profile import _probe_plan, _run_trial, profile_inference_batches


class DiagnosticDataset(Dataset):
    def __init__(self):
        self.samples = [
            {"sequence_id": f"sequence-{stream}", "sequence_index": index,
             "sensor_size": (4, 6) if stream < 3 else (5, 6)}
            for stream in range(4) for index in range(5)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            "events": torch.tensor([[0., 0., 0., 1.], [1., 1., 1., -1.]]),
            "sensor_size": self.samples[index]["sensor_size"],
            "metadata": self.samples[index],
        }


def loader_factory(dataset, batch_sampler, num_workers):
    return DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=pack_samples
    )


def settings(**changes):
    return {
        "batch_candidates": [1, 2, 4], "worker_candidates": [0],
        "profile_warmup": 1, "profile_steps": 1, "profile_debug_cpu": True,
        "batch_probe_indices": [1], **changes,
    }


def test_profile_cpu_is_explicit_and_never_quality_eligible():
    model = nn.Identity().train()
    seen = []

    def run_batch(batch):
        assert batch.events.device.type == "cpu"
        assert not model.training
        seen.extend(sample["metadata"]["sequence_index"] for sample in batch)
        return model(batch.events).sum()

    result = profile_inference_batches(
        DiagnosticDataset(), model, torch.device("cpu"), section=settings(),
        run_batch=run_batch, loader_factory=loader_factory,
    )
    report = result["report"]
    assert report["debug_cpu"] and not report["cuda_measured"]
    assert report["report_eligible"] is False
    assert model.training
    assert report["dataset_size"] == 20
    assert 1 in report["probe_indices"] and report["mandatory_probe_indices"] == [1]
    assert set(seen) == {0, 1, 2, 4}
    assert 1 <= result["batch_size"] <= 3
    for trial in report["trials"]:
        assert trial["probe_indices"] == report["probe_indices"]
        assert trial["timing_includes_io"] and trial["timing_includes_host_to_device"]
        assert trial["measured_batches"] >= trial["requested_steps"]
        assert trial["samples_per_second"] > 0
        assert trial["events_per_second"] > 0
        assert trial["peak_allocated_bytes"] is None
        assert "memory" in trial["host_resources_after"]


def test_probe_batches_preserve_shapes_and_independent_streams():
    dataset = DiagnosticDataset()
    topology = SequenceBatchSampler(dataset, 8)
    probes = list(range(len(dataset)))
    batches = _probe_plan(topology, probes, 8, calibration=False)
    assert sorted(index for batch in batches for index in batch) == probes
    for batch in batches:
        records = [dataset.samples[index] for index in batch]
        assert len({record["sequence_id"] for record in records}) == len(batch)
        assert len({record["sensor_size"] for record in records}) == 1
    calibration = _probe_plan(topology, probes, 8, calibration=True)
    assert max(map(len, calibration)) == 8
    assert sorted(index for batch in calibration for index in batch) == probes


def test_cpu_production_auto_profile_is_rejected():
    with pytest.raises(RuntimeError, match="requires the allocated CUDA"):
        profile_inference_batches(
            DiagnosticDataset(), nn.Identity(), torch.device("cpu"), section={},
            run_batch=lambda batch: batch.events, loader_factory=loader_factory,
        )


def test_non_oom_errors_propagate_and_model_mode_is_restored():
    model = nn.Identity().train()

    def fail(batch):
        raise ValueError("real graph error")

    with pytest.raises(ValueError, match="real graph error"):
        profile_inference_batches(
            DiagnosticDataset(), model, torch.device("cpu"), section=settings(),
            run_batch=fail, loader_factory=loader_factory,
        )
    assert model.training


@pytest.mark.parametrize("override", [
    {"batch_candidates": [0]}, {"batch_candidates": [True]},
    {"batch_candidates": [2, 2]}, {"worker_candidates": [-1]},
    {"profile_steps": 0}, {"profile_memory_fraction": 1.1},
    {"batch_probe_indices": [20]}, {"batch_probe_indices": [-1]},
])
def test_invalid_profile_contract_fails(override):
    with pytest.raises(ValueError):
        profile_inference_batches(
            DiagnosticDataset(), nn.Identity(), torch.device("cpu"),
            section=settings(**override), run_batch=lambda batch: batch.events,
            loader_factory=loader_factory,
        )


def test_measured_batch_then_workers_selection_and_concurrency_limit(monkeypatch):
    monkeypatch.setattr(inference_profile, "_resources", lambda device: {
        "allocation": {"cpu": {"effective_cpu_limit": 4}},
    })
    measured = []

    def trial(dataset, device, **kwargs):
        batch = kwargs["requested_batch_size"]
        workers = kwargs["num_workers"]
        measured.append((batch, workers))
        rates = {(1, 4): 10., (2, 4): 20., (8, 4): 30., (8, 0): 25., (8, 2): 35.}
        return {
            "requested_batch_size": batch, "num_workers": workers, "status": "ok",
            "samples_per_second": rates[batch, workers],
            "actual_batch_sizes": list(map(len, kwargs["batches"])),
        }

    monkeypatch.setattr(inference_profile, "_run_trial", trial)
    result = profile_inference_batches(
        DiagnosticDataset(), nn.Identity(), torch.device("cpu"),
        section=settings(batch_candidates=[1, 2, 8], worker_candidates=[0, 2, 4, 8]),
        run_batch=lambda batch: batch.events, loader_factory=loader_factory,
    )
    assert measured == [(1, 4), (2, 4), (8, 4), (8, 0), (8, 2)]
    assert result["batch_size"] == 3 and result["num_workers"] == 2
    assert result["report"]["selected"]["concurrency_limited"]
    assert result["report"]["skipped_workers_outside_cpu_allocation"] == [8]


@pytest.mark.parametrize("oom,peak,status", [
    (False, 700, "ok"), (False, 750, "memory_margin_exceeded"),
    (True, 0, "cuda_out_of_memory"),
])
def test_mock_cuda_memory_margin_and_oom_are_explicit(monkeypatch, oom, peak, status):
    # Every CUDA API is mocked. This tests accounting on CPU without accessing
    # an accelerator or claiming that the synthetic candidate fits real VRAM.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (800, 1000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 90)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 500)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: peak)

    class MockDeviceBatch:
        event_counts = (1,)
        events = torch.zeros(1, 4)
        targets = None
        sensor_size = (1, 1)

        def __len__(self):
            return 1

        def to(self, device):
            return self

    def run_batch(batch):
        if oom:
            raise torch.cuda.OutOfMemoryError("unit-test simulated allocation failure")
        return batch.events

    result = _run_trial(
        object(), torch.device("cuda"), batches=[[0]], requested_batch_size=1,
        num_workers=0, warmup=1, steps=1, memory_fraction=0.8, run_batch=run_batch,
        loader_factory=lambda dataset, batches, workers: [MockDeviceBatch() for _ in batches],
    )
    assert result["status"] == status
    assert result["memory_budget_bytes"] == 740
    if oom:
        assert "unit-test simulated" in result["error"]


def test_loader_batch_size_mismatch_is_not_silently_measured():
    def mismatched_loader(dataset, batches, workers):
        return [pack_samples([dataset[0], dataset[1]]) for _ in batches]

    with pytest.raises(RuntimeError, match="changed the requested batch size"):
        profile_inference_batches(
            DiagnosticDataset(), nn.Identity(), torch.device("cpu"), section=settings(),
            run_batch=lambda batch: batch.events, loader_factory=mismatched_loader,
        )


def test_unallocated_cuda_profile_is_rejected_before_any_gpu_query(monkeypatch):
    # This ordering test never queries or initializes an actual CUDA device.
    calls = []

    def reject_allocation():
        calls.append("allocation")
        raise RuntimeError("diagnostic test: no allocation evidence")

    def forbidden_query(*args, **kwargs):
        pytest.fail("GPU resources must not be queried before allocation is established")

    monkeypatch.setattr(inference_profile, "require_gpu_allocation", reject_allocation)
    monkeypatch.setattr(inference_profile, "_resources", forbidden_query)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden_query)
    with pytest.raises(RuntimeError, match="no allocation evidence"):
        profile_inference_batches(
            DiagnosticDataset(), nn.Identity(), torch.device("cuda"), section=settings(),
            run_batch=forbidden_query, loader_factory=forbidden_query,
        )
    assert calls == ["allocation"]
