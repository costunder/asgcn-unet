from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asgcn_unet import resources


def test_cgroup_v2_ancestor_limits_and_zero_current_memory(tmp_path: Path) -> None:
    parent = tmp_path / "job"
    leaf = parent / "step"
    leaf.mkdir(parents=True)
    (parent / "cpu.max").write_text("250000 100000")
    (parent / "memory.max").write_text("4096")
    (parent / "memory.current").write_text("1024")
    (leaf / "cpu.max").write_text("max 100000")
    (leaf / "cpuset.cpus.effective").write_text("2-3,7")
    (leaf / "memory.max").write_text("8192")
    (leaf / "memory.current").write_text("0")

    report = resources._cgroup_resources(tmp_path, "0::/job/step")

    assert report["cpu_quota_cores"] == 2.5
    assert report["cpuset_cpu_count"] == 3
    assert report["memory_limit_bytes"] == 4096
    assert report["memory_headroom_bytes"] == 3072
    assert any(item["memory_current_bytes"] == 0 for item in report["measurements"])


def test_cgroup_v1_quota_and_unlimited_memory(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu,cpuacct" / "job"
    cpu.mkdir(parents=True)
    (cpu / "cpu.cfs_quota_us").write_text("150000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    memory = tmp_path / "memory" / "job"
    memory.mkdir(parents=True)
    (memory / "memory.limit_in_bytes").write_text("9223372036854771712")
    (memory / "memory.usage_in_bytes").write_text("256")

    report = resources._cgroup_resources(tmp_path, "4:cpu,cpuacct:/job\n7:memory:/job")

    assert report["cpu_quota_cores"] == 1.5
    assert report["memory_limit_bytes"] is None
    assert report["memory_headroom_bytes"] is None


def test_cgroup_namespace_membership_does_not_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    (tmp_path / "memory.max").write_text("64")
    report = resources._cgroup_resources(root, "0::/../")
    assert report["memory_limit_bytes"] is None
    assert report["available"] is False


def test_cgroup_zero_memory_limit_is_a_real_limit(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("0")
    (tmp_path / "memory.current").write_text("0")
    report = resources._cgroup_resources(tmp_path, "0::/")
    assert report["memory_limit_bytes"] == 0
    assert report["memory_headroom_bytes"] == 0


def test_resources_respect_affinity_scheduler_and_cgroup_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("CPU resource reporting must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setattr(resources.platform, "system", lambda: "Linux")
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.delenv("NSLOTS", raising=False)
    monkeypatch.delenv("NCPUS", raising=False)
    monkeypatch.setattr(resources, "_linux_memory", lambda _: (10000, 5000, 1000))
    monkeypatch.setattr(
        resources,
        "_cgroup_resources",
        lambda *_: {
            "cpu_quota_cores": 2.5,
            "cpuset_cpu_count": 6,
            "memory_limit_bytes": 4000,
            "memory_headroom_bytes": 1500,
            "measurements": [],
            "available": True,
        },
    )

    report = resources.collect_runtime_resources(include_cuda=False)

    assert report["cpu"]["effective_cpu_limit"] == 2.5
    assert report["memory"]["effective_limit_bytes"] == 4000
    assert report["memory"]["effective_available_bytes"] == 1500
    assert report["cpu"]["utilization_percent"] is None
    assert report["gpu"]["devices"] is None
    json.dumps(report, allow_nan=False)


def test_execution_report_counts_parameters_and_actual_loader_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parameter:
        def __init__(self, size: int, trainable: bool):
            self.size = size
            self.requires_grad = trainable

        def numel(self):
            return self.size

    model = SimpleNamespace(parameters=lambda: iter([Parameter(5, True), Parameter(3, False)]))
    monkeypatch.setattr(resources, "collect_runtime_resources", lambda **_: {"measured": True})
    config = {
        "device": "cuda",
        "model": {"hidden_dim": 64},
        "dataset": {"max_events": 8192, "crop_size": None},
        "train": {"batch_size": 16, "epochs": 40, "num_workers": 4},
    }
    report = resources.build_execution_report(
        model,
        config,
        phase="train",
        device="cuda",
        gradient_accumulation_steps=2,
        data_parallel_workers=3,
        dataset_size=100,
        used_samples=100,
        loader=SimpleNamespace(
            num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=2
        ),
    )
    assert report["model"]["total_parameters"] == 8
    assert report["model"]["trainable_parameters"] == 5
    assert report["batching"]["effective_batch_size"] == 96
    assert report["loader"]["num_workers"] == 8
    assert report["loader"]["pin_memory"] is True
    assert report["data"]["config"]["max_events"] == 8192
    assert report["data"]["used_ratio"] == 1
    assert report["data"]["sampling_ratio"] is None
    assert report["optimization_steps"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"physical_batch_size": 0},
        {"gradient_accumulation_steps": 0},
        {"data_parallel_workers": True},
        {"dataset_size": 5, "used_samples": 6},
    ],
)
def test_execution_report_rejects_invalid_counts(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        resources.build_execution_report(None, {}, phase="train", **kwargs)


def test_missing_linux_memory_fields_are_not_reported_as_zero(tmp_path: Path) -> None:
    (tmp_path / "self").mkdir()
    (tmp_path / "meminfo").write_text("MemTotal: 512 kB\n")
    assert resources._linux_memory(tmp_path) == (512 * 1024, None, None)


def test_resource_interval_measures_cpu_instead_of_guessing_utilization() -> None:
    before = {
        "monotonic_seconds": 10.0,
        "cpu": {"process_cpu_seconds": 0.25},
        "memory": {"process_rss_bytes": 4096},
    }
    after = {
        "monotonic_seconds": 12.0,
        "cpu": {"process_cpu_seconds": 3.25, "effective_cpu_limit": 2},
        "memory": {"process_rss_bytes": 8192},
    }
    usage = resources.summarize_resource_interval(before, after)
    assert usage["process_cpu_percent"] == 150
    assert usage["process_cpu_allocation_percent"] == 75
    assert usage["includes_child_processes"] is False
    assert usage["gpu_utilization_percent"] is None
    with pytest.raises(ValueError, match="increasing time"):
        resources.summarize_resource_interval(after, before)


@pytest.mark.parametrize("cpu_delta", [0.0, 0.03125])
def test_resource_interval_equal_clock_readings_preserve_counters_without_fake_rates(
    cpu_delta: float,
) -> None:
    before = {
        "monotonic_seconds": 760261.375,
        "cpu": {"process_cpu_seconds": 1.0},
        "memory": {"process_rss_bytes": 4096},
    }
    after = {
        "monotonic_seconds": 760261.375,
        "cpu": {"process_cpu_seconds": 1.0 + cpu_delta, "effective_cpu_limit": 2},
        "memory": {"process_rss_bytes": 8192},
    }
    usage = resources.summarize_resource_interval(before, after)
    assert usage["wall_seconds"] == 0.0
    assert usage["process_cpu_seconds"] == cpu_delta
    assert usage["process_cpu_percent"] is None
    assert usage["process_cpu_allocation_percent"] is None
    assert usage["process_cpu_utilization_status"] == "unavailable_zero_wall_interval"
    assert "clock resolution" in usage["process_cpu_utilization_note"]
    assert usage["rss_before_bytes"] == 4096
    assert usage["rss_after_bytes"] == 8192


@pytest.mark.parametrize(
    ("wall_delta", "cpu_delta"),
    [
        (-1.0, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, -1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (1.0, -1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_resource_interval_invalid_counters_still_fail(
    wall_delta: float, cpu_delta: float
) -> None:
    before = {
        "monotonic_seconds": 10.0,
        "cpu": {"process_cpu_seconds": 1.0},
        "memory": {},
    }
    after = {
        "monotonic_seconds": 10.0 + wall_delta,
        "cpu": {"process_cpu_seconds": 1.0 + cpu_delta, "effective_cpu_limit": 2},
        "memory": {},
    }
    with pytest.raises(ValueError, match="valid CPU counters"):
        resources.summarize_resource_interval(before, after)


def test_calibration_report_uses_resolved_batch_and_calibration_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "collect_runtime_resources", lambda **_: {})
    report = resources.build_execution_report(
        None,
        {
            "train": {"batch_size": 16, "num_workers": 4},
            "calibration": {"batch_size": "auto", "num_workers": 8},
        },
        phase="calibration",
        physical_batch_size=8,
    )
    assert report["batching"]["physical_batch_size"] == 8
    assert report["loader"]["num_workers"] == 8


def test_cuda_inventory_initializes_before_count_for_mig(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    calls = []

    class Cuda:
        def is_available(self):
            calls.append("available")
            return True

        def init(self):
            calls.append("init")

        def device_count(self):
            assert calls == ["available", "init"]
            calls.append("count")
            return 1

        def get_device_properties(self, index):
            assert index == 0
            return SimpleNamespace(name="A100 MIG 1g.10gb", total_memory=9728 * 1024**2)

        def memory_allocated(self, index):
            return 100

        memory_reserved = memory_allocated
        max_memory_allocated = memory_allocated
        max_memory_reserved = memory_allocated

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=Cuda()))
    report = resources.collect_runtime_resources(device="cuda", include_cuda=True)
    assert report["gpu"]["visible_device_count"] == 1
    assert report["gpu"]["devices"][0]["mig"] is True
    assert report["gpu"]["devices"][0]["total_memory_bytes"] == 9728 * 1024**2
