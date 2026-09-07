"""Small mocked resource-preflight tests; no model, GPU, or server execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from asgcn_unet import diagnostic_resources as resources


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _mount(path: Path) -> str:
    return path.as_posix().replace(" ", "\\040")


def _v2(tmp_path: Path):
    proc = tmp_path / "proc"
    mount = tmp_path / "group mount"
    leaf = mount / "job" / "task"
    leaf.mkdir(parents=True)
    _write(proc / "self/cgroup", "0::/job/task\n")
    _write(proc / "self/mountinfo", f"10 1 0:10 / {_mount(mount)} rw - cgroup2 cgroup rw\n")
    for directory, limit, usage, quota in (
        (mount, "max", "200", "max 100000"),
        (mount / "job", "1000", "700", "150000 100000"),
        (leaf, "max", "20", "max 100000"),
    ):
        _write(directory / "memory.max", limit)
        _write(directory / "memory.current", usage)
        _write(directory / "cpu.max", quota)
    return proc, mount, leaf


def _snapshot(*, headroom=500, cores=4):
    return {
        "headroom_bytes": headroom,
        "system": {"available_bytes": 700, "total_bytes": 1000, "process_rss_bytes": 100},
        "cpu": {"effective_cores": cores, "affinity_count": 8},
        "cgroup": {"memory_headroom_bytes": headroom},
    }


def test_v2_uses_actual_mount_and_all_ancestor_headrooms(tmp_path):
    proc, mount, leaf = _v2(tmp_path)
    report = resources._linux_cgroups(proc)
    assert report["memory_headroom_bytes"] == 300
    assert report["cpu_quota_cores"] == 1.5
    assert report["locations"]["unified"]["leaf"] == str(leaf.resolve())
    assert report["locations"]["unified"]["mount"] == str(mount.resolve())
    assert len(report["memory_measurements"]) == 3


def test_v2_true_root_without_limit_files_uses_host_memory(tmp_path):
    proc, mount, _ = _v2(tmp_path)
    (mount / "memory.max").unlink()
    (mount / "cpu.max").unlink()
    report = resources._linux_cgroups(proc)
    assert report["memory_headroom_bytes"] == 300
    assert report["memory_measurements"][-1]["status"].startswith("v2_hierarchy_root")
    assert report["memory_measurements"][-1]["current_bytes"] is None


def test_missing_nonroot_memory_measurement_fails_closed(tmp_path):
    proc, _, leaf = _v2(tmp_path)
    (leaf / "memory.current").unlink()
    with pytest.raises(resources.DiagnosticResourceError, match="required resource file"):
        resources._linux_cgroups(proc)


@pytest.mark.parametrize("membership", ["0::/../job", "0::relative", "0::/job/../../task"])
def test_membership_traversal_is_not_guessed_or_normalized(tmp_path, membership):
    proc, _, _ = _v2(tmp_path)
    _write(proc / "self/cgroup", membership)
    with pytest.raises(resources.DiagnosticResourceError, match="Unverifiable"):
        resources._linux_cgroups(proc)


def test_hidden_mount_ancestors_are_rejected(tmp_path):
    proc, mount, _ = _v2(tmp_path)
    _write(proc / "self/cgroup", "0::/hidden/job/task")
    _write(proc / "self/mountinfo", f"10 1 0:10 /hidden {_mount(mount)} rw - cgroup2 cgroup rw")
    with pytest.raises(resources.DiagnosticResourceError, match="ancestors are hidden"):
        resources._linux_cgroups(proc)


def test_wrong_mount_does_not_fall_back_to_sys_cgroup_root(tmp_path):
    proc, mount, _ = _v2(tmp_path)
    _write(proc / "self/mountinfo", f"10 1 0:10 /other {_mount(mount)} rw - cgroup2 cgroup rw")
    with pytest.raises(resources.DiagnosticResourceError, match="Cannot resolve actual"):
        resources._linux_cgroups(proc)


def test_ambiguous_membership_fails(tmp_path):
    proc, _, _ = _v2(tmp_path)
    _write(proc / "self/cgroup", "0::/job/task\n0::/job")
    with pytest.raises(resources.DiagnosticResourceError, match="Ambiguous"):
        resources._linux_cgroups(proc)


def test_usage_above_limit_means_zero_remaining_not_negative(tmp_path):
    proc, mount, _ = _v2(tmp_path)
    _write(mount / "job/memory.current", "1100")
    assert resources._linux_cgroups(proc)["memory_headroom_bytes"] == 0


@pytest.mark.parametrize("empty_unified_hierarchy", [False, True])
def test_v1_separate_mounts_and_unlimited_sentinel(tmp_path, empty_unified_hierarchy):
    proc, memory, cpu = tmp_path / "proc", tmp_path / "memory", tmp_path / "cpu"
    (memory / "task").mkdir(parents=True)
    (cpu / "task").mkdir(parents=True)
    _write(
        proc / "self/cgroup",
        "3:memory:/task\n2:cpu,cpuacct:/task"
        + ("\n0::/unused-v2" if empty_unified_hierarchy else ""),
    )
    _write(
        proc / "self/mountinfo",
        f"10 1 0:10 / {_mount(memory)} rw - cgroup cgroup rw,memory\n"
        f"11 1 0:11 / {_mount(cpu)} rw - cgroup cgroup rw,cpu,cpuacct",
    )
    for directory, limit, usage in (
        (memory, "1000", "800"),
        (memory / "task", str(2**63 - 4096), "10"),
    ):
        _write(directory / "memory.limit_in_bytes", limit)
        _write(directory / "memory.usage_in_bytes", usage)
    for directory, quota in ((cpu, "200000"), (cpu / "task", "-1")):
        _write(directory / "cpu.cfs_quota_us", quota)
        _write(directory / "cpu.cfs_period_us", "100000")
    report = resources._linux_cgroups(proc)
    assert report["memory_headroom_bytes"] == 200
    assert report["cpu_quota_cores"] == 2
    assert report["memory_measurements"][0]["limit_bytes"] is None


def test_hybrid_uses_v2_memory_and_v1_cpu_without_reading_missing_v2_cpu(tmp_path):
    proc, mount, leaf = _v2(tmp_path)
    cpu = tmp_path / "v1-cpu"
    for directory in (cpu, cpu / "task"):
        _write(directory / "cpu.cfs_quota_us", "250000")
        _write(directory / "cpu.cfs_period_us", "100000")
    for directory in (mount, mount / "job", leaf):
        (directory / "cpu.max").unlink()
    _write(proc / "self/cgroup", "0::/job/task\n2:cpu,cpuacct:/task")
    original_mounts = (proc / "self/mountinfo").read_text(encoding="utf-8")
    _write(
        proc / "self/mountinfo",
        original_mounts + f"11 1 0:11 / {_mount(cpu)} rw - cgroup cgroup rw,cpu,cpuacct\n",
    )
    report = resources._linux_cgroups(proc)
    assert report["memory_headroom_bytes"] == 300
    assert report["cpu_quota_cores"] == 2.5


@pytest.mark.parametrize(
    "filename,value",
    [("cpu.max", "1 0"), ("cpu.max", "oops"), ("memory.max", "nan"), ("memory.current", "-2")],
)
def test_malformed_kernel_counter_fails_closed(tmp_path, filename, value):
    proc, _, leaf = _v2(tmp_path)
    _write(leaf / filename, value)
    with pytest.raises(resources.DiagnosticResourceError):
        resources._linux_cgroups(proc)


def test_preflight_accepts_exact_budget_plus_reserve_boundary(monkeypatch):
    monkeypatch.setattr(resources, "_snapshot", lambda: _snapshot())
    report = resources.preflight(budget_bytes=400, reserve_bytes=100, cpu_threads=4)
    assert report["passed"] is True
    assert report["headroom_bytes"] == 500
    assert report["required_headroom_bytes"] == 500
    assert report["hard_memory_isolation"] is False
    assert report["resource_settings_modified"] is False
    assert report["gpu_queried"] is False


def test_budget_shortfall_refuses_before_any_processing(monkeypatch):
    monkeypatch.setattr(resources, "_snapshot", lambda: _snapshot())
    with pytest.raises(resources.DiagnosticResourceError, match="Insufficient measured RAM"):
        resources.preflight(budget_bytes=401, reserve_bytes=100, cpu_threads=4)


@pytest.mark.parametrize("cores,threads", [(4, 5), (1.5, 2)])
def test_threads_cannot_exceed_affinity_or_quota(monkeypatch, cores, threads):
    monkeypatch.setattr(resources, "_snapshot", lambda: _snapshot(cores=cores))
    with pytest.raises(resources.DiagnosticResourceError, match="exceeds measured allocation"):
        resources.preflight(budget_bytes=1, reserve_bytes=1, cpu_threads=threads)


def test_fractional_cpu_allocation_allows_one_os_thread(monkeypatch):
    monkeypatch.setattr(resources, "_snapshot", lambda: _snapshot(cores=0.5))
    report = resources.preflight(budget_bytes=1, reserve_bytes=1, cpu_threads=1)
    assert report["max_cpu_threads"] == 1


@pytest.mark.parametrize("name", ["budget_bytes", "reserve_bytes", "cpu_threads"])
@pytest.mark.parametrize("value", [0, -1, True, None, 1.5])
def test_explicit_positive_parameters_are_required(monkeypatch, name, value):
    def not_called():
        pytest.fail("Invalid input must fail before querying resources")

    monkeypatch.setattr(resources, "_snapshot", not_called)
    args = {"budget_bytes": 1, "reserve_bytes": 1, "cpu_threads": 1, name: value}
    with pytest.raises(resources.DiagnosticResourceError, match="explicit positive integer"):
        resources.preflight(**args)


def test_snapshot_linux_uses_minimum_host_and_cgroup_headroom(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Linux")
    monkeypatch.setattr(resources, "_linux_memory", lambda _: (1000, 600, 50))
    monkeypatch.setattr(
        resources,
        "_linux_cgroups",
        lambda _: {"memory_headroom_bytes": 300, "cpu_quota_cores": 3.5},
    )
    monkeypatch.setattr(resources.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "NCPUS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    report = resources._snapshot()
    assert report["headroom_bytes"] == 300
    assert report["cpu"]["effective_cores"] == 2
    assert report["cpu"]["affinity_count"] == 8


def test_windows_reads_ram_affinity_without_job_or_cuda(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Windows")
    monkeypatch.setattr(resources, "_windows_in_job", lambda: False)
    monkeypatch.setattr(resources, "_windows_memory", lambda: (1000, 600, 50))
    monkeypatch.setattr(resources, "_windows_affinity_count", lambda: 4)
    for key in ("SLURM_CPUS_PER_TASK", "NSLOTS", "NCPUS"):
        monkeypatch.delenv(key, raising=False)
    report = resources.preflight(budget_bytes=100, reserve_bytes=100, cpu_threads=4)
    assert report["headroom_bytes"] == 600
    assert report["system"]["platform"] == "Windows"
    assert report["gpu_queried"] is False


def test_unknown_windows_nested_job_limits_fail_closed(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Windows")
    monkeypatch.setattr(resources, "_windows_in_job", lambda: True)
    with pytest.raises(resources.DiagnosticResourceError, match="Job Object"):
        resources.preflight(budget_bytes=1, reserve_bytes=1, cpu_threads=1)


def test_unavailable_measurements_never_become_zero_or_fallback(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "Windows")
    monkeypatch.setattr(resources, "_windows_in_job", lambda: False)
    monkeypatch.setattr(resources, "_windows_memory", lambda: (1000, None, 50))
    monkeypatch.setattr(resources, "_windows_affinity_count", lambda: 4)
    with pytest.raises(resources.DiagnosticResourceError, match="Cannot measure current"):
        resources.preflight(budget_bytes=1, reserve_bytes=1, cpu_threads=1)


def test_unsupported_platform_has_no_silent_fallback(monkeypatch):
    monkeypatch.setattr(resources.platform, "system", lambda: "UnknownOS")
    with pytest.raises(resources.DiagnosticResourceError, match="not implemented"):
        resources.preflight(budget_bytes=1, reserve_bytes=1, cpu_threads=1)
