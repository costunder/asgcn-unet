"""CPU unit tests of allocation evidence; never initialize or allocate a GPU."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asgcn_unet import allocation

_CONTAINER_DEVICE_EVIDENCE = allocation._container_device_evidence


@pytest.fixture(autouse=True)
def no_real_container_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(allocation, "_container_device_evidence", lambda: None)


@pytest.mark.parametrize(
    "mask",
    [
        "4",
        "2,4",
        "GPU-8932f937",
        "GPU-8932f937-d72c",
        "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6",
        "MIG-8932f937-d72c-4106-c12f-20bd9faed9f6",
        "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/0",
    ],
)
def test_explicit_allocated_mask_is_preserved(mask: str) -> None:
    environment = {"CUDA_VISIBLE_DEVICES": mask}
    evidence = allocation.inspect_gpu_allocation(environment)
    assert evidence["verified"] is True
    assert evidence["source"] == "explicit_cuda_visible_devices"
    assert evidence["cuda_visible_devices"] == mask
    assert evidence["mask_modified"] is False
    assert evidence["ownership_verified"] is False
    assert environment == {"CUDA_VISIBLE_DEVICES": mask}


@pytest.mark.parametrize(
    "mask",
    [
        "",
        "-1",
        "all",
        "none",
        "4,",
        ",4",
        "4,4",
        "4,04",
        " 4",
        "4, 2",
        "GPU-",
        "GPU-12345678-",
        "GPU-12345678-1-1",
        "MIG-bogus",
    ],
)
def test_disabled_malformed_or_duplicate_masks_fail_closed(mask: str) -> None:
    evidence = allocation.inspect_gpu_allocation({"CUDA_VISIBLE_DEVICES": mask})
    assert evidence["verified"] is False
    assert "malformed" in evidence["reason"]


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"NVIDIA_VISIBLE_DEVICES": "all"},
        {"NVIDIA_VISIBLE_DEVICES": "4"},
        {"SLURM_JOB_ID": "123", "SLURM_GPUS": "1"},
        {"GPU_COUNT": "1", "GPU_NAME": "NVIDIA A100 MIG 1g.10gb"},
    ],
)
def test_counts_job_ids_mig_names_and_runtime_env_are_not_kernel_proof(
    environment: dict[str, str],
) -> None:
    assert allocation.inspect_gpu_allocation(environment)["verified"] is False


def test_verified_kernel_evidence_can_replace_unset_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        allocation,
        "_container_device_evidence",
        lambda: {
            "source": "kernel_cgroup_v1_device_whitelist",
            "kernel_device_restriction_verified": True,
        },
    )
    assert allocation.inspect_gpu_allocation({})["verified"] is True
    # An explicitly disabled/malformed CUDA mask cannot be bypassed by fallback.
    assert allocation.inspect_gpu_allocation({"CUDA_VISIBLE_DEVICES": "-1"})["verified"] is False


@pytest.mark.parametrize(
    ("whitelist", "expected"),
    [
        ("c 195:4 rwm\nc 195:255 rwm\nc 511:* rwm\n", True),
        ("a *:* rwm\n", False),
        ("c 195:* rwm\n", False),
        ("c *:4 rwm\n", False),
        ("c 195:4 r\n", False),
        ("c 195:4 r\nc 195:4 w\n", True),
        ("c 195:4 rw\nc 195:0 rw\n", False),
        ("garbage", False),
        ("", False),
    ],
)
def test_device_whitelist_must_restrict_exact_visible_gpu_nodes(
    whitelist: str, expected: bool
) -> None:
    assert allocation._restricted_gpu_whitelist(whitelist, {195: {4}}) is expected


def test_cgroup_membership_resolves_own_devices_controller(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    cgroup_root = tmp_path / "devices"
    cgroup_root.mkdir()
    # Fixture files describe a fake kernel view; no device/cgroup is changed.
    (proc / "self/cgroup").write_text("5:devices:/tenant/job\n", encoding="utf-8")
    (proc / "self/mountinfo").write_text(
        f"1 2 0:3 /tenant {cgroup_root.as_posix()} rw - cgroup cgroup rw,devices\n",
        encoding="utf-8",
    )
    assert allocation._device_cgroup(proc) == cgroup_root / "job"
    (proc / "self/cgroup").write_text("5:devices:/../other\n", encoding="utf-8")
    assert allocation._device_cgroup(proc) is None
    (proc / "self/cgroup").write_text("0::/tenant/job\n", encoding="utf-8")
    assert allocation._device_cgroup(proc) is None


def test_kernel_device_evidence_uses_restricted_whitelist_and_character_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup = tmp_path / "group"
    cgroup.mkdir()
    whitelist = cgroup / "devices.list"
    whitelist.write_text("c 195:4 rw\nc 195:255 rw\n", encoding="utf-8")
    monkeypatch.setattr(allocation, "_device_cgroup", lambda _: cgroup)
    monkeypatch.setattr(allocation, "_nvidia_nodes", lambda *_: {195: {4}})
    evidence = _CONTAINER_DEVICE_EVIDENCE(proc_root=tmp_path, dev_root=tmp_path)
    assert evidence is not None
    assert evidence["kernel_device_restriction_verified"] is True
    assert evidence["visible_gpu_device_nodes"] == 1
    assert str(tmp_path) not in str(evidence)
    whitelist.write_text("a *:* rwm\n", encoding="utf-8")
    assert _CONTAINER_DEVICE_EVIDENCE(proc_root=tmp_path, dev_root=tmp_path) is None


def test_gpu_node_inventory_rejects_regular_files_and_wrong_device_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "devices").write_text(
        "Character devices:\n195 nvidia-frontend\nBlock devices:\n", encoding="utf-8"
    )
    info = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=4)
    node = SimpleNamespace(name="nvidia4", stat=lambda: info)
    dev = SimpleNamespace(glob=lambda _: [node])
    # Mock device numbers for Windows CPU unit tests; no character device exists.
    monkeypatch.setattr(allocation.os, "major", lambda _: 195, raising=False)
    monkeypatch.setattr(allocation.os, "minor", lambda value: value, raising=False)
    assert allocation._nvidia_nodes(tmp_path, dev) == {195: {4}}
    info.st_mode = stat.S_IFREG
    assert allocation._nvidia_nodes(tmp_path, dev) is None
    info.st_mode = stat.S_IFCHR
    info.st_rdev = 0
    assert allocation._nvidia_nodes(tmp_path, dev) is None


def test_resource_cuda_report_rejects_unmasked_before_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asgcn_unet import resources

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: calls.append("query"))),
    )
    with pytest.raises(RuntimeError, match="allocation safety check"):
        resources.collect_runtime_resources(device="cuda", include_cuda=True)
    assert calls == []


def test_cpu_seed_does_not_probe_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from asgcn_unet import utils

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    recorded: list[int] = []
    monkeypatch.setattr(utils.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(utils.torch, "manual_seed", lambda seed: recorded.append(seed))

    def forbidden() -> bool:
        raise AssertionError("CPU seed initialization must not query CUDA")

    monkeypatch.setattr(utils.torch.cuda, "is_available", forbidden)
    utils.set_seed(9)
    assert recorded == [9]


@pytest.mark.parametrize("requested", ["auto", "cuda", "cuda:0"])
def test_device_resolution_rejects_unmasked_before_any_cuda_probe(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    from asgcn_unet import utils

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls: list[str] = []
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: calls.append("probe") or True)
    with pytest.raises(RuntimeError, match="allocation safety check"):
        utils.resolve_device(requested)
    assert calls == []
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_explicit_cpu_resolution_never_queries_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from asgcn_unet import utils

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def forbidden() -> bool:
        raise AssertionError("CPU diagnostics must not query CUDA")

    monkeypatch.setattr(utils.torch.cuda, "is_available", forbidden)
    assert str(utils.resolve_device("cpu")) == "cpu"


def test_allocation_inspection_imports_without_torch_site_packages() -> None:
    module_path = Path(allocation.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import importlib.util,sys; "
                "spec=importlib.util.spec_from_file_location('allocation',sys.argv[1]); "
                "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "assert module.inspect_gpu_allocation({'CUDA_VISIBLE_DEVICES':'4'})['verified']; "
                "assert 'torch' not in sys.modules"
            ),
            str(module_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
