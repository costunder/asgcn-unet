from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.cuda import DeferredCudaCallError

from scripts import check_env


class FakeCuda:
    """Model a physical NVML count that differs from the initialized CUDA count."""

    def __init__(
        self,
        *,
        available: bool = True,
        physical_count: int = 8,
        runtime_count: int = 1,
        fail_on: str | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.available = available
        self.physical_count = physical_count
        self.runtime_count = runtime_count
        self.fail_on = fail_on
        self.failure = failure
        self.initialized = False
        self.calls: list[str] = []

    def _record(self, operation: str) -> None:
        self.calls.append(operation)
        if self.fail_on == operation:
            assert self.failure is not None
            raise self.failure

    def is_available(self) -> bool:
        self._record("is_available")
        return self.available

    def init(self) -> None:
        self._record("init")
        self.initialized = True

    def device_count(self) -> int:
        self._record("device_count")
        return self.runtime_count if self.initialized else self.physical_count

    def get_device_properties(self, index: int) -> SimpleNamespace:
        self._record(f"properties:{index}")
        assert self.initialized, "CUDA initialization must precede properties"
        assert 0 <= index < self.runtime_count, "Invalid device id"
        return SimpleNamespace(
            name=f"Visible GPU {index}",
            total_memory=int((index + 1) * 10.25 * (1024**3)),
        )


def _install_cuda(monkeypatch: pytest.MonkeyPatch, cuda: FakeCuda) -> None:
    monkeypatch.setattr(
        check_env,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            __version__="2.13.0",
            version=SimpleNamespace(cuda="13.0"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90000)),
        ),
    )


def _set_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *arguments: str,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            *arguments,
        ],
    )


def test_cuda_inventory_uses_initialized_count_for_mig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=1)
    _install_cuda(monkeypatch, cuda)

    assert check_env._cuda_inventory() == (True, ["Visible GPU 0"], [10.25])
    assert cuda.calls == ["is_available", "init", "device_count", "properties:0"]


def test_cuda_inventory_unavailable_skips_physical_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = FakeCuda(available=False, physical_count=8)
    _install_cuda(monkeypatch, cuda)

    assert check_env._cuda_inventory() == (False, [], [])
    assert cuda.calls == ["is_available"]


def test_cuda_inventory_reports_all_runtime_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=3)
    _install_cuda(monkeypatch, cuda)

    assert check_env._cuda_inventory() == (
        True,
        ["Visible GPU 0", "Visible GPU 1", "Visible GPU 2"],
        [10.25, 20.5, 30.75],
    )
    assert cuda.calls == [
        "is_available",
        "init",
        "device_count",
        "properties:0",
        "properties:1",
        "properties:2",
    ]


def test_check_env_reports_runtime_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=1)
    _install_cuda(monkeypatch, cuda)
    _set_arguments(monkeypatch, tmp_path, "--require-cuda")

    check_env.main()

    report = json.loads(capsys.readouterr().out)
    assert report["cuda_available"] is True
    assert report["gpu_devices"] == ["Visible GPU 0"]
    assert report["gpu_memory_gib"] == [10.25]
    assert cuda.calls == ["is_available", "init", "device_count", "properties:0"]


@pytest.mark.parametrize("require_cuda", [False, True])
def test_check_env_unavailable_cuda_is_not_inventoried_or_accepted_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    require_cuda: bool,
) -> None:
    cuda = FakeCuda(available=False, physical_count=8)
    _install_cuda(monkeypatch, cuda)
    arguments = ["--require-cuda"] if require_cuda else []
    _set_arguments(monkeypatch, tmp_path, *arguments)

    if require_cuda:
        with pytest.raises(SystemExit, match="CUDA was required"):
            check_env.main()
    else:
        check_env.main()

    report = json.loads(capsys.readouterr().out)
    assert report["cuda_available"] is False
    assert report["gpu_devices"] == []
    assert report["gpu_memory_gib"] == []
    assert cuda.calls == ["is_available"]


@pytest.mark.parametrize(
    "failure_type", [AssertionError, RuntimeError, OSError, DeferredCudaCallError]
)
@pytest.mark.parametrize("fail_on", ["is_available", "init", "device_count", "properties:0"])
def test_check_env_cuda_failures_exit_without_accepting_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_type: type[Exception],
    fail_on: str,
) -> None:
    cuda = FakeCuda(fail_on=fail_on, failure=failure_type("simulated CUDA failure"))
    _install_cuda(monkeypatch, cuda)
    _set_arguments(monkeypatch, tmp_path, "--require-cuda")

    with pytest.raises(SystemExit) as error:
        check_env.main()

    assert "CUDA" in str(error.value)
    assert error.value.code not in (None, 0)
    assert error.value.__suppress_context__ is True
    assert error.value.__cause__ is None
    assert capsys.readouterr().out == ""
    assert cuda.calls[-1] == fail_on


def test_check_env_rejects_zero_devices_after_cuda_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=0)
    _install_cuda(monkeypatch, cuda)
    _set_arguments(monkeypatch, tmp_path, "--require-cuda")

    with pytest.raises(SystemExit) as error:
        check_env.main()

    assert "CUDA" in str(error.value)
    assert error.value.code not in (None, 0)
    assert error.value.__suppress_context__ is True
    assert capsys.readouterr().out == ""
    assert cuda.calls == ["is_available", "init", "device_count"]


@pytest.mark.parametrize("include_private", [False, True])
@pytest.mark.parametrize(
    "failure_type", [AssertionError, RuntimeError, OSError, DeferredCudaCallError]
)
def test_check_env_cuda_error_details_require_private_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    include_private: bool,
    failure_type: type[Exception],
) -> None:
    private_path = tmp_path / "private-environment" / "torch" / "cuda" / "__init__.py"
    private_message = f"Invalid device id at {private_path} on private-compute-node"
    cuda = FakeCuda(fail_on="properties:0", failure=failure_type(private_message))
    _install_cuda(monkeypatch, cuda)
    arguments = ["--require-cuda"]
    if include_private:
        arguments.append("--include-private-host-provenance")
    _set_arguments(monkeypatch, tmp_path, *arguments)

    with pytest.raises(SystemExit) as error:
        check_env.main()

    message = str(error.value)
    captured = capsys.readouterr()
    assert "CUDA" in message
    assert error.value.__suppress_context__ is True
    if include_private:
        assert private_message in message
    else:
        for private_value in (str(tmp_path), private_path.as_posix(), "private-compute-node"):
            assert private_value not in message + captured.out + captured.err
    assert captured.out == ""
