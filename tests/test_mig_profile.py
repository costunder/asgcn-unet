from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from asgcn_unet import engine, preflight


class FakeCuda:
    """CPU-only model of NVML physical devices versus initialized MIG devices."""

    def __init__(
        self,
        *,
        available: bool = True,
        physical_count: int = 8,
        runtime_count: int = 1,
        current_index: int = 0,
        init_failure: Exception | None = None,
    ) -> None:
        self.available = available
        self.physical_count = physical_count
        self.runtime_count = runtime_count
        self.current_index = current_index
        self.init_failure = init_failure
        self.initialized = False
        self.calls: list[str] = []
        self.rng_states = [
            torch.tensor([index, 41], dtype=torch.uint8) for index in range(runtime_count)
        ]
        self.restored_states: list[torch.Tensor] | None = None

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return self.available

    def init(self) -> None:
        self.calls.append("init")
        if self.init_failure is not None:
            raise self.init_failure
        self.initialized = True

    def _lazy_init(self) -> None:
        if not self.initialized:
            self.init()

    def device_count(self) -> int:
        self.calls.append("device_count")
        return self.runtime_count if self.initialized else self.physical_count

    def current_device(self) -> int:
        self.calls.append("current_device")
        self._lazy_init()
        return self.current_index

    def get_device_properties(self, index: int) -> SimpleNamespace:
        self.calls.append(f"properties:{index}")
        self._lazy_init()
        assert 0 <= index < self.runtime_count, "Invalid device id"
        return SimpleNamespace(
            name=f"Runtime MIG {index}",
            major=8,
            minor=0,
            total_memory=(index + 1) * 10 * 1024**3,
            multi_processor_count=(index + 1) * 14,
        )

    def get_device_capability(self, index: int) -> tuple[int, int]:
        self.calls.append(f"capability:{index}")
        properties = self.get_device_properties(index)
        return properties.major, properties.minor

    def get_rng_state(self, index: int) -> torch.Tensor:
        self.calls.append(f"get_rng_state:{index}")
        self._lazy_init()
        assert 0 <= index < self.runtime_count, "Invalid device id"
        return self.rng_states[index].clone()

    def get_rng_state_all(self) -> list[torch.Tensor]:
        self.calls.append("get_rng_state_all")
        # Match PyTorch's range(device_count()) before get_rng_state lazy init.
        return [self.get_rng_state(index) for index in range(self.device_count())]

    def set_rng_state_all(self, states: list[torch.Tensor]) -> None:
        self.calls.append("set_rng_state_all")
        assert self.initialized
        assert len(states) == self.runtime_count
        self.restored_states = [state.clone() for state in states]


class FakeCudnn:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda

    def version(self) -> int:
        self.cuda.calls.append("cudnn.version")
        # PyTorch 2.13 checks visible device capabilities inside cuDNN init.
        # Capturing the pre-init NVML count here reproduces the MIG failure.
        for index in range(self.cuda.device_count()):
            self.cuda.get_device_capability(index)
        return 91002


def _install_preflight_cuda(monkeypatch: pytest.MonkeyPatch, cuda: FakeCuda) -> None:
    monkeypatch.setattr(
        preflight,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            __version__="2.13.0+cu126",
            version=SimpleNamespace(cuda="12.6"),
            backends=SimpleNamespace(cudnn=FakeCudnn(cuda)),
        ),
    )


def _install_engine_cuda(monkeypatch: pytest.MonkeyPatch, cuda: FakeCuda) -> None:
    monkeypatch.setattr(
        engine,
        "torch",
        SimpleNamespace(
            cuda=cuda,
            get_rng_state=torch.get_rng_state,
            set_rng_state=torch.set_rng_state,
            is_tensor=torch.is_tensor,
        ),
    )


def _cpu_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def _assert_cpu_rng_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    assert torch.equal(actual["torch"], expected["torch"])


@pytest.fixture(autouse=True)
def preserve_cpu_rng_streams():
    state = _cpu_rng_state()
    yield
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def test_fake_cudnn_reproduces_preinitialization_mig_failure() -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=1)
    with pytest.raises(AssertionError, match="Invalid device id"):
        FakeCudnn(cuda).version()
    assert cuda.calls.index("device_count") < cuda.calls.index("init")
    assert "capability:1" in cuda.calls


@pytest.mark.parametrize(
    ("requested", "runtime_count", "current_index", "expected_index"),
    [("cuda:0", 1, 0, 0), ("cuda:2", 3, 0, 2), ("cuda", 3, 2, 2)],
)
def test_profile_initializes_runtime_before_cudnn_device_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    runtime_count: int,
    current_index: int,
    expected_index: int,
) -> None:
    cuda = FakeCuda(runtime_count=runtime_count, current_index=current_index)
    _install_preflight_cuda(monkeypatch, cuda)
    before = _cpu_rng_state()

    runtime = preflight._runtime_provenance(torch.device(requested))

    assert cuda.calls.index("init") < cuda.calls.index("cudnn.version")
    assert cuda.calls.index("init") < cuda.calls.index("device_count")
    assert [call for call in cuda.calls if call.startswith("capability:")] == [
        f"capability:{index}" for index in range(runtime_count)
    ]
    assert runtime["requested_device"] == requested
    assert runtime["cuda_available"] is True
    assert runtime["cuda_runtime"] == "12.6"
    assert runtime["cudnn"] == 91002
    assert runtime["gpu"] == {
        "index": expected_index,
        "name": f"Runtime MIG {expected_index}",
        "compute_capability": [8, 0],
        "total_memory_mib": (expected_index + 1) * 10 * 1024,
        "multiprocessors": (expected_index + 1) * 14,
    }
    assert ("current_device" in cuda.calls) == (requested == "cuda")
    _assert_cpu_rng_equal(_cpu_rng_state(), before)


@pytest.mark.parametrize(
    ("requested", "available"), [("cpu", True), ("cpu", False), ("cuda", False)]
)
def test_profile_without_selected_available_cuda_does_not_initialize_cudnn(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    available: bool,
) -> None:
    cuda = FakeCuda(available=available, init_failure=RuntimeError("Must not initialize CUDA"))
    _install_preflight_cuda(monkeypatch, cuda)

    runtime = preflight._runtime_provenance(torch.device(requested))

    assert runtime["cuda_available"] is available
    assert runtime["cudnn"] is None
    assert runtime["gpu"] is None
    assert cuda.calls == ["is_available"]


def test_profile_does_not_hide_selected_invalid_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = FakeCuda(runtime_count=1)
    _install_preflight_cuda(monkeypatch, cuda)
    with pytest.raises(AssertionError, match="Invalid device id"):
        preflight._runtime_provenance(torch.device("cuda:1"))
    assert "properties:1" in cuda.calls


def test_profile_propagates_cuda_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = RuntimeError("CUDA driver initialization failed")
    cuda = FakeCuda(init_failure=failure)
    _install_preflight_cuda(monkeypatch, cuda)
    with pytest.raises(RuntimeError) as caught:
        preflight._runtime_provenance(torch.device("cuda:0"))
    assert caught.value is failure
    assert "cudnn.version" not in cuda.calls


def test_fake_rng_capture_reproduces_preinitialization_mig_failure() -> None:
    cuda = FakeCuda(physical_count=8, runtime_count=1)
    with pytest.raises(AssertionError, match="Invalid device id"):
        cuda.get_rng_state_all()
    assert cuda.calls.index("device_count") < cuda.calls.index("init")


@pytest.mark.parametrize("runtime_count", [1, 3])
def test_rng_capture_preserves_every_runtime_device_after_initialization(
    monkeypatch: pytest.MonkeyPatch, runtime_count: int
) -> None:
    cuda = FakeCuda(runtime_count=runtime_count)
    _install_engine_cuda(monkeypatch, cuda)
    before = _cpu_rng_state()

    state = engine._capture_rng_state()

    assert cuda.calls.index("init") < cuda.calls.index("get_rng_state_all")
    assert cuda.calls.index("init") < cuda.calls.index("device_count")
    assert len(state["cuda"]) == runtime_count
    for actual, expected in zip(state["cuda"], cuda.rng_states, strict=True):
        assert torch.equal(actual, expected)
    _assert_cpu_rng_equal(state, before)
    _assert_cpu_rng_equal(_cpu_rng_state(), before)


@pytest.mark.parametrize("runtime_count", [1, 3])
def test_rng_restore_validates_runtime_count_and_restores_every_visible_device(
    monkeypatch: pytest.MonkeyPatch, runtime_count: int
) -> None:
    cuda = FakeCuda(runtime_count=runtime_count)
    _install_engine_cuda(monkeypatch, cuda)
    state = _cpu_rng_state()
    state["cuda"] = [value.clone() for value in cuda.rng_states]
    expected_next = (random.random(), np.random.random(), torch.rand(3))

    engine._restore_rng_state(state)

    assert cuda.calls.index("init") < cuda.calls.index("device_count")
    assert cuda.restored_states is not None
    assert len(cuda.restored_states) == runtime_count
    for actual, expected in zip(cuda.restored_states, state["cuda"], strict=True):
        assert torch.equal(actual, expected)
    assert random.random() == expected_next[0]
    assert np.random.random() == expected_next[1]
    assert torch.equal(torch.rand(3), expected_next[2])


@pytest.mark.parametrize("operation", ["capture", "restore"])
def test_rng_operations_propagate_cuda_initialization_failure(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    failure = RuntimeError("CUDA driver initialization failed")
    cuda = FakeCuda(init_failure=failure)
    _install_engine_cuda(monkeypatch, cuda)
    before = _cpu_rng_state()
    state = {**before, "cuda": cuda.rng_states}

    with pytest.raises(RuntimeError) as caught:
        if operation == "capture":
            engine._capture_rng_state()
        else:
            engine._restore_rng_state(state)

    assert caught.value is failure
    assert "device_count" not in cuda.calls
    assert cuda.restored_states is None
    _assert_cpu_rng_equal(_cpu_rng_state(), before)


def test_cpu_rng_capture_restore_does_not_initialize_unavailable_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = FakeCuda(available=False, init_failure=RuntimeError("Must not initialize CUDA"))
    _install_engine_cuda(monkeypatch, cuda)
    state = engine._capture_rng_state()
    assert "cuda" not in state
    expected = (random.random(), np.random.random(), torch.rand(3))

    engine._restore_rng_state(state)

    assert set(cuda.calls) == {"is_available"}
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


@pytest.mark.parametrize(
    "cuda_states", [None, [], [torch.zeros(2, dtype=torch.uint8)] * 8, ["invalid"]]
)
def test_rng_restore_rejects_bad_cuda_state_without_changing_cpu_streams(
    monkeypatch: pytest.MonkeyPatch, cuda_states: Any
) -> None:
    cuda = FakeCuda(runtime_count=1)
    _install_engine_cuda(monkeypatch, cuda)
    state = _cpu_rng_state()
    state["cuda"] = cuda_states
    before = _cpu_rng_state()

    with pytest.raises(ValueError, match="CUDA"):
        engine._restore_rng_state(state)

    assert "init" in cuda.calls
    if isinstance(cuda_states, list):
        assert cuda.calls.index("init") < cuda.calls.index("device_count")
    assert cuda.restored_states is None
    _assert_cpu_rng_equal(_cpu_rng_state(), before)


@pytest.mark.parametrize(
    ("state", "exception", "message"),
    [
        (None, TypeError, "dictionary"),
        ({}, ValueError, "missing"),
        ({"python": None, "numpy": None, "torch": "invalid"}, ValueError, "must be a tensor"),
    ],
)
def test_rng_restore_retains_base_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
    state: Any,
    exception: type[Exception],
    message: str,
) -> None:
    cuda = FakeCuda()
    _install_engine_cuda(monkeypatch, cuda)
    before = _cpu_rng_state()
    with pytest.raises(exception, match=message):
        engine._restore_rng_state(state)
    assert not cuda.calls
    _assert_cpu_rng_equal(_cpu_rng_state(), before)
