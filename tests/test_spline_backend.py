"""Backend regressions; generated tensors are test fixtures, not experiment data."""

from __future__ import annotations

import ast
import builtins
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from asgcn_unet import ops
from asgcn_unet.graph import ASGCNEncoder, EventGraph, PaperSplineConv
from asgcn_unet.model import ASGCNUNet
from tests.test_spline_opt import _reference_sum


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _case(dtype, device="cpu", kind="sparse", noncontiguous=False):
    generator = torch.Generator().manual_seed(9021)
    nodes, kernels, channels = 7, 4, 5
    projected = torch.randn((kernels, nodes, channels), generator=generator).transpose(0, 1) * 0.2
    if not noncontiguous:
        projected = projected.contiguous()
    if kind == "empty":
        edges = torch.empty((2, 0), dtype=torch.long)
    elif kind == "repeated":
        edges = torch.tensor([[0, 0, 3, 3, 6], [1, 1, 1, 4, 1]]).repeat(1, 5)
    elif kind == "dense":
        edges = torch.cartesian_prod(torch.arange(nodes), torch.arange(nodes)).T
    else:
        edges = torch.randint(nodes, (2, 23), generator=generator)
    count = edges.shape[1]
    indices = torch.randint(kernels, (count * 2, 2), generator=generator)[::2]
    basis = torch.rand((2, count), generator=generator).T
    if noncontiguous:
        edge_storage = torch.empty((count, 4), dtype=torch.long)
        edge_storage[:, ::2] = edges.T
        source, destination = edge_storage[:, 0], edge_storage[:, 2]
    else:
        source, destination = edges.contiguous()
        indices, basis = indices.contiguous(), basis.contiguous()
    basis_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return (
        projected.to(device=device, dtype=dtype).requires_grad_(),
        source.to(device), destination.to(device), indices.to(device),
        basis.to(device=device, dtype=basis_dtype).requires_grad_(),
    )


def _tolerance(dtype):
    if dtype == torch.float64:
        return 1e-11
    if dtype == torch.float16:
        return 0.005
    if dtype == torch.bfloat16:
        return 0.035
    return 2e-5


def _parity(backend, dtype, kind, chunk_size, *, device="cpu", noncontiguous=False, output_dtype=None):
    projected, source, destination, indices, basis = _case(dtype, device, kind, noncontiguous)
    reference_projected = projected.detach().clone().requires_grad_()
    reference_basis = basis.detach().clone().requires_grad_()
    output_dtype = output_dtype or (torch.float64 if dtype == torch.float64 else torch.float32)
    actual = ops.weighted_spline_sum(
        projected, source, destination, indices, basis, chunk_size, output_dtype, backend=backend,
    )
    expected = _reference_sum(
        reference_projected, source, destination, indices, reference_basis, chunk_size, output_dtype,
    )
    # The eager oracle has no graph for an empty input; make its zero derivative explicit.
    expected = expected + (reference_projected.sum() * 0 + reference_basis.sum() * 0).to(output_dtype)
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.linspace(-0.8, 0.9, actual.numel() * 2, device=device, dtype=output_dtype)
    cotangent = cotangent.reshape(actual.shape[0] * 2, actual.shape[1])[::2]
    actual_gradients = torch.autograd.grad(actual, (projected, basis), cotangent)
    expected_gradients = torch.autograd.grad(expected, (reference_projected, reference_basis), cotangent)
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["empty", "sparse", "dense", "repeated"])
@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_portable_fused_forward_projected_and_basis_gradients(dtype, kind, chunk_size):
    _parity("torch_fused", dtype, kind, chunk_size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_portable_fused_noncontiguous_and_same_dtype_accumulation(dtype):
    _parity("torch_fused", dtype, "repeated", 7, noncontiguous=True, output_dtype=dtype)


@pytest.mark.parametrize("requires_projection,requires_basis", [(True, False), (False, True)])
def test_portable_fused_selective_gradients(requires_projection, requires_basis):
    projected, source, destination, indices, basis = _case(torch.float64)
    projected.requires_grad_(requires_projection)
    basis.requires_grad_(requires_basis)
    reference_projected = projected.detach().clone().requires_grad_(requires_projection)
    reference_basis = basis.detach().clone().requires_grad_(requires_basis)
    actual = ops.weighted_spline_sum(
        projected, source, destination, indices, basis, 7, torch.float64, backend="torch_fused",
    )
    expected = _reference_sum(reference_projected, source, destination, indices, reference_basis, 7, torch.float64)
    actual.sum().backward()
    expected.sum().backward()
    for observed, reference in ((projected, reference_projected), (basis, reference_basis)):
        assert (observed.grad is None) == (reference.grad is None)
        if observed.grad is not None:
            torch.testing.assert_close(observed.grad, reference.grad, atol=1e-11, rtol=1e-11)


def _gradchecks(backend, device="cpu"):
    torch.manual_seed(14)
    projected = torch.randn((3, 2, 2), device=device, dtype=torch.float64, requires_grad=True)
    source = torch.tensor([0, 0, 2, 1], device=device)
    destination = torch.tensor([1, 1, 0, 2], device=device)
    indices = torch.tensor([[0, 1], [1, 0], [1, 1], [0, 1]], device=device)
    basis = torch.rand((4, 2), device=device, dtype=torch.float64, requires_grad=True)

    def operation(projected, basis):
        return ops.weighted_spline_sum(projected, source, destination, indices, basis, 3, torch.float64, backend=backend)

    assert torch.autograd.gradcheck(operation, (projected, basis), atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(operation, (projected, basis), atol=1e-5, rtol=1e-4)


def test_portable_fused_gradcheck_and_double_backward():
    _gradchecks("torch_fused")


@pytest.mark.parametrize("basis_grad", [False, True])
def test_fused_backward_retains_topology_not_per_edge_channel_messages(basis_grad):
    projected, source, destination, indices, basis = _case(torch.float64, kind="dense")
    basis.requires_grad_(basis_grad)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda tensor: tensor):
        output = ops.weighted_spline_sum(projected, source, destination, indices, basis, 7, torch.float64, backend="torch_fused")
    assert len(saved) == 4 + int(basis_grad)
    assert all(tensor is expected for tensor, expected in zip(saved[:4], (source, destination, indices, basis)))
    if basis_grad:
        assert saved[-1] is projected
    output.sum().backward()
    assert projected.grad is not None


def test_portable_fused_halves_forward_scatter_dispatches():
    inputs = _case(torch.float32, kind="dense")
    counts = {}
    for backend in ("torch", "torch_fused"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            ops.weighted_spline_sum(*inputs, 7, torch.float32, backend=backend)
        counts[backend] = sum(event.count for event in profiler.key_averages() if event.key == "aten::index_add_")
    assert counts == {"torch": 14, "torch_fused": 7}


def test_default_and_portable_backend_never_import_optional_cuda_module(monkeypatch):
    monkeypatch.setattr(ops, "_triton_ops", lambda: pytest.fail("optional CUDA import on portable backend"))
    inputs = _case(torch.float32)
    for backend in ("torch", "torch_fused"):
        ops.weighted_spline_sum(*inputs, 7, torch.float32, backend=backend).sum().backward()


def test_native_backend_cpu_is_rejected_even_for_empty_graphs():
    inputs = _case(torch.float32, kind="empty")
    with pytest.raises(ValueError, match="requires CUDA"):
        ops.weighted_spline_sum(*inputs, 7, torch.float32, backend="triton")
    layer = PaperSplineConv(1, 2, spline_backend="triton")
    with pytest.raises(ValueError, match="requires CUDA"):
        layer.spline_aggregate(torch.empty((0, 1)), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)))
    graph = EventGraph(torch.empty((0, 1)), torch.empty((0, 3)), torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)))
    with pytest.raises(ValueError, match="requires CUDA"):
        ASGCNEncoder(2, 1, spline_backend="triton").forward_ann(graph)
    with pytest.raises(ValueError, match="requires CUDA"):
        ASGCNUNet(spline_backend="triton").forward_sample({"events": torch.empty((0, 4)), "sensor_size": (16, 16)})


def test_native_requires_supported_device_and_determinism_contract(monkeypatch):
    monkeypatch.setattr(ops, "_triton_ops", lambda: pytest.fail("invalid backend should fail before import"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    with pytest.raises(RuntimeError, match="SM80"):
        ops.require_spline_backend("triton", torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="deterministic"):
        ops.require_spline_backend("triton", torch.device("cuda:0"))


def test_native_missing_dependency_is_explicit_no_fallback(monkeypatch):
    real_import = builtins.__import__

    def missing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "ops_cuda" in fromlist:
            raise ModuleNotFoundError("No module named 'triton'", name="triton")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_import)
    with pytest.raises(RuntimeError, match="no backend fallback was applied"):
        ops._triton_ops()


def test_native_metadata_validation_does_not_dispatch_kernel(monkeypatch):
    # CPU stand-in exercises metadata validation, not CUDA kernel execution.
    monkeypatch.setattr(ops, "require_spline_backend", lambda backend, device: None)
    monkeypatch.setattr(ops, "_triton_ops", lambda: pytest.fail("invalid input reached kernel dispatch"))
    projected, source, destination, indices, basis = _case(torch.float32)
    with pytest.raises(TypeError, match="integer topology"):
        ops.weighted_spline_sum(projected, source.float(), destination, indices, basis, 7, torch.float32, backend="triton")
    with pytest.raises(TypeError, match="real floating"):
        ops.weighted_spline_sum(projected.long(), source, destination, indices, basis, 7, torch.float32, backend="triton")
    for bad_indices, bad_basis, bad_chunk in ((indices[:, :1], basis, 7), (indices, basis[:, :1], 7), (indices, basis, 0)):
        with pytest.raises(ValueError, match="shapes or chunk"):
            ops.weighted_spline_sum(projected, source, destination, bad_indices, bad_basis, bad_chunk, torch.float32, backend="triton")


def test_native_higher_order_backward_uses_differentiable_torch_formula(monkeypatch):
    def fake_native_forward(projected, source, destination, indices, basis, output, chunk):
        output.copy_(_reference_sum(projected, source, destination, indices, basis, chunk, output.dtype))

    stand_in = SimpleNamespace(
        spline_forward=fake_native_forward,
        spline_backward=lambda *args: pytest.fail("native backward cannot provide higher derivatives"),
    )
    monkeypatch.setattr(ops, "_triton_ops", lambda: stand_in)
    projected, source, destination, indices, basis = _case(torch.float64)
    output = ops._SplineSumTriton.apply(projected, source, destination, indices, basis, 7, torch.float64)
    first = torch.autograd.grad(output.sum(), projected, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), basis)[0]
    torch.testing.assert_close(second, torch.full_like(basis, projected.shape[2]), atol=0, rtol=0)


def test_native_kernels_do_not_specialize_on_node_count_and_always_assert_bounds():
    # This checks declarations even when the optional CUDA module cannot import.
    path = Path(ops.__file__).with_name("ops_cuda.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kernels = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.endswith("_kernel")]
    assert len(kernels) == 3
    for kernel in kernels:
        parameters = {argument.arg: argument.annotation for argument in kernel.args.args}
        assert parameters["N"] is None
        jit = kernel.decorator_list[0]
        options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in jit.keywords}
        assert options["debug"] is True
        assert {"N", "start"}.issubset(options["do_not_specialize"])
        if "edge_count" in parameters:
            assert "edge_count" in options["do_not_specialize"]
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "device_assert" for node in ast.walk(kernel))


def _require_native_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; native kernels have not been exercised on this host")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Native candidate requires NVIDIA SM80+")
    if importlib.util.find_spec("triton") is None:
        pytest.fail("CUDA native-backend tests require Triton; missing dependency is not a passing fallback")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["empty", "sparse", "dense", "repeated"])
@pytest.mark.parametrize("backend", ["torch_fused", "triton"])
def test_cuda_forward_projected_and_basis_gradients(backend, dtype, kind):
    if backend == "triton":
        _require_native_cuda()
    elif not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    _parity(backend, dtype, kind, 7, device="cuda", noncontiguous=True)


def test_cuda_native_gradcheck_and_higher_order_fallback():
    _require_native_cuda()
    _gradchecks("triton", "cuda")


@pytest.mark.parametrize("backend", ["torch_fused", "triton"])
def test_cuda_full_model_amp_training_step_matches_reference(backend):
    _require_native_cuda() if backend == "triton" else None
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from asgcn_unet.engine import _training_step
    from asgcn_unet.losses import ReconstructionLoss

    torch.manual_seed(2026)
    reference = ASGCNUNet().cuda().train()
    candidate = ASGCNUNet(spline_backend=backend).cuda().train()
    candidate.load_state_dict(reference.state_dict())
    index = torch.arange(96, device="cuda")
    sample = {
        "events": torch.stack((10 + index % 8, 10 + index // 8, index / 95, index % 2 * 2 - 1), dim=-1).float(),
        "sensor_size": (240, 320),
        "target": torch.linspace(0.1, 0.9, 240 * 320, device="cuda").reshape(1, 240, 320),
        "sample_id": "synthetic-regression/native-parity", "metadata": {},
    }
    criterion = ReconstructionLoss()
    results = []
    for model in (reference, candidate):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", init_scale=1.0)

        def closure(model=model):
            with torch.autocast("cuda", dtype=torch.float16):
                prediction, diagnostics = model.forward_sample(sample)
                loss, parts = criterion(prediction, sample["target"].unsqueeze(0))
            return loss, parts, (prediction, diagnostics)

        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            payload, losses, norm, amp = _training_step(
                model, optimizer, scaler, closure, optimizer_mode="adam", max_norm=1.0,
                epoch=0, step=0, sample_id=sample["sample_id"],
            )
        assert step.call_count == 1 and amp["retries"] == 0
        assert payload[1]["edges"] > 0
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
        results.append((payload[0].detach(), losses, norm))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0.005, rtol=0.005)
    assert results[0][1]["total"] == pytest.approx(results[1][1]["total"], abs=0.005, rel=0.005)
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        assert (expected.grad is None) == (actual.grad is None)
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=0.005, rtol=0.01)


def test_cuda_native_selected_device_and_nondefault_stream():
    _require_native_cuda()
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _parity("triton", torch.float32, "repeated", 7, device=device, noncontiguous=True)
    stream.synchronize()


def test_cuda_native_varying_graph_sizes_reuse_compiled_kernels(monkeypatch):
    _require_native_cuda()
    from asgcn_unet import ops_cuda

    def execute(nodes, edges):
        projected = torch.randn((nodes, 4, 5), device="cuda", requires_grad=True)
        source = torch.arange(edges, device="cuda") % nodes
        destination = (source + 1) % nodes
        indices = torch.stack((source % 4, (source + 1) % 4), dim=-1)
        basis = torch.full((edges, 2), 0.5, device="cuda", requires_grad=True)
        output = ops.weighted_spline_sum(projected, source, destination, indices, basis, 16, torch.float32, backend="triton")
        output.sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(output).all() and torch.isfinite(projected.grad).all()

    execute(7, 19)
    # Test-only instrumentation of the installed Triton JIT, never a production
    # dependency on its internal cache representation. All tensor strides/dtypes,
    # fixed K/C, and pointer alignment remain the same as in the warmup.
    for kernel in (ops_cuda._forward_kernel, ops_cuda._projected_backward_kernel, ops_cuda._basis_backward_kernel):
        monkeypatch.setattr(kernel, "_do_compile", lambda *args, **kwargs: pytest.fail("Changing graph N/E triggered a new compilation"))
    execute(8, 37)
    execute(13, 21)
    execute(31, 48)


@pytest.mark.parametrize("bad_index", ["source", "destination", "basis"])
def test_cuda_native_invalid_index_fails_even_when_environment_debug_is_disabled(bad_index):
    _require_native_cuda()
    # Device assertions can invalidate their CUDA context; isolate them from the
    # test runner, and require an error rather than accepting silently lost edges.
    script = f"""
import torch
from asgcn_unet.ops import weighted_spline_sum
p = torch.ones((2, 2, 3), device='cuda', requires_grad=True)
s = torch.tensor([0], device='cuda')
d = torch.tensor([1], device='cuda')
i = torch.tensor([[0, 1]], device='cuda')
b = torch.ones((1, 2), device='cuda')
{{'source': s, 'destination': d, 'basis': i}}[{bad_index!r}].fill_(7)
try:
    weighted_spline_sum(p, s, d, i, b, 7, torch.float32, backend='triton')
    torch.cuda.synchronize()
except RuntimeError as error:
    if 'device-side assert' not in str(error).lower():
        raise
    print('EXPECTED_DEVICE_ASSERT')
else:
    raise AssertionError('Invalid native indices were silently accepted')
"""
    environment = dict(os.environ, TRITON_DEBUG="0")
    result = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_DEVICE_ASSERT" in result.stdout
