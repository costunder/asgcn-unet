from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest
import torch

from asgcn_unet.graph import PaperSplineConv, linear_open_bspline_basis
from asgcn_unet.ops import weighted_spline_sum


def _reference_sum(
    projected: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    indices: torch.Tensor,
    basis: torch.Tensor,
    chunk_size: int | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Original eager implementation, retained only as an independent test oracle."""
    output = torch.zeros(
        (projected.shape[0], projected.shape[2]),
        device=projected.device,
        dtype=output_dtype,
    )
    edge_count = source.numel()
    if edge_count:
        chunk_size = edge_count if chunk_size is None else chunk_size
        for active_basis in range(2):
            for start in range(0, edge_count, chunk_size):
                stop = min(start + chunk_size, edge_count)
                messages = projected[source[start:stop], indices[start:stop, active_basis]]
                messages = messages * basis[start:stop, active_basis, None].to(messages.dtype)
                output.index_add_(0, destination[start:stop], messages.to(output.dtype))
    return output


def _reference_affine(
    layer: PaperSplineConv,
    x: torch.Tensor,
    edges: torch.Tensor,
    pseudo: torch.Tensor,
) -> torch.Tensor:
    source, destination = edges
    if source.numel():
        indices, basis = linear_open_bspline_basis(pseudo, layer.kernel_size)
        projected = torch.einsum("ni,kio->nko", x, layer.weight)
        output = _reference_sum(
            projected,
            source,
            destination,
            indices,
            basis,
            layer.edge_chunk_size,
            x.dtype,
        )
        degree = torch.bincount(destination, minlength=x.shape[0]).to(x.dtype).unsqueeze(-1)
        output = output / degree.clamp_min(1.0)
    else:
        output = x.new_zeros((x.shape[0], layer.out_channels))
    if layer.root is not None:
        output = output + x @ layer.root
    if layer.bias is not None:
        output = output + layer.bias
    return output


def _graph(kind: str, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(319)
    if kind == "empty":
        edges = torch.empty((2, 0), dtype=torch.long)
    elif kind == "dense":
        edges = torch.cartesian_prod(torch.arange(6), torch.arange(6)).T
    elif kind == "repeated":
        edges = torch.tensor([[0, 0, 2, 2, 2, 5], [1, 1, 1, 4, 4, 1]]).repeat(1, 3)
    else:
        edges = torch.randint(0, 6, (2, 19), generator=generator)
    pseudo = torch.rand((edges.shape[1], 1), generator=generator, dtype=dtype)
    return edges.to(device), pseudo.to(device)


def _assert_optional_close(
    actual: tuple[torch.Tensor | None, ...],
    expected: tuple[torch.Tensor | None, ...],
    *,
    atol: float,
    rtol: float,
) -> None:
    assert len(actual) == len(expected)
    for actual_gradient, expected_gradient in zip(actual, expected):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            torch.testing.assert_close(
                actual_gradient, expected_gradient, atol=atol, rtol=rtol
            )


def _affine_parity(
    *,
    dtype: torch.dtype,
    device: str,
    chunk_size: int | None,
    graph_kind: str,
    gradient_case: str = "all",
    autocast: bool = False,
    autocast_dtype: torch.dtype = torch.bfloat16,
    cached_basis: bool = False,
) -> None:
    generator = torch.Generator().manual_seed(777)
    layer = PaperSplineConv(4, 5, kernel_size=4, edge_chunk_size=chunk_size).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=dtype).to(device) * 0.2
            )
    parameter_grad = gradient_case in {"all", "parameters"}
    layer.requires_grad_(parameter_grad)
    reference_layer = deepcopy(layer)
    x = torch.randn((6, 4), generator=generator, dtype=dtype).to(device)
    x.requires_grad_(gradient_case in {"all", "input"})
    reference_x = x.detach().clone().requires_grad_(x.requires_grad)
    edges, pseudo = _graph(graph_kind, device, dtype)
    pseudo.requires_grad_(gradient_case in {"all", "basis"})
    reference_pseudo = pseudo.detach().clone().requires_grad_(pseudo.requires_grad)
    basis_cache = linear_open_bspline_basis(pseudo, layer.kernel_size) if cached_basis else None
    degree = torch.bincount(edges[1], minlength=x.shape[0]) if cached_basis else None
    with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast):
        actual = layer.affine(x, edges, pseudo, basis_cache, degree)
        expected = _reference_affine(reference_layer, reference_x, edges, reference_pseudo)
    # CPU preserves the original cast and accumulation order exactly. CUDA
    # index_add_ uses atomics, so repeated destinations are not bitwise stable.
    forward_tolerance = 2e-6 if device == "cuda" else 0
    torch.testing.assert_close(actual, expected, atol=forward_tolerance, rtol=forward_tolerance)
    active = [tensor for tensor in (x, pseudo, *layer.parameters()) if tensor.requires_grad]
    reference_active = [
        tensor
        for tensor in (reference_x, reference_pseudo, *reference_layer.parameters())
        if tensor.requires_grad
    ]
    cotangent = torch.randn(actual.shape, generator=generator, dtype=dtype).to(device)
    actual_grads = torch.autograd.grad(actual, active, cotangent, allow_unused=True)
    reference_grads = torch.autograd.grad(expected, reference_active, cotangent, allow_unused=True)
    if autocast and autocast_dtype == torch.bfloat16:
        # bfloat16 has seven fraction bits; fused accumulation can change rounding.
        atol, rtol = 0.025, 0.025
    elif autocast:
        atol, rtol = 0.003, 0.003
    elif dtype == torch.float64:
        atol, rtol = 1e-11, 1e-11
    else:
        atol, rtol = 2e-5, 2e-5
    _assert_optional_close(actual_grads, reference_grads, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("chunk_size", [1, 7, None])
@pytest.mark.parametrize("graph_kind", ["normal", "repeated", "dense", "empty"])
def test_affine_forward_and_all_gradients_match_eager(dtype, chunk_size, graph_kind):
    _affine_parity(dtype=dtype, device="cpu", chunk_size=chunk_size, graph_kind=graph_kind)


@pytest.mark.parametrize("gradient_case", ["parameters", "input", "basis"])
@pytest.mark.parametrize("cached_basis", [False, True])
def test_selective_gradient_inputs_and_cached_basis(gradient_case, cached_basis):
    _affine_parity(
        dtype=torch.float64,
        device="cpu",
        chunk_size=5,
        graph_kind="repeated",
        gradient_case=gradient_case,
        cached_basis=cached_basis,
    )


@pytest.mark.parametrize("gradient_case", ["all", "parameters", "input", "basis"])
@pytest.mark.parametrize("chunk_size", [1, None])
def test_cpu_bfloat16_autocast_forward_and_gradients(gradient_case, chunk_size):
    _affine_parity(
        dtype=torch.float32,
        device="cpu",
        chunk_size=chunk_size,
        graph_kind="repeated",
        gradient_case=gradient_case,
        autocast=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this test host")
@pytest.mark.parametrize("chunk_size", [1, 7, None])
def test_cuda_forward_and_gradients(chunk_size):
    _affine_parity(
        dtype=torch.float32,
        device="cuda",
        chunk_size=chunk_size,
        graph_kind="dense",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this test host")
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast_forward_and_gradients(autocast_dtype):
    if autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support bfloat16")
    _affine_parity(
        dtype=torch.float32,
        device="cuda",
        chunk_size=7,
        graph_kind="dense",
        autocast=True,
        autocast_dtype=autocast_dtype,
    )


@pytest.mark.parametrize("chunk_size", [1, None])
def test_projected_basis_function_gradcheck_and_double_backward(chunk_size):
    generator = torch.Generator().manual_seed(15)
    x = torch.randn((3, 2), generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn((3, 2, 2), generator=generator, dtype=torch.float64, requires_grad=True)
    # Stay away from knots, where the piecewise-linear spline derivative jumps.
    pseudo = torch.tensor([[0.13], [0.37], [0.61], [0.88]], dtype=torch.float64, requires_grad=True)
    source = torch.tensor([0, 0, 2, 1])
    destination = torch.tensor([1, 1, 0, 2])

    def aggregate(features, kernels, coordinates):
        indices, basis = linear_open_bspline_basis(coordinates, 3)
        projected = torch.einsum("ni,kio->nko", features, kernels)
        return weighted_spline_sum(
            projected,
            source,
            destination,
            indices,
            basis,
            source.numel() if chunk_size is None else chunk_size,
            features.dtype,
        )

    inputs = (x, weight, pseudo)
    assert torch.autograd.gradcheck(aggregate, inputs, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(aggregate, inputs, atol=1e-5, rtol=1e-4)


def test_operator_noncontiguous_inputs_and_basis_gradients():
    generator = torch.Generator().manual_seed(318)
    projected = torch.randn((4, 3, 5), generator=generator, dtype=torch.float64).transpose(0, 1)
    projected.requires_grad_(True)
    basis = torch.rand((18, 2), generator=generator, dtype=torch.float64)[::2].requires_grad_(True)
    indices = torch.randint(0, 4, (18, 2), generator=generator)[::2]
    edge_storage = torch.randint(0, 3, (9, 2), generator=generator)
    source, destination = edge_storage.T
    assert not projected.is_contiguous()
    assert not basis.is_contiguous()
    assert not source.is_contiguous()
    reference_projected = projected.detach().clone().requires_grad_(True)
    reference_basis = basis.detach().clone().requires_grad_(True)
    actual = weighted_spline_sum(
        projected, source, destination, indices, basis, 4, torch.float64
    )
    expected = _reference_sum(
        reference_projected, source, destination, indices, reference_basis, 4, torch.float64
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grads = torch.autograd.grad(actual.square().sum(), (projected, basis))
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (reference_projected, reference_basis)
    )
    _assert_optional_close(actual_grads, expected_grads, atol=1e-11, rtol=1e-11)


def _saved_storage(
    operation: Callable[..., torch.Tensor],
    projected: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    indices: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[set[tuple[int, ...]], int]:
    shapes: set[tuple[int, ...]] = set()
    storage_bytes: dict[tuple[str, int], int] = {}

    def pack(tensor):
        shapes.add(tuple(tensor.shape))
        storage = tensor.untyped_storage()
        storage_bytes[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = operation(projected, source, destination, indices, basis, 64, torch.float64)
    # Run backward too: merely declining to save required operands is not sufficient.
    output.sum().backward()
    return shapes, sum(storage_bytes.values())


@pytest.mark.parametrize("basis_grad", [False, True])
def test_training_does_not_save_per_edge_channel_messages(basis_grad):
    generator = torch.Generator().manual_seed(510)
    projected = torch.randn((32, 5, 11), generator=generator, dtype=torch.float64)
    source = torch.randint(0, 32, (257,), generator=generator)
    destination = torch.randint(0, 32, (257,), generator=generator)
    indices = torch.randint(0, 5, (257, 2), generator=generator)
    basis = torch.rand((257, 2), generator=generator, dtype=torch.float64).requires_grad_(basis_grad)
    reference_shapes, reference_bytes = _saved_storage(
        _reference_sum,
        projected.clone().requires_grad_(True),
        source,
        destination,
        indices,
        basis,
    )
    actual_shapes, actual_bytes = _saved_storage(
        weighted_spline_sum,
        projected.clone().requires_grad_(True),
        source,
        destination,
        indices,
        basis,
    )
    assert (64, 11) in reference_shapes
    assert (64, 11) not in actual_shapes
    assert (1, 11) not in actual_shapes
    assert actual_bytes < reference_bytes / 2
