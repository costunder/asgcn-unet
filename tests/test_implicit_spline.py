"""Small CPU operator/autograd tests; not full-model or CUDA validation."""

import pytest
import torch

from asgcn_unet.graph import linear_open_bspline_basis
from asgcn_unet.implicit_radius import build_implicit_radius_graph
from asgcn_unet.implicit_spline import implicit_weighted_spline_sum
from asgcn_unet.ops import weighted_spline_sum


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fixture(dtype=torch.float64, count=11):
    generator = torch.Generator().manual_seed(902)
    positions = torch.rand((count, 4), generator=generator, dtype=dtype)
    batches = torch.arange(count) % 2
    graph = build_implicit_radius_graph(torch.zeros_like(positions), positions, batches,
                                        batch_size=3, radius=1., chunk_size=3, candidate_pair_budget=7)
    chunks = list(graph.iter_directed_neighbors())
    if chunks:
        source, destination, pseudo = (torch.cat(parts) for parts in zip(*chunks))
    else:
        source, destination = batches[:0], batches[:0]
        pseudo = positions.new_empty((0, 1))
    indices, basis = linear_open_bspline_basis(pseudo, 5)
    return graph, source, destination, indices, basis


def _reference(projected, fixture, selected=None, active=None, output_dtype=None, backend="torch"):
    _, source, destination, indices, basis = fixture
    if active is not None:
        keep = active[source]
        source, destination, indices, basis = source[keep], destination[keep], indices[keep], basis[keep]
    output = weighted_spline_sum(projected, source, destination, indices, basis, 4,
                                 projected.dtype if output_dtype is None else output_dtype, backend=backend)
    return output if selected is None else output[selected]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
@pytest.mark.parametrize("selection", ["all", "reordered", "empty"])
@pytest.mark.parametrize("mask_sources", [False, True])
def test_sum_and_projected_gradients_match_materialized(dtype, backend, selection, mask_sources):
    fixture = _fixture(dtype)
    graph = fixture[0]
    selected = None if selection == "all" else torch.tensor([8, 1, 4] if selection == "reordered" else [], dtype=torch.long)
    active = torch.arange(11) % 3 == 0 if mask_sources else None
    generator = torch.Generator().manual_seed(903)
    projected = torch.randn((11, 5, 4), generator=generator, dtype=dtype, requires_grad=True)
    reference_projected = projected.detach().clone().requires_grad_()
    actual = implicit_weighted_spline_sum(projected, graph, selected, active_sources=active,
                                          edge_chunk_size=3, backend=backend)
    expected = _reference(reference_projected, fixture, selected, active, backend=backend)
    tolerance = 2e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=dtype)
    (actual * cotangent).sum().backward()
    (expected * cotangent).sum().backward()
    torch.testing.assert_close(projected.grad, reference_projected.grad, rtol=tolerance, atol=tolerance)
    if active is not None:
        assert not bool(projected.grad[~active].any())


@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
def test_feature_kernel_root_bias_gradients_and_optimizer_update_match(backend):
    fixture = _fixture()
    graph = fixture[0]
    generator = torch.Generator().manual_seed(904)
    parameters = [torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_()
                  for shape in ((11, 4), (5, 4, 3), (4, 3), (3,))]
    reference_parameters = [value.detach().clone().requires_grad_() for value in parameters]
    target = torch.randn((11, 3), generator=generator, dtype=torch.float64)

    def forward(values, implicit):
        features, weight, root, bias = values
        projected = torch.einsum("ni,kio->nko", features, weight)
        summed = (implicit_weighted_spline_sum(projected, graph, backend=backend, edge_chunk_size=4)
                  if implicit else _reference(projected, fixture, backend=backend))
        affine = summed / graph.in_degree.clamp_min(1)[:, None] + features @ root + bias
        return torch.nn.functional.leaky_relu(affine, negative_slope=.1)

    output = forward(parameters, True)
    reference = forward(reference_parameters, False)
    torch.testing.assert_close(output, reference, rtol=1e-11, atol=1e-11)
    (output - target).square().mean().backward()
    (reference - target).square().mean().backward()
    for value, reference_value in zip(parameters, reference_parameters):
        assert value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool(value.grad.abs().sum() > 0)
        torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-11, atol=1e-11)
    torch.optim.SGD(parameters, lr=.02).step()
    torch.optim.SGD(reference_parameters, lr=.02).step()
    for value, reference_value in zip(parameters, reference_parameters):
        torch.testing.assert_close(value, reference_value, rtol=1e-11, atol=1e-11)


def test_saved_context_contains_only_node_geometry_and_optional_node_masks():
    graph = _fixture(count=15)[0]
    selected = torch.tensor([8, 2, 5])
    active = torch.arange(15) % 2 == 0
    projected = torch.ones((15, 5, 2), dtype=torch.float64, requires_grad=True)
    output = implicit_weighted_spline_sum(projected, graph, selected, active_sources=active)
    saved = output.grad_fn.saved_tensors
    assert len(saved) == 4
    assert [value.shape for value in saved] == [(15, 4), (15,), (3,), (15,)]
    assert saved[0] is graph.positions and saved[1] is graph.node_batch
    assert all(value is not projected for value in saved)
    output.sum().backward()


def test_backward_regenerates_neighbors_without_rebuilding_index(monkeypatch):
    graph = _fixture()[0]
    index = graph.index
    calls = []
    original = index.iter_directed_neighbors

    def neighbors(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(index, "iter_directed_neighbors", neighbors)
    projected = torch.ones((11, 5, 2), dtype=torch.float64, requires_grad=True)
    output = implicit_weighted_spline_sum(projected, graph)
    assert len(calls) == 1
    output.sum().backward()
    assert len(calls) == 2 and graph.index is index


def test_small_double_gradcheck_and_gradgradcheck():
    graph = _fixture(count=3)[0]
    projected = torch.randn((3, 5, 1), dtype=torch.float64, requires_grad=True)
    operation = lambda value: implicit_weighted_spline_sum(value, graph, edge_chunk_size=2)
    assert torch.autograd.gradcheck(operation, (projected,), fast_mode=True)
    assert torch.autograd.gradgradcheck(operation, (projected,), fast_mode=True)


@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_message_cast_boundaries_and_float32_accumulation(dtype, backend):
    fixture = _fixture()
    projected = torch.randn((11, 5, 3), dtype=dtype, requires_grad=True)
    clone = projected.detach().clone().requires_grad_()
    actual = implicit_weighted_spline_sum(projected, fixture[0], backend=backend, output_dtype=torch.float32)
    expected = _reference(clone, fixture, backend=backend, output_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    expected.sum().backward()
    tolerance = .02 if dtype == torch.bfloat16 else .003 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(projected.grad, clone.grad, rtol=tolerance, atol=tolerance)


def test_empty_graph_and_inference_mode():
    graph = _fixture(count=0)[0]
    projected = torch.empty((0, 5, 2), dtype=torch.float64, requires_grad=True)
    output = implicit_weighted_spline_sum(projected, graph)
    output.sum().backward()
    assert output.shape == (0, 2) and projected.grad.shape == projected.shape
    with torch.inference_mode():
        inference_graph = _fixture(count=3)[0]
        result = implicit_weighted_spline_sum(torch.ones((3, 5, 2), dtype=torch.float64), inference_graph)
        assert bool(torch.isfinite(result).all())


def test_backend_and_masks_are_not_silently_reinterpreted():
    graph = _fixture()[0]
    projected = torch.ones((11, 5, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="requires CUDA"):
        implicit_weighted_spline_sum(projected, graph, backend="triton")
    with pytest.raises(ValueError, match="Unknown spline_backend"):
        implicit_weighted_spline_sum(projected, graph, backend="auto")
    with pytest.raises(ValueError, match="active_sources"):
        implicit_weighted_spline_sum(projected, graph, active_sources=torch.ones(11))
    with pytest.raises(ValueError, match="unique"):
        implicit_weighted_spline_sum(projected, graph, torch.tensor([0, 0]))


def test_geometry_mutation_between_forward_and_backward_is_rejected():
    graph = _fixture()[0]
    projected = torch.ones((11, 5, 2), dtype=torch.float64, requires_grad=True)
    output = implicit_weighted_spline_sum(projected, graph)
    graph.positions.add_(0)
    with pytest.raises(RuntimeError, match="modified"):
        output.sum().backward()
