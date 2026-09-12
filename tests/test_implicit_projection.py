"""Small CPU inference parity/work tests; not accelerator throughput evidence."""

import copy

import pytest
import torch

from asgcn_unet.graph import PaperSplineConv
from asgcn_unet.implicit_model import affine
from asgcn_unet.implicit_projection import selected_spline_sum
from asgcn_unet.implicit_radius import build_implicit_radius_graph


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fixture(dtype, backend, source_mode):
    generator = torch.Generator().manual_seed(905)
    positions = torch.rand((13, 4), generator=generator, dtype=torch.float64)
    positions[-1, :3] = 10.0  # An isolated destination/source, not a hidden subset.
    batches = torch.arange(13) % 2
    graph = build_implicit_radius_graph(
        torch.zeros_like(positions), positions, batches, batch_size=2,
        radius=1.0, chunk_size=3, candidate_pair_budget=7,
    )
    x = torch.randn((13, 4), generator=generator, dtype=dtype)
    if source_mode == "sparse":
        x[torch.arange(13) % 3 != 0] = 0
    elif source_mode == "zero":
        x.zero_()
    layer = PaperSplineConv(4, 3, edge_chunk_size=2, spline_backend=backend).to(dtype)
    return graph, x, layer


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
@pytest.mark.parametrize("selection", [[8, 1, 4, 12], [], [12]])
@pytest.mark.parametrize("source_mode", ["all", "sparse", "zero"])
def test_selected_one_pass_matches_autograd_path_and_actual_work(
        dtype, backend, selection, source_mode, monkeypatch):
    graph, x, layer = _fixture(dtype, backend, source_mode)
    selected = torch.tensor(selection, dtype=torch.long)
    omit_zero = source_mode != "all"
    # Gradient-enabled selected affine retains the original two-pass path.
    expected, sources, edges = affine(layer, x, graph, selected, omit_zero_sources=omit_zero)
    calls = []
    original = graph.index.iter_directed_neighbors

    def neighbors(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph.index, "iter_directed_neighbors", neighbors)
    with torch.no_grad():
        actual, actual_sources, actual_edges = affine(
            layer, x, graph, selected, omit_zero_sources=omit_zero,
        )
    tolerance = {torch.float64: 2e-12, torch.float32: 2e-6,
                 torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert len(calls) == 1  # No extra support-discovery radius pass, even when empty.
    assert actual_sources == sources and actual_edges == edges
    assert actual.shape == (len(selection), 3)


def test_each_source_projected_once_across_repeated_message_chunks(monkeypatch):
    graph, x, layer = _fixture(torch.float64, "torch", "all")
    selected = torch.arange(13)
    original = torch.einsum
    projected_rows = []

    def projection(equation, features, weight):
        assert equation == "ni,kio->nko"
        projected_rows.extend(features.tolist())
        return original(equation, features, weight)

    monkeypatch.setattr(torch, "einsum", projection)
    with torch.no_grad():
        _, sources, edges = affine(layer, x, graph, selected)
    assert edges > sources and len(projected_rows) == sources
    assert len({tuple(row) for row in projected_rows}) == sources
    assert tuple(x[-1].tolist()) not in {tuple(row) for row in projected_rows}


@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
def test_training_feature_weight_root_bias_gradients_still_match_materialized(backend):
    graph, x, layer = _fixture(torch.float64, backend, "all")
    x.requires_grad_()
    reference_x = x.detach().clone().requires_grad_()
    reference_layer = copy.deepcopy(layer)
    chunks = list(graph.iter_directed_neighbors())
    source, destination, pseudo = (torch.cat(parts) for parts in zip(*chunks))
    selected = torch.tensor([8, 1, 4, 12])
    actual, _, _ = affine(layer, x, graph, selected)
    expected = reference_layer.affine(
        reference_x, torch.stack((source, destination)), pseudo, in_degree=graph.in_degree,
    )[selected]
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=2e-12, atol=2e-12)
    for name in ("weight", "root", "bias"):
        actual_parameter = getattr(layer, name)
        reference_parameter = getattr(reference_layer, name)
        assert actual_parameter.grad is not None
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad,
                                   rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("backend", ["torch", "torch_fused"])
def test_cpu_autocast_preserves_projection_and_message_casts(backend):
    graph, x, layer = _fixture(torch.float32, backend, "sparse")
    selected = torch.tensor([8, 1, 4, 12])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _, _ = affine(layer, x, graph, selected, omit_zero_sources=True)
        with torch.no_grad():
            actual, _, _ = affine(layer, x, graph, selected, omit_zero_sources=True)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)


def test_lazy_projection_cannot_silently_detach_training_or_change_backend():
    graph, x, layer = _fixture(torch.float64, "torch", "all")
    selected = torch.tensor([1, 4])
    with pytest.raises(RuntimeError, match="inference-only"):
        selected_spline_sum(layer, x, graph, selected)
    layer.spline_backend = "triton"
    with torch.no_grad(), pytest.raises(ValueError, match="requires CUDA"):
        affine(layer, x, graph, selected)


def test_geometry_integrity_and_selection_validation_are_preserved():
    graph, x, layer = _fixture(torch.float64, "torch", "all")
    with torch.no_grad(), pytest.raises(ValueError, match="unique"):
        affine(layer, x, graph, torch.tensor([1, 1]))
    graph.positions.add_(0)
    with torch.no_grad(), pytest.raises(RuntimeError, match="modified"):
        affine(layer, x, graph, torch.tensor([1]))
