from __future__ import annotations

import pytest
import torch

from asgcn_unet.graph import build_radius_graph, radius_graph_topology


def _pairwise_reference(
    positions: torch.Tensor, radius: float, dimensions: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | float]]:
    """Independent dense oracle, only for small regression-test graphs."""
    count = positions.shape[0]
    coordinates = positions[:, :dimensions]
    difference = coordinates[:, None] - coordinates[None, :]
    distance = torch.linalg.vector_norm(difference, dim=-1)
    nonself = ~torch.eye(count, dtype=torch.bool, device=positions.device)
    retained = nonself & (distance < radius)
    edges = retained.nonzero().t().contiguous()
    attributes = (distance[retained] / radius).clamp(0, 1).unsqueeze(-1)
    cells = torch.floor(coordinates / radius).to(torch.long)
    adjacent = ((cells[:, None] - cells[None, :]).abs() <= 1).all(-1)
    degree = retained.sum(0)
    isolated = int((degree == 0).sum()) if count else 0
    topology = {
        "nodes": count,
        "candidate_directed_edges": int((adjacent & nonself).sum()),
        "actual_directed_edges": int(retained.sum()),
        "max_degree": int(degree.max()) if count else 0,
        "isolated_nodes": isolated,
        "isolate_ratio": isolated / count if count else 0.0,
    }
    return edges, attributes, topology


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
@pytest.mark.parametrize("chunk_size", [1, 7, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_radius_graph_and_topology_match_dense_oracle(
    dimensions: int, chunk_size: int, dtype: torch.dtype
) -> None:
    positions = torch.rand(
        (31, 4), generator=torch.Generator().manual_seed(1603), dtype=dtype
    )
    # Include boundary cells and distinct nodes at an identical position. Invalid
    # neighbor-cell hashes can alias occupied cells unless their counts are masked.
    positions[0].zero_()
    positions[1].fill_(1)
    positions[2] = positions[0]
    radius = 0.25
    expected_edges, expected_attributes, expected_topology = _pairwise_reference(
        positions, radius, dimensions
    )
    edges, attributes = build_radius_graph(
        positions, radius, position_dims=dimensions, chunk_size=chunk_size
    )
    assert torch.equal(edges, expected_edges)
    torch.testing.assert_close(attributes, expected_attributes, rtol=0, atol=0)
    assert attributes.dtype == dtype
    assert attributes.device == positions.device
    assert radius_graph_topology(
        positions, radius, position_dims=dimensions, chunk_size=chunk_size
    ) == expected_topology


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_singleton_graphs_have_no_edges_or_candidates(count: int) -> None:
    positions = torch.zeros((count, 4), dtype=torch.float64)
    edges, attributes = build_radius_graph(positions, 0.08, chunk_size=1)
    assert edges.shape == (2, 0)
    assert attributes.shape == (0, 1)
    assert attributes.dtype == positions.dtype
    assert radius_graph_topology(positions, 0.08, chunk_size=1) == {
        "nodes": count,
        "candidate_directed_edges": 0,
        "actual_directed_edges": 0,
        "max_degree": 0,
        "isolated_nodes": count,
        "isolate_ratio": float(count),
    }


@pytest.mark.parametrize("radius", [0.25, 1.0, 2.0])
def test_coincident_nodes_are_distinct_directed_neighbors(radius: float) -> None:
    positions = torch.full((13, 4), 0.5)
    edges, attributes = build_radius_graph(positions, radius, chunk_size=5)
    assert edges.shape[1] == 13 * 12
    assert torch.count_nonzero(attributes) == 0
    assert torch.all(edges[0] != edges[1])
    topology = radius_graph_topology(positions, radius, chunk_size=5)
    assert topology["candidate_directed_edges"] == 13 * 12
    assert topology["actual_directed_edges"] == 13 * 12
    assert topology["max_degree"] == 12
    assert topology["isolated_nodes"] == 0


def test_radius_boundary_remains_strict_without_rounding_approximation() -> None:
    radius = 0.25
    exact = torch.tensor(radius, dtype=torch.float64)
    below = torch.nextafter(exact, torch.tensor(0.0, dtype=torch.float64))
    above = torch.nextafter(exact, torch.tensor(1.0, dtype=torch.float64))
    positions = torch.zeros((4, 4), dtype=torch.float64)
    positions[1:, 0] = torch.stack((below, exact, above))
    expected_edges, expected_attributes, expected_topology = _pairwise_reference(
        positions, radius, 3
    )
    for chunk_size in (1, 4):
        edges, attributes = build_radius_graph(positions, radius, chunk_size=chunk_size)
        assert torch.equal(edges, expected_edges)
        torch.testing.assert_close(attributes, expected_attributes, atol=0, rtol=0)
        assert radius_graph_topology(
            positions, radius, chunk_size=chunk_size
        ) == expected_topology
        assert edges[1, edges[0] == 0].tolist() == [1]


@pytest.mark.parametrize("chunk_size", [1, 13])
def test_memory_guard_fails_instead_of_truncating_exact_graph(chunk_size: int) -> None:
    positions = torch.full((13, 4), 0.5)
    edge_count = 13 * 12
    edges, _ = build_radius_graph(
        positions, 0.25, chunk_size=chunk_size, max_edges=edge_count
    )
    assert edges.shape[1] == edge_count
    with pytest.raises(RuntimeError, match="max_graph_edges=155"):
        build_radius_graph(
            positions, 0.25, chunk_size=chunk_size, max_edges=edge_count - 1
        )
    # The preflight counter must measure the complete graph even when a model's
    # explicit materialization guard would reject it.
    assert radius_graph_topology(
        positions, 0.25, chunk_size=chunk_size
    )["actual_directed_edges"] == edge_count


@pytest.mark.parametrize("dimensions", [1, 3, 4])
def test_edge_attribute_position_gradients_match_dense_reference(dimensions: int) -> None:
    initial = torch.rand(
        (17, 4), generator=torch.Generator().manual_seed(605), dtype=torch.float64
    )
    initial[1] = initial[0]
    actual_positions = initial.clone().requires_grad_()
    reference_positions = initial.clone().requires_grad_()
    edges, attributes = build_radius_graph(
        actual_positions, 0.6, position_dims=dimensions, chunk_size=3
    )
    expected_edges, expected_attributes, _ = _pairwise_reference(
        reference_positions, 0.6, dimensions
    )
    assert torch.equal(edges, expected_edges)
    coefficients = torch.linspace(0.2, 1.3, len(attributes), dtype=torch.float64)[:, None]
    actual_gradient = torch.autograd.grad(
        (attributes * coefficients).sum(), actual_positions
    )[0]
    reference_gradient = torch.autograd.grad(
        (expected_attributes * coefficients).sum(), reference_positions
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(actual_gradient).all()
    assert torch.count_nonzero(actual_gradient[:, dimensions:]) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_selected_coordinates_are_not_silently_discarded(value: float) -> None:
    positions = torch.zeros((3, 4))
    positions[0, 0] = value
    for operation in (build_radius_graph, radius_graph_topology):
        with pytest.raises(ValueError, match="finite|\\[0,1\\]"):
            operation(positions, 0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA accelerator unavailable")
def test_cuda_exact_graph_topology_and_position_gradients() -> None:
    positions = torch.rand(
        (37, 4), generator=torch.Generator().manual_seed(77), dtype=torch.float64
    )
    positions[0].zero_()
    positions[1].fill_(1)
    positions[2] = positions[0]
    reference = positions.cuda().requires_grad_()
    actual = positions.cuda().requires_grad_()
    expected_edges, expected_attributes, expected_topology = _pairwise_reference(
        reference, 0.4, 3
    )
    edges, attributes = build_radius_graph(actual, 0.4, chunk_size=7)
    assert torch.equal(edges, expected_edges)
    torch.testing.assert_close(attributes, expected_attributes, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(attributes.sum(), actual)[0],
        torch.autograd.grad(expected_attributes.sum(), reference)[0],
        atol=1e-12,
        rtol=1e-12,
    )
    assert radius_graph_topology(actual, 0.4, chunk_size=7) == expected_topology
