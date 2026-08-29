from __future__ import annotations

import torch

from asgcn_recon.graph import build_radius_graph


def _reference_radius_graph(
    positions: torch.Tensor,
    radius: float,
    position_dims: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    destinations: list[int] = []
    attributes: list[float] = []
    for source in range(positions.shape[0]):
        for destination in range(positions.shape[0]):
            if source == destination:
                continue
            distance = torch.linalg.vector_norm(
                positions[source, :position_dims] - positions[destination, :position_dims]
            ).item()
            if distance < radius:
                sources.append(source)
                destinations.append(destination)
                attributes.append(distance / radius)
    return (
        torch.tensor((sources, destinations), dtype=torch.long),
        torch.tensor(attributes, dtype=positions.dtype).unsqueeze(-1),
    )


def test_chunked_radius_graph_matches_pairwise_reference() -> None:
    generator = torch.Generator().manual_seed(2026)
    positions = torch.rand((23, 4), generator=generator)
    radius = 0.45
    expected = _reference_radius_graph(positions, radius, position_dims=3)

    for chunk_size in (1, 2, 7, 23, 64):
        actual = build_radius_graph(
            positions,
            radius,
            position_dims=3,
            chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1], atol=1e-6, rtol=1e-6)


def test_radius_graph_handles_empty_input() -> None:
    edge_index, edge_attr = build_radius_graph(
        torch.empty((0, 4)),
        radius=0.08,
        position_dims=3,
        chunk_size=8,
    )

    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 1)
