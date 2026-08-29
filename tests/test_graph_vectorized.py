from __future__ import annotations

import torch

from asgcn_recon.graph import build_causal_graph


def _reference_graph(
    positions: torch.Tensor,
    candidates: int,
    spatial_radius: float,
    temporal_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[torch.Tensor] = []
    destinations: list[torch.Tensor] = []
    attributes: list[torch.Tensor] = []
    count = positions.shape[0]
    for offset in range(1, min(candidates, max(count - 1, 0)) + 1):
        source = torch.arange(count - offset)
        destination = source + offset
        delta = positions[destination] - positions[source]
        valid = (
            torch.linalg.vector_norm(delta[:, :2], dim=-1) <= spatial_radius
        ) & (delta[:, 2] <= temporal_radius)
        delta = delta[valid]
        sources.append(source[valid])
        destinations.append(destination[valid])
        attributes.append(
            torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), dim=1)
        )
    self_nodes = torch.arange(count)
    sources.append(self_nodes)
    destinations.append(self_nodes)
    attributes.append(torch.zeros((count, 4)))
    return (
        torch.stack((torch.cat(sources), torch.cat(destinations))),
        torch.cat(attributes),
    )


def test_vectorized_causal_graph_matches_offset_loop() -> None:
    generator = torch.Generator().manual_seed(2026)
    positions = torch.rand((23, 3), generator=generator)
    positions[:, 2] = positions[:, 2].sort().values

    actual = build_causal_graph(positions, 7, 0.45, 0.35)
    expected = _reference_graph(positions, 7, 0.45, 0.35)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_vectorized_causal_graph_handles_empty_input() -> None:
    edge_index, edge_attr = build_causal_graph(torch.empty((0, 3)), candidates=32)

    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 4)
