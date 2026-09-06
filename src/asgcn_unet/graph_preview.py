"""CPU-only display data for an unchanged, exact model input graph.

The display edge budget never changes the model topology. This module neither
loads a checkpoint nor runs the encoder/decoder, and does not initialize CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import torch

from .graph import build_event_graph


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def _optional_timestamp(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite timestamp or null")
    return float(value)


@dataclass(frozen=True, slots=True)
class GraphPreview:
    """JSON display payload plus the complete CPU topology for exact selection."""

    payload: dict[str, Any]
    _edge_index: torch.Tensor = field(repr=False)
    _node_count: int = field(repr=False)

    def neighbors(self, node: int) -> dict[str, Any]:
        """Return all outgoing neighbors, not just the displayed edge subset.

        Exact radius graphs are symmetric. The model graph builder sorts edges
        by source then destination, allowing a small range lookup without an
        edge-sized temporary mask for every interactive node selection.
        """
        node = _integer(node, "node", minimum=0)
        if node >= self._node_count:
            raise IndexError(f"node={node} is outside graph with {self._node_count} nodes")
        bounds = torch.searchsorted(
            self._edge_index[0], torch.tensor([node, node + 1], dtype=torch.long, device="cpu")
        ).tolist()
        neighbors = self._edge_index[1, bounds[0] : bounds[1]].tolist()
        return {"node": node, "neighbors": neighbors, "degree": len(neighbors)}


@torch.no_grad()
def build_graph_preview(
    sample: dict,
    model_config: dict,
    *,
    max_graph_edges: int,
    display_edges: int = 5000,
) -> GraphPreview:
    """Reconstruct full model topology on CPU and select edges only for display.

    ``sample`` must be the dataset's actual preprocessed evaluation sample.
    Required model settings are used verbatim: no node/event/radius reduction or
    changed sampling is introduced. ``max_graph_edges`` is an explicit safety
    guard, typically the saved evaluation's effective guard, not a display cap.
    CPU reconstruction uses the same float32 graph rules, but is not a claim of
    bitwise identity with a previous GPU run or a new quality evaluation.
    """
    if not isinstance(sample, dict) or not isinstance(model_config, dict):
        raise TypeError("sample and model_config must be dictionaries")
    max_graph_edges = _integer(max_graph_edges, "max_graph_edges")
    display_edges = _integer(display_edges, "display_edges", minimum=0)
    required = ("event_sampling_factor", "graph_radius", "graph_position_dims", "graph_chunk_size")
    missing = [name for name in required if name not in model_config]
    if missing:
        raise ValueError(f"Missing model graph settings: {', '.join(missing)}")
    factor = _integer(model_config["event_sampling_factor"], "event_sampling_factor")
    position_dims = _integer(model_config["graph_position_dims"], "graph_position_dims")
    if position_dims not in {1, 2, 3, 4}:
        raise ValueError("graph_position_dims must be one of 1, 2, 3, or 4")
    chunk_size = _integer(model_config["graph_chunk_size"], "graph_chunk_size")
    radius = model_config["graph_radius"]
    if isinstance(radius, bool) or not isinstance(radius, Real) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph_radius must be finite and positive")
    configured_guard = model_config.get("max_graph_edges")
    if configured_guard is not None:
        configured_guard = _integer(configured_guard, "model.max_graph_edges")
        if max_graph_edges < configured_guard:
            raise ValueError("max_graph_edges cannot be below model.max_graph_edges")

    events = sample.get("events")
    if not isinstance(events, torch.Tensor):
        raise TypeError("sample.events must be a CPU tensor")
    if events.device.type != "cpu":
        raise ValueError("Graph previews require CPU events; no GPU transfer is performed")
    if events.layout != torch.strided or events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("sample.events must be a strided [N,4] tensor")
    if events.is_complex() or events.dtype == torch.bool:
        raise TypeError("sample.events must contain real numeric event values")
    size = sample.get("sensor_size")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("sample.sensor_size must contain height and width")
    sensor_size = tuple(_integer(value, "sensor_size") for value in size)
    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("sample.metadata must be a dictionary")
    retained = int(events.shape[0])
    raw = _integer(metadata.get("raw_event_count", retained), "raw_event_count", minimum=0)
    if raw < retained:
        raise ValueError("raw_event_count cannot be below the retained event count")
    t0_us = _optional_timestamp(metadata.get("t0_us"), "t0_us")
    t1_us = _optional_timestamp(metadata.get("t1_us"), "t1_us")
    if t0_us is not None and t1_us is not None and t1_us < t0_us:
        raise ValueError("t1_us cannot precede t0_us")

    graph = build_event_graph(
        events.detach(), sensor_size,
        event_sampling_factor=factor,
        graph_radius=float(radius),
        graph_position_dims=position_dims,
        graph_chunk_size=chunk_size,
        max_graph_edges=max_graph_edges,
    )
    node_count = int(graph.node_features.shape[0])
    edge_count = int(graph.edge_index.shape[1])
    displayed = min(display_edges, edge_count)
    if displayed == edge_count:
        display_index = graph.edge_index
    elif displayed == 0:
        display_index = graph.edge_index[:, :0]
    else:
        # Evenly spaced integer edge positions; deterministic, unique, and small.
        positions = torch.arange(displayed, dtype=torch.long, device="cpu")
        positions = positions * (edge_count - 1) // max(displayed - 1, 1)
        display_index = graph.edge_index.index_select(1, positions)
    degrees = graph.in_degree
    if degrees is None:
        degrees = torch.bincount(graph.edge_index[1], minlength=node_count)
    payload = {
        "nodes": graph.node_features.tolist(),
        "edges": display_index.t().tolist(),
        "statistics": {
            "nodes": node_count,
            "actual_directed_edges": edge_count,
            "displayed_edges": displayed,
            "isolated_nodes": int((degrees == 0).sum()),
            "max_degree": int(degrees.max()) if node_count else 0,
        },
        "radius": float(radius),
        "position_dims": position_dims,
        "metadata": {
            "raw_events": raw,
            "retained_events": retained,
            "t0_us": t0_us,
            "t1_us": t1_us,
            "sensor_size": list(sensor_size),
        },
        "provenance_note": (
            "CPU reconstruction of the complete model input graph using the configured "
            "event sampling and strict-radius rules. All model nodes are shown; "
            f"only {displayed} of {edge_count} directed edges are displayed "
            "(display-only limit; full topology and neighbor queries are unchanged). "
            "Node columns are normalized x, y, t, and polarity (-1/+1). "
            "No inference or quality evaluation was performed, and bitwise identity "
            "with an earlier GPU graph is not asserted."
        ),
    }
    return GraphPreview(payload, graph.edge_index, node_count)
