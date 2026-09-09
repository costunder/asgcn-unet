"""Memory-budgeted CPU display data for the complete evaluation input graph.

Every ordered node pair is checked using the model's float32 strict-radius
predicate in vectorized tiles. A second streaming pass selects display edges
in source-major order. Neither pass truncates the measured graph topology.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import torch

from .graph import prepare_event_nodes, uniformly_sample_events
from .graph_preview import _encoder_topology_kind, _integer, _optional_timestamp

# Conservative allowances for tensor scratch, Python lists, and JSON conversion.
# These describe the graph operation, not the interpreter or dataset decoding.
_FIXED_BYTES = 1 << 20
_BYTES_PER_NODE = 768
_BYTES_PER_DISPLAY_EDGE = 384
_BYTES_PER_PAIR = 96


def _memory_plan(
    events: torch.Tensor,
    node_count: int,
    display_edges: int,
    configured_chunk_size: int,
    memory_budget_bytes: int,
    *,
    topology_kind: str = "radius_graph",
) -> dict[str, int | str]:
    has_graph = topology_kind == "radius_graph"
    displayed_bound = min(display_edges, node_count * max(0, node_count - 1)) if has_graph else 0
    input_bytes = events.untyped_storage().nbytes()
    persistent_bytes = (
        _FIXED_BYTES
        + input_bytes
        + node_count * _BYTES_PER_NODE
        + displayed_bound * _BYTES_PER_DISPLAY_EDGE
    )
    available = memory_budget_bytes - persistent_bytes
    if available < (_BYTES_PER_PAIR if has_graph else 0):
        raise MemoryError(
            "Diagnostic graph memory budget is insufficient before topology allocation: "
            f"budget={memory_budget_bytes:,} bytes, estimated input/node/display storage="
            f"{persistent_bytes:,} bytes. Increase the explicitly verified CPU memory "
            "budget or use fewer displayed lines; all model nodes and topology rules "
            "must remain unchanged. No topology was truncated."
        )
    pair_capacity = available // _BYTES_PER_PAIR
    source_tile = (
        min(max(node_count, 1), configured_chunk_size, math.isqrt(pair_capacity))
        if has_graph
        else 0
    )
    destination_tile = min(max(node_count, 1), pair_capacity // source_tile) if has_graph else 0
    scratch_bytes = source_tile * destination_tile * _BYTES_PER_PAIR
    return {
        "budget_bytes": memory_budget_bytes,
        "input_storage_bytes": input_bytes,
        "estimated_persistent_bytes": persistent_bytes,
        "estimated_tile_scratch_bytes": scratch_bytes,
        "estimated_working_set_bytes": persistent_bytes + scratch_bytes,
        "source_tile_nodes": source_tile,
        "destination_tile_nodes": destination_tile,
        "maximum_pair_tile": source_tile * destination_tile,
        "configured_graph_chunk_size": configured_chunk_size,
        "topology_kind": topology_kind,
        "scope": (
            "Conservative graph-operation estimate including the supplied event storage, "
            "normalized nodes, statistics, display JSON, and tensor scratch; excludes "
            "interpreter/Torch baseline, other samples, targets, and dataset decoding. "
            "The caller must preflight those resources separately. Not an OS memory limit."
        ),
    }


def _pair_tiles(
    coordinates: torch.Tensor,
    sources: torch.Tensor,
    *,
    radius: float,
    source_tile: int,
    destination_tile: int,
) -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
    """Yield exact predicate tiles; never retain a previous tile's storage."""
    for start in range(0, sources.numel(), source_tile):
        selected_sources = sources[start : start + source_tile]
        source_positions = coordinates.index_select(0, selected_sources)
        for destination_start in range(0, coordinates.shape[0], destination_tile):
            destination_stop = min(destination_start + destination_tile, coordinates.shape[0])
            # Match the model's norm, not squared-distance/cdist: rounding at
            # the strict radius boundary is part of the original graph rule.
            distance = torch.linalg.vector_norm(
                source_positions[:, None, :]
                - coordinates[None, destination_start:destination_stop, :],
                dim=-1,
            )
            mask = distance < radius
            destinations = torch.arange(destination_start, destination_stop, device="cpu")
            mask &= selected_sources[:, None] != destinations[None, :]
            del distance, destinations
            yield selected_sources, destination_start, mask
            # The caller releases its mask too, before advancing this generator.
            del mask


def _display_subset(
    coordinates: torch.Tensor,
    radius: float,
    out_degree: torch.Tensor,
    edge_count: int,
    displayed: int,
    *,
    source_tile: int,
    destination_tile: int,
) -> list[list[int]]:
    if displayed == 0:
        return []
    targets = torch.arange(displayed, dtype=torch.long, device="cpu")
    targets *= edge_count - 1
    targets //= max(displayed - 1, 1)
    cumulative = out_degree.cumsum(0)
    offsets = cumulative - out_degree
    wanted_sources = torch.unique(torch.searchsorted(cumulative, targets, right=True))
    seen = torch.zeros_like(out_degree)
    result = torch.empty((displayed, 2), dtype=torch.long, device="cpu")
    filled = torch.zeros(displayed, dtype=torch.bool, device="cpu")
    for sources, destination_start, mask in _pair_tiles(
        coordinates,
        wanted_sources,
        radius=radius,
        source_tile=source_tile,
        destination_tile=destination_tile,
    ):
        # Full-graph source-major ranks remain correct across destination tiles.
        ranks = mask.cumsum(dim=1, dtype=torch.long)
        ranks += (offsets[sources] + seen[sources] - 1)[:, None]
        seen[sources] += mask.sum(dim=1)
        slots = torch.searchsorted(targets, ranks).clamp_max_(displayed - 1)
        matches = mask & (targets[slots] == ranks)
        local_source, local_destination = torch.nonzero(matches, as_tuple=True)
        selected_slots = slots[local_source, local_destination]
        result[selected_slots, 0] = sources[local_source]
        result[selected_slots, 1] = destination_start + local_destination
        filled[selected_slots] = True
        del mask, ranks, slots, matches, local_source, local_destination, selected_slots
    if not bool(filled.all()):
        raise RuntimeError("Diagnostic graph display selection did not cover the exact edge ranks")
    return result.tolist()


@torch.no_grad()
def build_diagnostic_graph(
    sample: dict,
    model_config: dict,
    *,
    memory_budget_bytes: int,
    display_edges: int = 5000,
) -> dict[str, Any]:
    """Return an offline graph payload from an actual preprocessed CPU sample.

    The caller must verify allocated/available RAM and reserve process/dataset
    overhead before passing the required graph-operation memory budget. Budget
    failure is explicit, never a smaller graph or a GPU/CPU fallback.

    ``display_edges`` limits browser drawing/JSON only. All model nodes and exact
    directed edges contribute to statistics/degrees. This is a selected-frame
    diagnostic, not inference or a quality evaluation.
    """
    if not isinstance(sample, dict) or not isinstance(model_config, dict):
        raise TypeError("sample and model_config must be dictionaries")
    from .stream_input import reject_streaming_frame_diagnostic

    reject_streaming_frame_diagnostic(model_config, sample=sample)
    topology_kind = _encoder_topology_kind(model_config)
    memory_budget_bytes = _integer(memory_budget_bytes, "memory_budget_bytes")
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
    radius = _optional_timestamp(model_config["graph_radius"], "graph_radius")
    if radius is None or radius <= 0:
        raise ValueError("graph_radius must be finite and positive")
    events = sample.get("events")
    if not isinstance(events, torch.Tensor):
        raise TypeError("sample.events must be a CPU tensor")
    if events.device.type != "cpu":
        raise ValueError("Diagnostic graphs require CPU events; no GPU transfer is performed")
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
    node_count = (retained + factor - 1) // factor
    plan = _memory_plan(
        events,
        node_count,
        display_edges,
        chunk_size,
        memory_budget_bytes,
        topology_kind=topology_kind,
    )
    nodes, positions = prepare_event_nodes(
        uniformly_sample_events(events.detach(), factor),
        sensor_size,
    )
    coordinates = positions[:, :position_dims]
    if coordinates.numel() and bool(((coordinates < 0) | (coordinates > 1)).any()):
        raise ValueError("Normalized graph coordinates must lie in [0,1]")
    if (
        topology_kind == "radius_graph"
        and node_count
        and max(2, math.ceil(1.0 / radius) + 1) ** position_dims >= (torch.iinfo(torch.long).max)
    ):
        raise ValueError("graph_radius is too small for the model's integer spatial hashing")
    source_tile = int(plan["source_tile_nodes"])
    destination_tile = int(plan["destination_tile_nodes"])
    in_degree = torch.zeros(node_count, dtype=torch.long, device="cpu")
    out_degree = torch.zeros_like(in_degree)
    if topology_kind == "radius_graph":
        all_sources = torch.arange(node_count, dtype=torch.long, device="cpu")
        for sources, destination_start, mask in _pair_tiles(
            coordinates,
            all_sources,
            radius=radius,
            source_tile=source_tile,
            destination_tile=destination_tile,
        ):
            out_degree[sources] += mask.sum(dim=1)
            in_degree[destination_start : destination_start + mask.shape[1]] += mask.sum(dim=0)
            del mask
    edge_count = int(in_degree.sum())
    if not torch.equal(in_degree, out_degree):
        raise RuntimeError("Strict-radius diagnostic topology unexpectedly lost symmetry")
    displayed = min(display_edges, edge_count)
    edges = _display_subset(
        coordinates,
        radius,
        out_degree,
        edge_count,
        displayed,
        source_tile=source_tile,
        destination_tile=destination_tile,
    )
    payload = {
        "nodes": nodes.tolist(),
        "edges": edges,
        "degrees": in_degree.tolist(),
        "statistics": {
            "nodes": node_count,
            "actual_directed_edges": edge_count,
            "displayed_edges": displayed,
            "isolated_nodes": int((in_degree == 0).sum()),
            "max_degree": int(in_degree.max()) if node_count else 0,
        },
        "topology_kind": topology_kind,
        "encoder_kind": model_config.get("encoder_kind", "graph"),
        "radius": radius if topology_kind == "radius_graph" else None,
        "position_dims": position_dims if topology_kind == "radius_graph" else None,
        "configured_radius": radius,
        "configured_position_dims": position_dims,
        "metadata": {
            "raw_events": raw,
            "retained_events": retained,
            "t0_us": t0_us,
            "t1_us": t1_us,
            "sensor_size": list(sensor_size),
        },
        "memory_plan": plan,
        "provenance_note": (
            "CPU reconstruction of the complete model input topology from the actual "
            "preprocessed sample, using the configured event sampling and float32 "
            "strict-radius rules. Every directed pair is checked in memory-budgeted "
            "tiles; the complete edge list is not retained. All model nodes and full "
            f"degrees are included; only {displayed} of {edge_count} directed edges "
            "are displayed (display-only limit, not model or graph truncation). "
            "Node columns are normalized x, y, t, and polarity (-1/+1). "
            "No inference or quality evaluation was performed, and bitwise identity "
            "with an earlier GPU graph is not asserted."
        ),
    }
    if topology_kind == "no_graph":
        payload["provenance_note"] = (
            "Actual normalized x/y/t/p event nodes for the explicitly configured "
            f"{model_config['encoder_kind']} encoder. Zero edges/degrees are the "
            "declared no-graph architecture, not a failure fallback or topology "
            "truncation. No pairwise distances or graph edges were computed. "
            "Configured event sampling and every remaining event node are retained; "
            "radius/position settings are recorded but not applied. "
            "No inference or quality evaluation was performed."
        )
    return payload
