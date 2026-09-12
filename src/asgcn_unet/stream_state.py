"""Explicit, transferable streaming reconstruction state; no model-global cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from .graph import EventGraph
from .stream_graph import StreamGraph


def map_graph(graph: StreamGraph, fn) -> StreamGraph:
    from .implicit_radius import ImplicitRadiusGraph

    value = graph.graph
    if isinstance(value, ImplicitRadiusGraph):
        value.validate_integrity()
        if not torch.equal(value.node_batch, graph.node_batch):
            raise ValueError("Implicit graph and stream node namespaces disagree")
        node_batch = fn(graph.node_batch)
        mapped = ImplicitRadiusGraph.from_counted_nodes(
            fn(value.node_features), fn(value.positions), node_batch, fn(value.in_degree),
            batch_size=value.batch_size, radius=value.radius, position_dims=value.position_dims,
            chunk_size=value.chunk_size, candidate_pair_budget=value.candidate_pair_budget,
            edge_counts=fn(value.edge_counts),
        )
        return StreamGraph(mapped, node_batch, fn(graph.timestamps))
    return StreamGraph(
        EventGraph(*(fn(getattr(value, name)) for name in
                     ("node_features", "positions", "edge_index", "edge_attr", "in_degree"))),
        fn(graph.node_batch), fn(graph.timestamps),
    )


def map_cache(cache, graph: StreamGraph, fn):
    if cache is None:
        return None
    values = {}
    for field in fields(cache):
        value = getattr(cache, field.name)
        if field.name == "graph":
            value = graph
        elif isinstance(value, torch.Tensor):
            value = fn(value)
        elif isinstance(value, tuple) and all(isinstance(item, torch.Tensor) for item in value):
            value = tuple(fn(item) for item in value)
        values[field.name] = value
    return replace(cache, **values)


@dataclass
class StreamingReconstructionState:
    graph: StreamGraph
    encoder: Any
    decoder: torch.Tensor | None
    origin_seconds: float
    watermark_seconds: float
    sequence_index: int
    sequence_identity: tuple[str, str]
    last_event_id: tuple[int, int] | None
    contract: str
    sampling_offset: int = 0
    hierarchy: Any = None

    def _map(self, fn):
        graph = map_graph(self.graph, fn)
        return replace(self, graph=graph, encoder=map_cache(self.encoder, graph, fn),
                       decoder=None if self.decoder is None else fn(self.decoder),
                       hierarchy=None if self.hierarchy is None else self.hierarchy.map(fn, graph))

    def detach(self):
        return self._map(lambda value: value.detach())

    def clone(self):
        return self._map(lambda value: value.clone())

    def to(self, device=None, *, copy=False):
        return self._map(lambda value: value.to(device=device, copy=copy))

    def finite(self):
        from .implicit_radius import ImplicitRadiusGraph

        graph = self.graph.graph
        values = [graph.node_features, graph.positions, self.graph.timestamps]
        if not isinstance(graph, ImplicitRadiusGraph):
            values.append(graph.edge_attr)
        if self.decoder is not None:
            values.append(self.decoder)
        if self.encoder is not None:
            for field in fields(self.encoder):
                value = getattr(self.encoder, field.name)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    values.append(value)
                elif isinstance(value, tuple):
                    values.extend(item for item in value if isinstance(item, torch.Tensor)
                                  and item.is_floating_point())
        if self.hierarchy is not None:
            values.extend(self.hierarchy.tensors())
        return torch.stack([torch.isfinite(value).all() for value in values]).all()

    def training_payload(self):
        # Training is synchronous ANN. Learned activation/membrane caches cannot
        # survive an optimizer update, and must never enter its resume contract.
        if self.encoder is not None or self.hierarchy is not None:
            raise ValueError("Only raw-graph ANN training state can be checkpointed here")
        offset = _validate_sampling_offset(self.sampling_offset)
        raw_state = self._raw_training_payload()
        if offset == 0:
            return raw_state
        return {"schema": "asgcn_stream_training_state_v3", "sampling_offset": offset,
                "raw_state": raw_state}

    def _raw_training_payload(self):
        from .implicit_radius import ImplicitRadiusGraph

        graph = self.graph.graph
        if isinstance(graph, ImplicitRadiusGraph):
            graph.validate_integrity()
            if graph.batch_size != 1 or not torch.equal(graph.node_batch, self.graph.node_batch):
                raise ValueError("A stored implicit stream state must contain exactly one graph namespace")
            return {
                "schema": "asgcn_stream_training_state_v2",
                "graph": {name: getattr(graph, name) for name in
                          ("node_features", "positions", "in_degree", "edge_counts")},
                "geometry": {"representation": "implicit_radius_v1", "batch_size": graph.batch_size,
                             "radius": graph.radius, "position_dims": graph.position_dims,
                             "chunk_size": graph.chunk_size, "candidate_pair_budget": graph.candidate_pair_budget},
                "node_batch": self.graph.node_batch, "timestamps": self.graph.timestamps,
                "decoder": self.decoder, "origin_seconds": self.origin_seconds,
                "watermark_seconds": self.watermark_seconds, "sequence_index": self.sequence_index,
                "sequence_identity": self.sequence_identity, "last_event_id": self.last_event_id,
                "contract": self.contract,
            }
        return {
            "schema": "asgcn_stream_training_state_v1",
            "graph": {name: getattr(graph, name) for name in
                      ("node_features", "positions", "edge_index", "edge_attr", "in_degree")},
            "node_batch": self.graph.node_batch, "timestamps": self.graph.timestamps,
            "decoder": self.decoder, "origin_seconds": self.origin_seconds,
            "watermark_seconds": self.watermark_seconds, "sequence_index": self.sequence_index,
            "sequence_identity": self.sequence_identity, "last_event_id": self.last_event_id,
            "contract": self.contract,
        }


def recurrent_isfinite(value):
    return value.finite() if isinstance(value, StreamingReconstructionState) else torch.isfinite(value).all()


def _validate_sampling_offset(value):
    if type(value) is not int or value < 0:
        raise ValueError("Streaming sampling_offset must be a nonnegative integer")
    return value


def _unwrap_stream_training_payload(payload):
    """Keep legacy raw-state validation intact; never accept nested wrappers."""
    if not isinstance(payload, dict) or payload.get("schema") != "asgcn_stream_training_state_v3":
        return payload, 0
    if set(payload) != {"schema", "sampling_offset", "raw_state"}:
        raise ValueError("Invalid streaming sampling state fields")
    offset = _validate_sampling_offset(payload["sampling_offset"])
    raw_state = payload["raw_state"]
    if (not isinstance(raw_state, dict) or not isinstance(raw_state.get("schema"), str)
            or raw_state["schema"] not in
            {"asgcn_stream_training_state_v1", "asgcn_stream_training_state_v2"}):
        raise ValueError("Streaming sampling state requires a legacy v1/v2 raw_state")
    return raw_state, offset


def restore_stream_training_state(payload):
    payload, sampling_offset = _unwrap_stream_training_payload(payload)
    if isinstance(payload, dict) and payload.get("schema") == "asgcn_stream_training_state_v2":
        return replace(_restore_implicit_stream_training_state(payload), sampling_offset=sampling_offset)
    expected = {"schema", "graph", "node_batch", "timestamps", "decoder", "origin_seconds",
                "watermark_seconds", "sequence_index", "sequence_identity", "last_event_id", "contract"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Invalid streaming training state fields")
    if payload["schema"] != "asgcn_stream_training_state_v1":
        raise ValueError("Unsupported streaming training state schema")
    graph_fields = ("node_features", "positions", "edge_index", "edge_attr", "in_degree")
    if not isinstance(payload["graph"], dict) or set(payload["graph"]) != set(graph_fields):
        raise ValueError("Invalid streaming graph fields")
    raw_graph = payload["graph"]
    if any(not isinstance(raw_graph[name], torch.Tensor)
           or raw_graph[name].layout != torch.strided for name in graph_fields):
        raise ValueError("Streaming graph fields must be dense tensors")
    features, positions, edges, attributes, degree = (raw_graph[name] for name in graph_fields)
    count = features.shape[0] if features.ndim else -1
    if (features.shape != (count, 4) or features.dtype != torch.float32
            or positions.shape != (count, 4) or positions.dtype != torch.float64
            or edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long
            or attributes.shape != (edges.shape[1], 1) or attributes.dtype != torch.float64
            or degree.shape != (count,) or degree.dtype != torch.long):
        raise ValueError("Invalid physical streaming graph tensor contract")
    for name in ("node_batch", "timestamps"):
        if (not isinstance(payload[name], torch.Tensor) or payload[name].shape != (count,)
                or payload[name].layout != torch.strided):
            raise ValueError("Invalid streaming node metadata")
    if any(value.device != features.device for value in
           (*raw_graph.values(), payload["node_batch"], payload["timestamps"])):
        raise ValueError("All streaming state tensors must share a device")
    if (payload["node_batch"].dtype != torch.long or bool((payload["node_batch"] != 0).any())
            or payload["timestamps"].dtype != torch.float64):
        raise ValueError("A stored streaming state must contain exactly one graph namespace")
    if edges.numel() and (
        bool((edges < 0).any()) or bool((edges >= count).any())
        or bool((edges[0] == edges[1]).any())
    ):
        raise ValueError("Invalid streaming graph edge index")
    if not torch.equal(degree, torch.bincount(edges[1], minlength=count)):
        raise ValueError("Streaming graph degree does not match its edges")
    if bool(((attributes < 0) | (attributes >= 1)).any()):
        raise ValueError("Streaming edge distances must be in [0,1)")
    for name in ("origin_seconds", "watermark_seconds"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid streaming clock")
    if (payload["origin_seconds"] > payload["watermark_seconds"]
            or bool((payload["timestamps"][1:] < payload["timestamps"][:-1]).any())
            or bool((payload["timestamps"] < payload["origin_seconds"]).any())
            or bool((payload["timestamps"] > payload["watermark_seconds"]).any())):
        raise ValueError("Invalid streaming clock ordering")
    index = payload["sequence_index"]
    key = payload["sequence_identity"]
    last_id = payload["last_event_id"]
    if type(index) is not int or index < 0:
        raise ValueError("Invalid streaming sequence index")
    if (not isinstance(key, (tuple, list)) or len(key) != 2
            or not all(isinstance(part, str) for part in key) or not key[0]):
        raise ValueError("Invalid streaming sequence identity")
    if last_id is not None and (not isinstance(last_id, (tuple, list)) or len(last_id) != 2
                               or any(type(item) is not int or item < 0 for item in last_id)):
        raise ValueError("Invalid streaming event identity")
    if (not isinstance(payload["contract"], str) or len(payload["contract"]) != 64
            or any(character not in "0123456789abcdef" for character in payload["contract"])):
        raise ValueError("Invalid streaming model contract")
    decoder = payload["decoder"]
    if decoder is not None and (not isinstance(decoder, torch.Tensor) or decoder.ndim != 4
                               or decoder.shape[0] != 1 or not decoder.is_floating_point()
                               or decoder.layout != torch.strided
                               or any(size < 1 for size in decoder.shape)
                               or decoder.device != features.device):
        raise ValueError("Invalid streaming decoder state")
    if count and last_id is None:
        raise ValueError("A nonempty streaming graph requires last_event_id")
    graph = EventGraph(*(raw_graph[name] for name in graph_fields))
    state = StreamingReconstructionState(
        StreamGraph(graph, payload["node_batch"], payload["timestamps"]), None, decoder,
        float(payload["origin_seconds"]), float(payload["watermark_seconds"]), index,
        tuple(key), None if last_id is None else tuple(last_id), payload["contract"],
        sampling_offset=sampling_offset,
    )
    if not bool(state.finite()):
        raise ValueError("Nonfinite streaming state")
    return state


def _validate_training_capture_state(state):
    """Validate a live instance for capture without recounting its trusted graph.

    Only the private capture path supplies instances. External dictionaries use
    restore_stream_training_state and always recount exact geometric degrees.
    """
    from .implicit_radius import ImplicitRadiusGraph

    if not isinstance(state, StreamingReconstructionState):
        raise TypeError("Trusted capture requires a live StreamingReconstructionState instance")
    payload, sampling_offset = _unwrap_stream_training_payload(state.training_payload())
    if isinstance(state.graph.graph, ImplicitRadiusGraph):
        restored = _restore_implicit_stream_training_state(payload, _capture_graph=state.graph.graph)
    else:
        restored = restore_stream_training_state(payload)
    return replace(restored, sampling_offset=sampling_offset)


def _restore_implicit_stream_training_state(payload, *, _capture_graph=None):
    """Recount exact geometric degrees in bounded chunks; never trust cached E.

    No materialized edge list or zero-edge placeholder is constructed. This is
    checkpoint validation, not a change to the model's graph or window contract.
    """
    from .implicit_radius import ImplicitRadiusGraph

    expected = {"schema", "graph", "geometry", "node_batch", "timestamps", "decoder", "origin_seconds",
                "watermark_seconds", "sequence_index", "sequence_identity", "last_event_id", "contract"}
    if set(payload) != expected:
        raise ValueError("Invalid implicit streaming training state fields")
    raw = payload["graph"]
    graph_fields = {"node_features", "positions", "in_degree", "edge_counts"}
    if not isinstance(raw, dict) or set(raw) != graph_fields:
        raise ValueError("Invalid implicit graph fields; materialized edge placeholders are forbidden")
    if any(not isinstance(raw[name], torch.Tensor) or raw[name].layout != torch.strided for name in graph_fields):
        raise ValueError("Implicit graph fields must be dense tensors")
    features, positions, degree, edge_counts = (raw[name] for name in
                                             ("node_features", "positions", "in_degree", "edge_counts"))
    count = features.shape[0] if features.ndim else -1
    if (features.shape != (count, 4) or features.dtype != torch.float32
            or positions.shape != (count, 4) or positions.dtype != torch.float64
            or positions.requires_grad or degree.shape != (count,) or degree.dtype != torch.long
            or edge_counts.shape != (1,) or edge_counts.dtype != torch.long):
        raise ValueError("Invalid implicit physical streaming tensor shape/dtype")
    geometry = payload["geometry"]
    geometry_fields = {"representation", "batch_size", "radius", "position_dims", "chunk_size", "candidate_pair_budget"}
    if (not isinstance(geometry, dict) or set(geometry) != geometry_fields
            or geometry["representation"] != "implicit_radius_v1"
            or type(geometry["batch_size"]) is not int or geometry["batch_size"] != 1):
        raise ValueError("Invalid implicit radius graph representation/namespace")
    node_batch, timestamps = payload["node_batch"], payload["timestamps"]
    if any(not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != (count,)
           for value in (node_batch, timestamps)):
        raise ValueError("Invalid implicit streaming node metadata")
    if (node_batch.dtype != torch.long or timestamps.dtype != torch.float64
            or bool((node_batch != 0).any())):
        raise ValueError("A stored implicit stream must contain one graph namespace and float64 timestamps")
    if any(value.device != features.device for value in (*raw.values(), node_batch, timestamps)):
        raise ValueError("All implicit streaming tensors must share a device")
    for name in ("origin_seconds", "watermark_seconds"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid implicit streaming clock")
    if (payload["origin_seconds"] > payload["watermark_seconds"]
            or not bool(torch.isfinite(timestamps).all())
            or bool((timestamps[1:] < timestamps[:-1]).any())
            or bool((timestamps < payload["origin_seconds"]).any())
            or bool((timestamps > payload["watermark_seconds"]).any())):
        raise ValueError("Invalid implicit streaming clock ordering")
    index, key, last_id = payload["sequence_index"], payload["sequence_identity"], payload["last_event_id"]
    if type(index) is not int or index < 0:
        raise ValueError("Invalid implicit streaming sequence index")
    if (not isinstance(key, (tuple, list)) or len(key) != 2
            or not all(isinstance(part, str) for part in key) or not key[0]):
        raise ValueError("Invalid implicit streaming sequence identity")
    if last_id is not None and (not isinstance(last_id, (tuple, list)) or len(last_id) != 2
                               or any(type(item) is not int or item < 0 for item in last_id)):
        raise ValueError("Invalid implicit streaming event identity")
    if count and last_id is None:
        raise ValueError("A nonempty implicit stream requires last_event_id")
    if (not isinstance(payload["contract"], str) or len(payload["contract"]) != 64
            or any(character not in "0123456789abcdef" for character in payload["contract"])):
        raise ValueError("Invalid implicit streaming model contract")
    decoder = payload["decoder"]
    if decoder is not None and (not isinstance(decoder, torch.Tensor) or decoder.ndim != 4
                               or decoder.shape[0] != 1 or not decoder.is_floating_point()
                               or decoder.layout != torch.strided or any(size < 1 for size in decoder.shape)
                               or decoder.device != features.device or not bool(torch.isfinite(decoder).all())):
        raise ValueError("Invalid implicit streaming decoder state")
    graph = ImplicitRadiusGraph.from_counted_nodes(
        features, positions, node_batch, degree, batch_size=1,
        radius=geometry["radius"], position_dims=geometry["position_dims"],
        chunk_size=geometry["chunk_size"], candidate_pair_budget=geometry["candidate_pair_budget"],
        edge_counts=edge_counts,
    )
    if _capture_graph is None:
        counted_degree = torch.zeros_like(degree)
        for _source, destination, _distance in graph.iter_directed_neighbors():
            counted_degree.index_add_(0, destination, torch.ones_like(destination))
        if not torch.equal(counted_degree, degree):
            raise ValueError("Implicit streaming degree/counts do not match exact radius geometry")
    else:
        # Runtime graphs are constructed/count-updated by the exact graph engine.
        # Verify ownership/version, not just matching user-supplied count values.
        if (not isinstance(_capture_graph, ImplicitRadiusGraph)
                or any(getattr(_capture_graph, name) is not raw[name] for name in graph_fields)
                or _capture_graph.node_batch is not node_batch
                or any(getattr(_capture_graph, name) != geometry[name] for name in
                       ("batch_size", "radius", "position_dims", "chunk_size", "candidate_pair_budget"))):
            raise ValueError("Trusted capture graph does not own these exact runtime tensors/geometry")
        _capture_graph.validate_integrity()
    state = StreamingReconstructionState(
        StreamGraph(graph, node_batch, timestamps), None, decoder,
        float(payload["origin_seconds"]), float(payload["watermark_seconds"]), index,
        tuple(key), None if last_id is None else tuple(last_id), payload["contract"],
    )
    if not bool(state.finite()):
        raise ValueError("Nonfinite implicit streaming state")
    return state
