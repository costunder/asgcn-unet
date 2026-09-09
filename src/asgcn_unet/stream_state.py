"""Explicit, transferable streaming reconstruction state; no model-global cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from .graph import EventGraph
from .stream_graph import StreamGraph


def map_graph(graph: StreamGraph, fn) -> StreamGraph:
    value = graph.graph
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

    def _map(self, fn):
        graph = map_graph(self.graph, fn)
        return replace(self, graph=graph, encoder=map_cache(self.encoder, graph, fn),
                       decoder=None if self.decoder is None else fn(self.decoder))

    def detach(self):
        return self._map(lambda value: value.detach())

    def clone(self):
        return self._map(lambda value: value.clone())

    def to(self, device=None, *, copy=False):
        return self._map(lambda value: value.to(device=device, copy=copy))

    def finite(self):
        values = [self.graph.graph.node_features, self.graph.graph.positions,
                  self.graph.graph.edge_attr, self.graph.timestamps]
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
        return torch.stack([torch.isfinite(value).all() for value in values]).all()

    def training_payload(self):
        # Training is synchronous ANN. Learned activation/membrane caches cannot
        # survive an optimizer update, and must never enter its resume contract.
        if self.encoder is not None:
            raise ValueError("Only raw-graph ANN training state can be checkpointed here")
        graph = self.graph.graph
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


def restore_stream_training_state(payload):
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
    )
    if not bool(state.finite()):
        raise ValueError("Nonfinite streaming state")
    return state
