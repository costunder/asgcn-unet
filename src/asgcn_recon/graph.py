from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EventGraph:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Use running statistics when a graph has fewer than two real events."""
    if norm.training and values.shape[0] < 2:
        return F.batch_norm(
            values,
            norm.running_mean,
            norm.running_var,
            norm.weight,
            norm.bias,
            training=False,
            momentum=0.0,
            eps=norm.eps,
        )
    return norm(values)


def prepare_event_nodes(
    events: torch.Tensor, sensor_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize raw [x,y,t,p] while retaining event order."""
    height, width = sensor_size
    if events.numel() == 0:
        return (
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
            torch.empty((0, 3), device=events.device, dtype=torch.float32),
        )
    events = events.float()
    x = events[:, 0] / max(width - 1, 1)
    y = events[:, 1] / max(height - 1, 1)
    t = events[:, 2]
    t = (t - t[0]) / (t[-1] - t[0]).abs().clamp_min(1e-6)
    p = torch.where(events[:, 3] >= 0, 1.0, -1.0)
    positions = torch.stack((x, y, t), dim=-1)
    node_features = torch.stack((x * 2 - 1, y * 2 - 1, t * 2 - 1, p), dim=-1)
    return node_features, positions


def build_causal_graph(
    positions: torch.Tensor,
    candidates: int = 32,
    spatial_radius: float = 0.12,
    temporal_radius: float = 0.30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Connect each event only to recent events, enabling streaming inference.

    Restricting candidates avoids the quadratic radius-graph materialization that is
    unsuitable for the target low-latency system.
    """
    n = positions.shape[0]
    device = positions.device
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    attr_parts: list[torch.Tensor] = []
    max_offset = min(max(0, int(candidates)), max(0, n - 1))
    for offset in range(1, max_offset + 1):
        src = torch.arange(0, n - offset, device=device)
        dst = src + offset
        delta = positions[dst] - positions[src]
        spatial = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        valid = (spatial <= spatial_radius) & (delta[:, 2] <= temporal_radius)
        # Appending empty tensors is cheap and avoids a device-to-host synchronization
        # from ``if valid.any()`` for every candidate offset on CUDA.
        kept_delta = delta[valid]
        src_parts.append(src[valid])
        dst_parts.append(dst[valid])
        attr_parts.append(
            torch.cat(
                (kept_delta, torch.linalg.vector_norm(kept_delta, dim=-1, keepdim=True)),
                dim=-1,
            )
        )

    # Self edges guarantee a defined degree for sparse non-empty crops.
    self_nodes = torch.arange(n, device=device)
    src_parts.append(self_nodes)
    dst_parts.append(self_nodes)
    attr_parts.append(torch.zeros((n, 4), device=device, dtype=positions.dtype))
    edge_index = torch.stack((torch.cat(src_parts), torch.cat(dst_parts)), dim=0)
    edge_attr = torch.cat(attr_parts, dim=0)
    return edge_index, edge_attr


def build_event_graph(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    candidates: int,
    spatial_radius: float,
    temporal_radius: float,
) -> EventGraph:
    node_features, positions = prepare_event_nodes(events, sensor_size)
    edge_index, edge_attr = build_causal_graph(
        positions,
        candidates=candidates,
        spatial_radius=spatial_radius,
        temporal_radius=temporal_radius,
    )
    return EventGraph(node_features, positions, edge_index, edge_attr)


class SplineMessageLayer(nn.Module):
    """Distance-conditioned graph aggregation inspired by ASGCN's B-spline kernel."""

    def __init__(self, channels: int, edge_dim: int = 4) -> None:
        super().__init__()
        self.message = nn.Linear(channels, channels, bias=False)
        self.self_projection = nn.Linear(channels, channels)
        self.edge_kernel = nn.Sequential(
            nn.Linear(edge_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm1d(channels)
        self.register_buffer("bn_bypassed", torch.tensor(False), persistent=True)
        self._bn_is_folded = False
        self.register_buffer("threshold", torch.ones(channels), persistent=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._bn_is_folded = bool(self.bn_bypassed.item())

    def preactivation(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index
        gates = self.edge_kernel(edge_attr)
        messages = self.message(x[src]) * gates
        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, dst, messages)
        degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        degree.index_add_(0, dst, torch.ones((dst.numel(), 1), device=x.device, dtype=x.dtype))
        aggregate = aggregate / degree.clamp_min(1.0)
        output = self.self_projection(x) + aggregate
        return output if self._bn_is_folded else _safe_batch_norm(self.norm, output)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        variance = self.norm.running_var
        mean = self.norm.running_mean
        gamma = self.norm.weight
        beta = self.norm.bias
        scale = gamma / torch.sqrt(variance + self.norm.eps)
        self.message.weight.mul_(scale[:, None])
        self.self_projection.weight.mul_(scale[:, None])
        self.self_projection.bias.copy_((self.self_projection.bias - mean) * scale + beta)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.preactivation(x, edge_index, edge_attr)
        return torch.relu(z), z

    def rate_convert(
        self, z: torch.Tensor, simulation_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.numel() == 0:
            return z, torch.zeros((), device=z.device, dtype=z.dtype)
        threshold = self.threshold.to(dtype=z.dtype, device=z.device).clamp_min(1e-6)
        normalized = torch.clamp(torch.relu(z) / threshold, 0.0, 1.0)
        spike_count = torch.floor(normalized * simulation_steps + 1e-6)
        rate_output = spike_count * threshold / float(simulation_steps)
        # Keep diagnostics on-device so a timed CUDA forward has no hidden host sync.
        firing_rate = (spike_count / float(simulation_steps)).mean().detach()
        return rate_output, firing_rate


class ASGCNEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, graph_layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(4, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        self.register_buffer("input_bn_bypassed", torch.tensor(False), persistent=True)
        self._input_bn_is_folded = False
        self.layers = nn.ModuleList(
            [SplineMessageLayer(hidden_dim) for _ in range(int(graph_layers))]
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._input_bn_is_folded = bool(self.input_bn_bypassed.item())

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.input_linear(graph.node_features)
        if not self._input_bn_is_folded:
            h = _safe_batch_norm(self.input_norm, h)
        h = torch.relu(h)
        activations: list[torch.Tensor] = []
        for layer in self.layers:
            h, z = layer(h, graph.edge_index, graph.edge_attr)
            if return_activations:
                activations.append(torch.relu(z))
        return h, activations

    def forward_snn(
        self, graph: EventGraph, simulation_steps: int = 16
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.input_linear(graph.node_features)
        if not self._input_bn_is_folded:
            h = _safe_batch_norm(self.input_norm, h)
        h = torch.relu(h)
        firing_rates: list[torch.Tensor] = []
        for layer in self.layers:
            z = layer.preactivation(h, graph.edge_index, graph.edge_attr)
            h, firing_rate = layer.rate_convert(z, simulation_steps)
            firing_rates.append(firing_rate)
        return h, firing_rates

    @torch.no_grad()
    def update_thresholds(self, activations: list[torch.Tensor], momentum: float = -1.0) -> None:
        if len(activations) != len(self.layers):
            raise ValueError("Activation count does not match graph layer count")
        for layer, activation in zip(self.layers, activations, strict=True):
            if activation.numel() == 0:
                continue
            maxima = activation.amax(dim=0).clamp_min(1e-6)
            if momentum < 0:
                layer.threshold.copy_(torch.maximum(layer.threshold, maxima))
            else:
                layer.threshold.mul_(momentum).add_(maxima * (1.0 - momentum))

    @torch.no_grad()
    def reset_thresholds(self) -> None:
        for layer in self.layers:
            layer.threshold.fill_(1e-6)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.eval()
        if not self._input_bn_is_folded:
            variance = self.input_norm.running_var
            mean = self.input_norm.running_mean
            gamma = self.input_norm.weight
            beta = self.input_norm.bias
            scale = gamma / torch.sqrt(variance + self.input_norm.eps)
            self.input_linear.weight.mul_(scale[:, None])
            self.input_linear.bias.copy_((self.input_linear.bias - mean) * scale + beta)
            self.input_bn_bypassed.fill_(True)
            self._input_bn_is_folded = True
        for layer in self.layers:
            layer.fold_batch_norm()
