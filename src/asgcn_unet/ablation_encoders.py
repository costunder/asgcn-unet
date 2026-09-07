"""Explicit no-neighbour architecture controls, not graph failure fallbacks.

All configured events and their original normalized x/y/t/p features are retained.
The pointwise encoder shares the existing calibrated IF recurrence with ASGCN,
but its learned affine transform has no spline kernel or neighbour aggregation.
"""

from __future__ import annotations

import math
from itertools import accumulate

import torch
from torch import nn

from .graph import (
    ASGCNEncoder,
    EventGraph,
    PackedEventGraph,
    PaperSplineConv,
    _validate_packed_counts,
    prepare_event_nodes,
    uniformly_sample_events,
)


def _edgeless_container(features: torch.Tensor, positions: torch.Tensor) -> EventGraph:
    """No edges is the declared ablation architecture, never an OOM fallback."""
    return EventGraph(
        features,
        positions,
        torch.empty((2, 0), device=features.device, dtype=torch.long),
        features.new_empty((0, 1)),
        torch.zeros(features.shape[0], device=features.device, dtype=torch.long),
    )


def prepare_event_container(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
) -> EventGraph:
    """Use original event normalization/R sampling without any radius graph."""
    sampled = uniformly_sample_events(events, event_sampling_factor)
    return _edgeless_container(*prepare_event_nodes(sampled, sensor_size))


def prepare_event_container_batch(
    events: torch.Tensor,
    event_counts: tuple[int, ...],
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
) -> PackedEventGraph:
    """Normalize all physical-batch nodes together; sampling restarts per sample.

    This is the vectorized normalization used by ``build_event_graph_batch`` with
    radius construction deliberately absent. Python iterates over CPU counts only;
    no sample-wise tensor normalization, encoder forward or device transfer occurs.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have shape [N,4] with x,y,t,p columns")
    _validate_packed_counts(event_counts, int(events.shape[0]))
    uniformly_sample_events(events[:0], event_sampling_factor)
    factor = int(event_sampling_factor)
    height, width = (int(value) for value in sensor_size)
    if height < 1 or width < 1:
        raise ValueError("sensor_size must contain positive height and width")
    node_counts = tuple((count + factor - 1) // factor for count in event_counts)
    node_total = sum(node_counts)
    counts_tensor = torch.tensor(node_counts, device=events.device, dtype=torch.long)
    node_batch = torch.repeat_interleave(
        torch.arange(len(node_counts), device=events.device),
        counts_tensor,
        output_size=node_total,
    )
    if node_total:
        node_starts = counts_tensor.cumsum(0) - counts_tensor
        if factor == 1:
            sampled = events.float()
        else:
            event_starts = torch.tensor(
                (0, *accumulate(event_counts)), device=events.device, dtype=torch.long
            )
            local_index = torch.arange(node_total, device=events.device) - node_starts[node_batch]
            sampled = events[event_starts[node_batch] + local_index * factor].float()
        adjacent_same_sample = node_batch[1:] == node_batch[:-1]
        finite, unordered = torch.stack(
            (
                torch.isfinite(sampled).all(),
                ((sampled[1:, 2] < sampled[:-1, 2]) & adjacent_same_sample).any(),
            )
        ).tolist()
        if not finite:
            raise ValueError("Event coordinates, timestamps, and polarities must be finite")
        if unordered:
            raise ValueError("Event timestamps must be monotonically non-decreasing")
        first = sampled[node_starts[node_batch], 2]
        last = sampled[(node_starts + counts_tensor - 1)[node_batch], 2]
        x = sampled[:, 0] / max(width - 1, 1)
        y = sampled[:, 1] / max(height - 1, 1)
        t = (sampled[:, 2] - first) / (last - first).abs().clamp_min(1e-6)
        polarity = torch.where(sampled[:, 3] > 0, 1.0, -1.0)
        features = torch.stack((x, y, t, polarity), dim=-1)
        positions = torch.stack((x, y, t, (polarity + 1.0) * 0.5), dim=-1)
    else:
        features = torch.empty((0, 4), device=events.device, dtype=torch.float32)
        positions = torch.empty_like(features)
    return PackedEventGraph(
        _edgeless_container(features, positions),
        node_counts,
        node_batch,
        torch.zeros(len(node_counts), device=events.device, dtype=torch.long),
    )


def _require_no_neighbours(graph: EventGraph) -> None:
    if graph.edge_index.shape[1] or graph.edge_attr.shape[0]:
        raise ValueError("No-graph ablations require an explicitly edgeless event container")


class IdentityEventEncoder(nn.Module):
    """Raw normalized event features for the no-GNN/no-SNN decoder baseline."""

    supports_snn = False

    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = 4
        self.layers = nn.ModuleList()
        self.register_buffer("calibration_samples_seen", torch.empty(0, dtype=torch.long))

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        _require_no_neighbours(graph)
        if graph.node_features.ndim != 2 or graph.node_features.shape[1] != 4:
            raise ValueError("Identity encoder expects normalized x,y,t,p features")
        return graph.node_features, []

    def forward_snn(self, *args, **kwargs):
        raise ValueError(
            "IdentityEventEncoder is the no-SNN baseline; SNN inference is unsupported"
        )

    def fold_batch_norm(self) -> None:
        raise ValueError("IdentityEventEncoder has no learned encoder or BatchNorm to convert")

    def reset_activation_maxima(self) -> None:
        raise ValueError("IdentityEventEncoder does not support SNN calibration")

    def update_activation_maxima(self, *args, **kwargs) -> None:
        raise ValueError("IdentityEventEncoder does not support SNN calibration")

    def apply_parameter_normalization(self) -> None:
        raise ValueError("IdentityEventEncoder does not support ANN-to-SNN conversion")

    def output_activation_scale(self, reference: torch.Tensor) -> torch.Tensor:
        raise ValueError("IdentityEventEncoder has no converted SNN activation scale")

    def calibration_summary(self) -> dict:
        raise ValueError("IdentityEventEncoder does not support SNN calibration")


class PointwiseEventLayer(PaperSplineConv):
    """One affine transform per event plus the existing BN/conversion state.

    Only the validated BN state loading, activation and forward interface are
    inherited. Construction never creates spline/root parameters or a spline backend.
    The weight is a plain [input, output] affine matrix, with every entry used.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        nn.Module.__init__(self)
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.norm = nn.BatchNorm1d(out_channels)
        self.register_buffer("bn_bypassed", torch.tensor(False))
        self.register_buffer("snn_normalized", torch.tensor(False))
        self.register_buffer("calibration_activation_max", torch.ones(out_channels))
        self.register_buffer("normalization_scale", torch.ones(out_channels))
        self.register_buffer("dead_channel_mask", torch.zeros(out_channels, dtype=torch.bool))
        self.register_buffer("threshold", torch.ones(out_channels))
        self._bn_is_folded = False
        self._snn_is_normalized = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        self.norm.reset_parameters()
        self.bn_bypassed.fill_(False)
        self.snn_normalized.fill_(False)
        self.calibration_activation_max.fill_(1.0)
        self.normalization_scale.fill_(1.0)
        self.dead_channel_mask.fill_(False)
        self.threshold.fill_(1.0)
        self._bn_is_folded = False
        self._snn_is_normalized = False

    def affine(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache=None,
        in_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_index.shape[1] or edge_attr.shape[0] or basis_cache is not None:
            raise ValueError("PointwiseEventLayer cannot consume neighbour edges or spline bases")
        return x @ self.weight + self.bias

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        self.weight.mul_(scale.view(1, -1))
        self.bias.copy_((self.bias - self.norm.running_mean) * scale + self.norm.bias)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    @torch.no_grad()
    def apply_parameter_normalization(
        self, input_scale: torch.Tensor, output_scale: torch.Tensor
    ) -> None:
        if not self._bn_is_folded:
            raise RuntimeError("Fold BatchNorm before ANN-to-SNN parameter normalization")
        if self._snn_is_normalized:
            raise RuntimeError("ANN-to-SNN parameter normalization was already applied")
        input_scale = input_scale.to(self.weight)
        output_scale = output_scale.to(self.weight)
        if input_scale.shape != (self.in_channels,):
            raise ValueError("Input activation scale does not match pointwise input channels")
        if output_scale.shape != (self.out_channels,):
            raise ValueError("Output activation scale does not match pointwise output channels")
        if not bool(torch.isfinite(input_scale).all()) or bool((input_scale <= 0).any()):
            raise ValueError("Input activation scale must be finite and positive")
        input_scale = input_scale.clamp_min(1e-6)
        output_scale = output_scale.clamp_min(1e-6)
        raw_max = self.calibration_activation_max.to(self.weight)
        if not bool(torch.isfinite(raw_max).all()) or bool((raw_max < 0).any()):
            raise ValueError("Calibration activation maximum must be finite and non-negative")
        dead_mask = raw_max <= 0
        expected = torch.where(dead_mask, torch.ones_like(raw_max), raw_max).clamp_min(1e-6)
        if not torch.equal(output_scale, expected):
            raise ValueError("Output activation scale must derive from the raw calibration maximum")
        self.weight.mul_(input_scale.view(-1, 1))
        self.weight.div_(output_scale.view(1, -1))
        self.bias.div_(output_scale)
        self.normalization_scale.copy_(output_scale)
        self.dead_channel_mask.copy_(dead_mask)
        self.threshold.fill_(1.0)
        self.snn_normalized.fill_(True)
        self._snn_is_normalized = True


class PointwiseEventEncoder(ASGCNEncoder):
    """Graph-free ANN/SNN control with explicitly configured matched depth/width.

    The IF timestep recurrence, feature maxima, dead-channel treatment, conversion
    order and decoder-unit restoration all use ASGCNEncoder's unchanged methods.
    Only neighbour-dependent affine operators and their basis cache are replaced.
    """

    supports_snn = True

    def __init__(self, hidden_dim: int, graph_layers: int) -> None:
        nn.Module.__init__(self)
        for name, value in (("hidden_dim", hidden_dim), ("graph_layers", graph_layers)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.hidden_dim = hidden_dim
        # The parent IF path validates this pure-PyTorch backend; no spline
        # operator, spline parameter or neighbour graph is created or executed.
        self.spline_backend = "torch"
        channels = [4] + [hidden_dim] * graph_layers
        self.layers = nn.ModuleList(
            PointwiseEventLayer(channels[index], channels[index + 1])
            for index in range(graph_layers)
        )
        self.register_buffer(
            "calibration_samples_seen", torch.zeros(graph_layers, dtype=torch.long)
        )

    def _basis_cache(self, graph: EventGraph) -> None:
        _require_no_neighbours(graph)
