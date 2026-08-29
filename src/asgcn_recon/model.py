from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import PAPER_CORE_VERSION, ASGCNEncoder, EventGraph, build_event_graph


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.body(x))


class ConvGRUCell(nn.Module):
    """Causal analog state; only the graph front-end is ANN-to-SNN converted."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        merged = input_channels + hidden_channels
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(merged, hidden_channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(merged, hidden_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        expected = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        if state is None or tuple(state.shape) != expected:
            state = torch.zeros(expected, device=x.device, dtype=x.dtype)
        reset, update = torch.sigmoid(self.gates(torch.cat((x, state), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * state), dim=1)))
        return (1.0 - update) * state + update * candidate


class RasterDecoder(nn.Module):
    def __init__(
        self, input_channels: int, base_channels: int, output_channels: int, recurrent: bool = True
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.enc1 = ResidualBlock(base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4), ResidualBlock(base_channels * 4)
        )
        self.recurrent = ConvGRUCell(base_channels * 4, base_channels * 4) if recurrent else None
        self.up2 = nn.Conv2d(base_channels * 6, base_channels * 2, 3, padding=1)
        self.dec2 = ResidualBlock(base_channels * 2)
        self.up1 = nn.Conv2d(base_channels * 3, base_channels, 3, padding=1)
        self.dec1 = ResidualBlock(base_channels)
        self.head = nn.Conv2d(base_channels, output_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))
        if self.recurrent is not None:
            state = self.recurrent(b, state)
            b = b + state
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        output = torch.sigmoid(self.head(u1))
        output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=False)
        return output, state


def rasterize_features(
    features: torch.Tensor,
    graph: EventGraph,
    sensor_size: tuple[int, int],
    downsample: int,
) -> torch.Tensor:
    height, width = sensor_size
    grid_h = max(1, (height + downsample - 1) // downsample)
    grid_w = max(1, (width + downsample - 1) // downsample)
    x = torch.clamp((graph.positions[:, 0] * width / downsample).long(), 0, grid_w - 1)
    y = torch.clamp((graph.positions[:, 1] * height / downsample).long(), 0, grid_h - 1)
    linear = y * grid_w + x
    raster = torch.zeros(
        (grid_h * grid_w, features.shape[-1]), device=features.device, dtype=features.dtype
    )
    raster.index_add_(0, linear, features)
    counts = torch.zeros((grid_h * grid_w, 1), device=features.device, dtype=features.dtype)
    counts.index_add_(
        0,
        linear,
        torch.ones((linear.numel(), 1), device=features.device, dtype=features.dtype),
    )
    raster = raster / counts.clamp_min(1.0)
    return raster.transpose(0, 1).reshape(1, features.shape[-1], grid_h, grid_w)


class ASGCNReconstructor(nn.Module):
    def __init__(
        self,
        architecture_version: int = PAPER_CORE_VERSION,
        graph_operator: str = "spline",
        spline_backend: str = "torch",
        spline_pseudo: str = "distance_over_radius",
        spline_is_open: bool = True,
        hidden_dim: int = 64,
        graph_layers: int = 6,
        event_sampling_factor: int = 1,
        graph_radius: float = 0.08,
        graph_position_dims: int = 3,
        graph_chunk_size: int = 512,
        max_graph_edges: int | None = 2_000_000,
        spline_kernel_size: int = 5,
        spline_degree: int = 1,
        spline_root_weight: bool = True,
        snn_dynamics: str = "literal_eq15",
        raster_downsample: int = 4,
        decoder_channels: int = 48,
        output_channels: int = 1,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        if int(architecture_version) != PAPER_CORE_VERSION:
            raise ValueError(
                f"architecture_version must be {PAPER_CORE_VERSION}; legacy edge-MLP "
                "checkpoints are intentionally incompatible"
            )
        if graph_operator != "spline":
            raise ValueError("graph_operator must be 'spline' for the ASGCN paper core")
        if spline_backend != "torch":
            raise ValueError("Only the portable pure-PyTorch spline backend is supported")
        if spline_pseudo != "distance_over_radius":
            raise ValueError(
                "spline_pseudo must be 'distance_over_radius'; this explicit "
                "reparameterization maps the paper's scalar distance to the "
                "SplineConv [0,1] domain"
            )
        if not bool(spline_is_open):
            raise ValueError("Only open B-spline bases are supported")
        if (
            isinstance(event_sampling_factor, bool)
            or int(event_sampling_factor) != event_sampling_factor
        ):
            raise ValueError("event_sampling_factor must be an integer")
        if int(event_sampling_factor) < 1:
            raise ValueError("event_sampling_factor must be at least 1")
        if not math.isfinite(float(graph_radius)) or float(graph_radius) <= 0:
            raise ValueError("graph_radius must be positive and finite")
        if int(graph_position_dims) not in {1, 2, 3, 4}:
            raise ValueError("graph_position_dims must be one of 1, 2, 3, or 4")
        if int(graph_chunk_size) < 1:
            raise ValueError("graph_chunk_size must be at least 1")
        if max_graph_edges is not None and (
            isinstance(max_graph_edges, bool)
            or int(max_graph_edges) != max_graph_edges
            or int(max_graph_edges) < 1
        ):
            raise ValueError("max_graph_edges must be a positive integer or null")
        if snn_dynamics not in {"literal_eq15", "standard_if"}:
            raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
        if int(raster_downsample) < 1:
            raise ValueError("raster_downsample must be at least 1")
        self.architecture_version = PAPER_CORE_VERSION
        self.encoder = ASGCNEncoder(
            hidden_dim,
            graph_layers,
            spline_kernel_size=spline_kernel_size,
            spline_degree=spline_degree,
            spline_root_weight=spline_root_weight,
        )
        self.decoder = RasterDecoder(hidden_dim, decoder_channels, output_channels, recurrent)
        self.event_sampling_factor = int(event_sampling_factor)
        self.graph_radius = float(graph_radius)
        self.graph_position_dims = int(graph_position_dims)
        self.graph_chunk_size = int(graph_chunk_size)
        self.max_graph_edges = int(max_graph_edges) if max_graph_edges is not None else None
        self.snn_dynamics = snn_dynamics
        self.raster_downsample = int(raster_downsample)

    def _graph(self, sample: dict[str, Any]) -> EventGraph:
        return build_event_graph(
            sample["events"],
            sample["sensor_size"],
            event_sampling_factor=self.event_sampling_factor,
            graph_radius=self.graph_radius,
            graph_position_dims=self.graph_position_dims,
            graph_chunk_size=self.graph_chunk_size,
            max_graph_edges=self.max_graph_edges,
        )

    def forward_sample(
        self,
        sample: dict[str, Any],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        return_activations: bool = False,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
            raise ValueError("simulation_steps must be an integer")
        simulation_steps = int(simulation_steps)
        graph = self._graph(sample)
        if inference_mode == "ann":
            features, activations = self.encoder.forward_ann(graph, return_activations)
            firing_rates: list[torch.Tensor] = []
        elif inference_mode == "snn":
            features, firing_rates = self.encoder.forward_snn(
                graph,
                simulation_steps,
                dynamics=self.snn_dynamics,
            )
            # Express normalized spike amplitudes in the analog decoder's trained
            # lambda_L units. For literal Eq. (15), this is dimensional rescaling,
            # not a claim of proven finite-T ANN-rate equivalence.
            features = features * self.encoder.output_activation_scale(features)
            activations = []
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}")
        raster = rasterize_features(features, graph, sample["sensor_size"], self.raster_downsample)
        prediction, next_state = self.decoder(raster, sample["sensor_size"], recurrent_state)
        dataset_sampling_factor = int(sample.get("metadata", {}).get("dataset_sampling_factor", 1))
        node_count = int(graph.node_features.shape[0])
        edge_count = int(graph.edge_index.shape[1])
        if node_count:
            degree = torch.bincount(graph.edge_index[1], minlength=node_count)
            isolated_nodes = (degree == 0).sum()
            max_degree = degree.max()
        else:
            isolated_nodes = graph.node_features.new_zeros((), dtype=torch.long)
            max_degree = graph.node_features.new_zeros((), dtype=torch.long)
        firing_rate_denominators = (
            [simulation_steps * node_count * layer.out_channels for layer in self.encoder.layers]
            if inference_mode == "snn"
            else []
        )
        diagnostics = {
            "paper_core_version": self.architecture_version,
            "nodes": node_count,
            "edges": edge_count,
            "isolated_nodes": isolated_nodes,
            "isolate_ratio": isolated_nodes.to(graph.node_features.dtype)
            / float(max(1, node_count)),
            "max_degree": max_degree,
            "edge_feature": "normalized_scalar_distance",
            "event_sampling_factor": self.event_sampling_factor,
            "dataset_sampling_factor": dataset_sampling_factor,
            "effective_sampling_factor": (dataset_sampling_factor * self.event_sampling_factor),
            "snn_dynamics": self.snn_dynamics if inference_mode == "snn" else None,
            "decoder_input_lambda_applied": inference_mode == "snn",
            "firing_rates": firing_rates,
            "firing_rate_denominators": firing_rate_denominators,
            "spike_counts": [
                rate * denominator
                for rate, denominator in zip(
                    firing_rates,
                    firing_rate_denominators,
                    strict=True,
                )
            ],
            "activations": activations,
            "recurrent_state": next_state,
        }
        return prediction, diagnostics

    def forward(
        self,
        batch: list[dict[str, Any]],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        recurrent_states: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        predictions, diagnostics = [], []
        if recurrent_states is None:
            recurrent_states = [None] * len(batch)
        for sample, state in zip(batch, recurrent_states, strict=True):
            prediction, detail = self.forward_sample(
                sample, inference_mode, simulation_steps, recurrent_state=state
            )
            predictions.append(prediction)
            diagnostics.append(detail)
        return predictions, diagnostics

    @torch.no_grad()
    def calibrate_sample(self, sample: dict[str, Any], momentum: float = -1.0) -> None:
        if momentum != -1.0:
            raise ValueError(
                "ASGCN paper-core calibration uses exact feature-wise maxima; momentum must be -1"
            )
        graph = self._graph(sample)
        _, activations = self.encoder.forward_ann(graph, return_activations=True)
        self.encoder.update_activation_maxima(activations)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.encoder.fold_batch_norm()

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        self.encoder.reset_activation_maxima()

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        self.encoder.apply_parameter_normalization()

    def calibration_summary(self) -> dict[str, list[int] | int]:
        return self.encoder.calibration_summary()
