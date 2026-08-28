from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import ASGCNEncoder, EventGraph, build_event_graph


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
        torch.ones(
            (linear.numel(), 1), device=features.device, dtype=features.dtype
        ),
    )
    raster = raster / counts.clamp_min(1.0)
    return raster.transpose(0, 1).reshape(1, features.shape[-1], grid_h, grid_w)


class ASGCNReconstructor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        graph_layers: int = 3,
        causal_candidates: int = 32,
        spatial_radius: float = 0.12,
        temporal_radius: float = 0.30,
        raster_downsample: int = 4,
        decoder_channels: int = 48,
        output_channels: int = 1,
        recurrent: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ASGCNEncoder(hidden_dim, graph_layers)
        self.decoder = RasterDecoder(hidden_dim, decoder_channels, output_channels, recurrent)
        self.causal_candidates = int(causal_candidates)
        self.spatial_radius = float(spatial_radius)
        self.temporal_radius = float(temporal_radius)
        self.raster_downsample = int(raster_downsample)

    def _graph(self, sample: dict[str, Any]) -> EventGraph:
        return build_event_graph(
            sample["events"],
            sample["sensor_size"],
            self.causal_candidates,
            self.spatial_radius,
            self.temporal_radius,
        )

    def forward_sample(
        self,
        sample: dict[str, Any],
        inference_mode: str = "ann",
        simulation_steps: int = 16,
        return_activations: bool = False,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        graph = self._graph(sample)
        if inference_mode == "ann":
            features, activations = self.encoder.forward_ann(graph, return_activations)
            firing_rates: list[torch.Tensor] = []
        elif inference_mode == "snn":
            features, firing_rates = self.encoder.forward_snn(graph, simulation_steps)
            activations = []
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}")
        raster = rasterize_features(
            features, graph, sample["sensor_size"], self.raster_downsample
        )
        prediction, next_state = self.decoder(raster, sample["sensor_size"], recurrent_state)
        diagnostics = {
            "nodes": int(graph.node_features.shape[0]),
            "edges": int(graph.edge_index.shape[1]),
            "firing_rates": firing_rates,
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
        _, diagnostics = self.forward_sample(sample, return_activations=True)
        self.encoder.update_thresholds(diagnostics["activations"], momentum=momentum)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.encoder.fold_batch_norm()
