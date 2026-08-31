from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from .batching import concatenate_graphs, sequence_key
from .graph import PAPER_CORE_VERSION, ASGCNEncoder, EventGraph, build_event_graph
from .unet import RecurrentUNetDecoder


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


class ASGCNUNet(nn.Module):
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
        spline_chunk_size: int | None = 65_536,
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
            spline_chunk_size=spline_chunk_size,
        )
        self.decoder = RecurrentUNetDecoder(
            hidden_dim, decoder_channels, output_channels, recurrent
        )
        self.register_buffer(
            "calibration_attempts",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "calibration_commitment_digest",
            torch.zeros(32, dtype=torch.uint8),
            persistent=True,
        )
        self.register_buffer(
            "calibration_commitment_sealed",
            torch.tensor(False),
            persistent=True,
        )
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
        dataset_sampling_ratio = float(
            sample.get("metadata", {}).get("dataset_sampling_ratio", 1.0)
        )
        node_count = int(graph.node_features.shape[0])
        edge_count = int(graph.edge_index.shape[1])
        if node_count:
            assert graph.in_degree is not None
            degree = graph.in_degree
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
            "dataset_sampling_ratio": dataset_sampling_ratio,
            "effective_sampling_ratio": (dataset_sampling_ratio * self.event_sampling_factor),
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

    def forward_training_batch(
        self,
        samples: list[dict[str, Any]],
        recurrent_states: list[torch.Tensor | None] | None = None,
        *,
        timing: Any = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Train independent sequence frames with one encoder and decoder call.

        This opt-in ANN path uses pooled-node graph BatchNorm in training mode;
        it is a distinct minibatch protocol, not gradient accumulation or a claim
        of equivalence to sequential batch-one parameter updates. No graph edge
        or recurrent state crosses a sample boundary.
        """
        if not samples:
            raise ValueError("Training batches must contain at least one sample")
        sensor_size = tuple(int(value) for value in samples[0]["sensor_size"])
        if len(sensor_size) != 2 or min(sensor_size) < 1:
            raise ValueError("Training batches require positive height and width")
        if any(tuple(sample["sensor_size"]) != sensor_size for sample in samples):
            raise ValueError("A training batch must contain one shared sensor_size")
        keys = [sequence_key(sample) for sample in samples]
        if len(set(keys)) != len(keys):
            raise ValueError("A training batch may contain only one frame from each sequence")
        if recurrent_states is None:
            recurrent_states = [None] * len(samples)
        if len(recurrent_states) != len(samples):
            raise ValueError("recurrent_states must contain one entry per batch sample")
        gpu = samples[0]["events"].device.type == "cuda"

        def scope(label: str):
            return timing.scope(label, gpu=gpu) if timing is not None else nullcontext()

        with scope("graph"):
            graphs = [self._graph(sample) for sample in samples]
            graph = concatenate_graphs(graphs)
            node_counts = [item.node_features.shape[0] for item in graphs]
        with scope("encoder"):
            features, _activations = self.encoder.forward_ann(graph)
        with scope("decoder"):
            per_sample_features = features.split(node_counts, dim=0)
            raster = torch.cat([
                rasterize_features(values, item, sensor_size, self.raster_downsample)
                for values, item in zip(per_sample_features, graphs, strict=True)
            ], dim=0)
            state_batch = None
            if self.decoder.recurrent is not None:
                # Two stride-two, padding-one downsampling convolutions each
                # round upward, so the recurrent grid is ceil(raster_size / 4).
                expected = (
                    1,
                    self.decoder.recurrent.hidden_channels,
                    (raster.shape[-2] + 3) // 4,
                    (raster.shape[-1] + 3) // 4,
                )
                valid_states = []
                for state in recurrent_states:
                    if state is not None and not isinstance(state, torch.Tensor):
                        raise TypeError("A recurrent state must be a tensor or None")
                    valid_states.append(
                        state if state is not None and tuple(state.shape) == expected else None
                    )
                reference = next((state for state in valid_states if state is not None), None)
                if reference is not None:
                    if reference.device != raster.device or any(
                        state is not None
                        and (state.device != reference.device or state.dtype != reference.dtype)
                        for state in valid_states
                    ):
                        raise ValueError("Batched recurrent states must share device and dtype")
                    state_batch = torch.cat([
                        reference.new_zeros(expected) if state is None else state
                        for state in valid_states
                    ], dim=0)
            predictions, next_state = self.decoder(raster, sensor_size, state_batch)
        diagnostics = []
        for index, (sample, item) in enumerate(zip(samples, graphs, strict=True)):
            nodes = int(item.node_features.shape[0])
            assert item.in_degree is not None
            isolated = (item.in_degree == 0).sum()
            maximum = item.in_degree.max() if nodes else item.in_degree.new_zeros(())
            sampling_ratio = float(sample.get("metadata", {}).get("dataset_sampling_ratio", 1.0))
            diagnostics.append({
                "paper_core_version": self.architecture_version,
                "nodes": nodes,
                "edges": int(item.edge_index.shape[1]),
                "isolated_nodes": isolated,
                "isolate_ratio": isolated.to(item.node_features.dtype) / float(max(1, nodes)),
                "max_degree": maximum,
                "edge_feature": "normalized_scalar_distance",
                "event_sampling_factor": self.event_sampling_factor,
                "dataset_sampling_ratio": sampling_ratio,
                "effective_sampling_ratio": sampling_ratio * self.event_sampling_factor,
                "snn_dynamics": None,
                "decoder_input_lambda_applied": False,
                "firing_rates": [],
                "firing_rate_denominators": [],
                "spike_counts": [],
                "activations": [],
                "recurrent_state": None if next_state is None else next_state[index : index + 1],
            })
        return predictions, diagnostics

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
        self.calibration_attempts.add_(1)

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.encoder.fold_batch_norm()

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        self.encoder.reset_activation_maxima()
        self.calibration_attempts.zero_()
        self.calibration_commitment_digest.zero_()
        self.calibration_commitment_sealed.fill_(False)

    @torch.no_grad()
    def seal_calibration_commitment(
        self,
        attempted_samples: int,
        commitment_sha256: str,
    ) -> None:
        if (
            isinstance(attempted_samples, bool)
            or not isinstance(attempted_samples, int)
            or attempted_samples < 1
        ):
            raise ValueError("attempted calibration sample count must be a positive integer")
        if int(self.calibration_attempts.item()) != attempted_samples:
            raise RuntimeError(
                "Persistent calibration attempt count differs from the selected samples"
            )
        try:
            digest = bytes.fromhex(commitment_sha256)
        except (TypeError, ValueError) as error:
            raise ValueError("calibration commitment digest must be SHA-256") from error
        if len(digest) != 32 or commitment_sha256 != commitment_sha256.lower():
            raise ValueError("calibration commitment digest must be lowercase SHA-256")
        self.calibration_commitment_digest.copy_(
            torch.tensor(
                list(digest),
                dtype=torch.uint8,
                device=self.calibration_commitment_digest.device,
            )
        )
        self.calibration_commitment_sealed.fill_(True)

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        self.encoder.apply_parameter_normalization()

    def calibration_summary(self) -> dict[str, list[int] | int | str | bool | None]:
        summary: dict[str, list[int] | int | str | bool | None] = (
            self.encoder.calibration_summary()
        )
        commitment_sealed = bool(self.calibration_commitment_sealed.item())
        summary.update(
            {
                "attempted_samples": int(self.calibration_attempts.item()),
                "calibration_commitment_sha256": (
                    bytes(self.calibration_commitment_digest.tolist()).hex()
                    if commitment_sealed
                    else None
                ),
                "commitment_sealed": commitment_sealed,
            }
        )
        return summary
