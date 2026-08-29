from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

PAPER_CORE_VERSION = 2


@dataclass
class EventGraph:
    node_features: torch.Tensor
    positions: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor


def _safe_batch_norm(norm: nn.BatchNorm1d, values: torch.Tensor) -> torch.Tensor:
    """Use running statistics when a graph has fewer than two events."""
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


def uniformly_sample_events(events: torch.Tensor, factor: int = 1) -> torch.Tensor:
    """Apply the paper's deterministic event sampling factor R."""
    if isinstance(factor, bool) or int(factor) != factor:
        raise ValueError("event_sampling_factor must be an integer")
    factor = int(factor)
    if factor < 1:
        raise ValueError("event_sampling_factor must be at least 1")
    return events[::factor]


def prepare_event_nodes(
    events: torch.Tensor, sensor_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Eq. (9)-(10) event nodes while retaining temporal order.

    The paper does not specify coordinate normalization. This implementation records
    the explicit assumption x,y,t in [0,1] and polarity in {-1,+1}. Graph distances
    use x,y,t by default; polarity remains a node feature.
    """
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have shape [N,4] with x,y,t,p columns")
    height, width = (int(value) for value in sensor_size)
    if height < 1 or width < 1:
        raise ValueError("sensor_size must contain positive height and width")
    if events.numel() == 0:
        return (
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
            torch.empty((0, 4), device=events.device, dtype=torch.float32),
        )
    events = events.float()
    if not bool(torch.isfinite(events).all()):
        raise ValueError("Event coordinates, timestamps, and polarities must be finite")
    if bool((events[1:, 2] < events[:-1, 2]).any()):
        raise ValueError("Event timestamps must be monotonically non-decreasing")
    x = events[:, 0] / max(width - 1, 1)
    y = events[:, 1] / max(height - 1, 1)
    t = events[:, 2]
    t = (t - t[0]) / (t[-1] - t[0]).abs().clamp_min(1e-6)
    polarity = torch.where(events[:, 3] > 0, 1.0, -1.0)
    polarity_position = (polarity + 1.0) * 0.5
    positions = torch.stack((x, y, t, polarity_position), dim=-1)
    node_features = torch.stack((x, y, t, polarity), dim=-1)
    return node_features, positions


def build_radius_graph(
    positions: torch.Tensor,
    radius: float,
    *,
    position_dims: int = 3,
    chunk_size: int = 512,
    max_edges: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the paper's exact radius graph with a uniform-cell candidate search.

    Every ordered edge direction is materialized so source-to-target aggregation is
    equivalent to an undirected graph.  Cells have width ``radius``; therefore only
    the 3^d adjacent cells can contain a valid neighbor.  Exact Euclidean filtering
    after candidate generation preserves the brute-force graph while avoiding the
    O(N^2) distance matrix on sparse event volumes.
    """
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("graph_radius must be positive")
    position_dims = int(position_dims)
    if position_dims < 1 or position_dims > positions.shape[1]:
        raise ValueError("graph_position_dims must select available position columns")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("graph_chunk_size must be at least 1")
    if max_edges is not None:
        if isinstance(max_edges, bool) or int(max_edges) != max_edges:
            raise ValueError("max_graph_edges must be an integer or null")
        max_edges = int(max_edges)
        if max_edges < 1:
            raise ValueError("max_graph_edges must be at least 1 or null")

    count = int(positions.shape[0])
    device = positions.device
    if count == 0:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )

    coordinates = positions[:, :position_dims]
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("Graph coordinates must be finite")
    if bool(((coordinates < 0) | (coordinates > 1)).any()):
        raise ValueError("Normalized graph coordinates must lie in [0,1]")

    cells_per_axis = max(2, math.ceil(1.0 / radius) + 1)
    if cells_per_axis**position_dims >= torch.iinfo(torch.long).max:
        raise ValueError("graph_radius is too small for integer spatial hashing")
    strides = torch.tensor(
        [cells_per_axis**dimension for dimension in range(position_dims)],
        device=device,
        dtype=torch.long,
    )
    cells = torch.floor(coordinates / radius).to(torch.long)
    cells = cells.clamp_(0, cells_per_axis - 1)
    cell_hash = (cells * strides).sum(dim=1)
    sorted_hash, sorted_nodes = torch.sort(cell_hash)
    offset_axis = torch.tensor((-1, 0, 1), device=device, dtype=torch.long)
    offsets = torch.cartesian_prod(*([offset_axis] * position_dims)).reshape(
        -1, position_dims
    )

    sources: list[torch.Tensor] = []
    destination_chunks: list[torch.Tensor] = []
    distances_kept: list[torch.Tensor] = []
    retained_edge_count = 0
    # Bound worst-case candidate materialization even if every event occupies one cell.
    effective_chunk_size = min(chunk_size, max(1, 4_000_000 // count))
    for start in range(0, count, effective_chunk_size):
        stop = min(start + effective_chunk_size, count)
        local_sources = torch.arange(start, stop, device=device, dtype=torch.long)
        neighbor_cells = cells[start:stop, None, :] + offsets[None, :, :]
        valid_cells = ((neighbor_cells >= 0) & (neighbor_cells < cells_per_axis)).all(dim=2)
        neighbor_hashes = (neighbor_cells * strides).sum(dim=2)
        candidate_sources = local_sources[:, None].expand_as(neighbor_hashes)[valid_cells]
        candidate_hashes = neighbor_hashes[valid_cells]
        left = torch.searchsorted(sorted_hash, candidate_hashes, right=False)
        right = torch.searchsorted(sorted_hash, candidate_hashes, right=True)
        counts = right - left
        nonempty = counts > 0
        if not bool(nonempty.any()):
            continue
        candidate_sources = candidate_sources[nonempty]
        left = left[nonempty]
        counts = counts[nonempty]
        candidate_count = int(counts.sum().item())
        expanded_sources = torch.repeat_interleave(
            candidate_sources, counts, output_size=candidate_count
        )
        expanded_left = torch.repeat_interleave(left, counts, output_size=candidate_count)
        starts = counts.cumsum(0) - counts
        within_group = torch.arange(candidate_count, device=device) - torch.repeat_interleave(
            starts, counts, output_size=candidate_count
        )
        candidate_destinations = sorted_nodes[expanded_left + within_group]
        candidate_distances = torch.linalg.vector_norm(
            coordinates[expanded_sources] - coordinates[candidate_destinations], dim=1
        )
        within_radius = (expanded_sources != candidate_destinations) & (
            candidate_distances < radius
        )
        chunk_edge_count = int(within_radius.sum().item())
        if (
            max_edges is not None
            and retained_edge_count + chunk_edge_count > max_edges
        ):
            raise RuntimeError(
                "Radius graph exceeded max_graph_edges="
                f"{max_edges:,} while processing {count:,} nodes. Reduce "
                "graph_radius/max_events or raise the explicit memory guard after "
                "measuring accelerator memory."
            )
        if chunk_edge_count == 0:
            continue
        retained_edge_count += chunk_edge_count
        sources.append(expanded_sources[within_radius])
        destination_chunks.append(candidate_destinations[within_radius])
        distances_kept.append(candidate_distances[within_radius])

    if not sources:
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0, 1), device=device, dtype=positions.dtype),
        )
    source = torch.cat(sources)
    destination = torch.cat(destination_chunks)
    distance = torch.cat(distances_kept)
    order = torch.argsort(source * count + destination)
    source = source[order]
    destination = destination[order]
    distance = distance[order]
    edge_index = torch.stack((source, destination), dim=0)
    edge_attr = (distance / radius).clamp(0.0, 1.0).unsqueeze(-1)
    return edge_index, edge_attr


def build_event_graph(
    events: torch.Tensor,
    sensor_size: tuple[int, int],
    *,
    event_sampling_factor: int,
    graph_radius: float,
    graph_position_dims: int,
    graph_chunk_size: int,
    max_graph_edges: int | None = None,
) -> EventGraph:
    sampled = uniformly_sample_events(events, event_sampling_factor)
    node_features, positions = prepare_event_nodes(sampled, sensor_size)
    edge_index, edge_attr = build_radius_graph(
        positions,
        graph_radius,
        position_dims=graph_position_dims,
        chunk_size=graph_chunk_size,
        max_edges=max_graph_edges,
    )
    return EventGraph(node_features, positions, edge_index, edge_attr)


def linear_open_bspline_basis(
    pseudo: torch.Tensor, kernel_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return indices and weights for an open degree-1 B-spline basis.

    For each scalar pseudo-coordinate this yields the two non-zero basis terms.
    The weights form a partition of unity, including exact endpoints.
    """
    kernel_size = int(kernel_size)
    if kernel_size < 2:
        raise ValueError("spline_kernel_size must be at least 2")
    if pseudo.ndim != 2 or pseudo.shape[1] != 1:
        raise ValueError("ASGCN edge pseudo-coordinates must have shape [E,1]")
    if not torch.isfinite(pseudo).all():
        raise ValueError("ASGCN edge pseudo-coordinates must be finite")
    if bool(((pseudo < 0) | (pseudo > 1)).any()):
        raise ValueError("ASGCN edge pseudo-coordinates must lie in [0,1]")

    scaled = pseudo[:, 0] * float(kernel_size - 1)
    cell = torch.floor(scaled)
    left = cell.to(torch.long).remainder(kernel_size)
    right = (left + 1).remainder(kernel_size)
    right_weight = scaled - cell
    left_weight = 1.0 - right_weight
    indices = torch.stack((left, right), dim=-1)
    weights = torch.stack((left_weight, right_weight), dim=-1)
    return indices, weights


class PaperSplineConv(nn.Module):
    """Eq. (11) weighted open B-spline convolution followed by BN.

    ASGCN names a PyG weighted B-spline tensor-product kernel but omits its
    hyperparameters. The reconstruction adaptation therefore exposes them in every
    config. Degree 1 is implemented directly in PyTorch and matches the public
    SplineCNN operator definition for scalar pseudo-coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 5,
        degree: int = 1,
        root_weight: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if int(degree) != 1:
            raise ValueError("Only the configured open degree-1 B-spline is supported")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.degree = int(degree)
        if self.kernel_size < 2:
            raise ValueError("spline_kernel_size must be at least 2")
        self.weight = nn.Parameter(
            torch.empty(self.kernel_size, self.in_channels, self.out_channels)
        )
        self.root = (
            nn.Parameter(torch.empty(self.in_channels, self.out_channels)) if root_weight else None
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.norm = nn.BatchNorm1d(self.out_channels)
        self.register_buffer("bn_bypassed", torch.tensor(False), persistent=True)
        self.register_buffer("snn_normalized", torch.tensor(False), persistent=True)
        self.register_buffer("activation_max", torch.ones(self.out_channels), persistent=True)
        self.register_buffer("threshold", torch.ones(self.out_channels), persistent=True)
        self._bn_is_folded = False
        self._snn_is_normalized = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_bound = 1.0 / math.sqrt(self.kernel_size * self.in_channels)
        nn.init.uniform_(self.weight, -weight_bound, weight_bound)
        if self.root is not None:
            root_bound = 1.0 / math.sqrt(self.in_channels)
            nn.init.uniform_(self.root, -root_bound, root_bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.norm.reset_parameters()
        self.bn_bypassed.fill_(False)
        self.snn_normalized.fill_(False)
        self.activation_max.fill_(1.0)
        self.threshold.fill_(1.0)
        self._bn_is_folded = False
        self._snn_is_normalized = False

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
        self._snn_is_normalized = bool(self.snn_normalized.item())

    def spline_aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        source, destination = edge_index
        output = torch.zeros((x.shape[0], self.out_channels), device=x.device, dtype=x.dtype)
        if source.numel() > 0:
            indices, basis = (
                basis_cache
                if basis_cache is not None
                else linear_open_bspline_basis(edge_attr, self.kernel_size)
            )
            # Project nodes once for every control point, then gather only the two
            # degree-1 basis terms that are active on each edge.
            projected = torch.einsum("ni,kio->nko", x, self.weight)
            for active_basis in range(2):
                messages = projected[source, indices[:, active_basis]]
                messages = messages * basis[:, active_basis, None].to(messages.dtype)
                # CPU autocast can produce bfloat16 projections while ``x``
                # (and therefore ``output``) remains float32. ``index_add_``
                # requires matching dtypes, so accumulate in the output dtype.
                output.index_add_(0, destination, messages.to(output.dtype))
            degree = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
            degree.index_add_(
                0,
                destination,
                torch.ones((destination.numel(), 1), device=x.device, dtype=x.dtype),
            )
            output = output / degree.clamp_min(1.0)
        return output

    def affine(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = self.spline_aggregate(x, edge_index, edge_attr, basis_cache)
        if self.root is not None:
            output = output + x @ self.root
        if self.bias is not None:
            output = output + self.bias
        return output

    def preactivation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        output = self.affine(x, edge_index, edge_attr, basis_cache)
        return output if self._bn_is_folded else _safe_batch_norm(self.norm, output)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        basis_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preactivation = self.preactivation(x, edge_index, edge_attr, basis_cache)
        return torch.relu(preactivation), preactivation

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        """Fold Eq. (13) into every output term of Eq. (11), yielding Eq. (14)."""
        if self._bn_is_folded:
            return
        if self.training:
            raise RuntimeError("Call eval() before folding BatchNorm")
        scale = self.norm.weight / torch.sqrt(self.norm.running_var + self.norm.eps)
        self.weight.mul_(scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(scale.view(1, -1))
        if self.bias is None:
            raise RuntimeError("BN folding requires an affine convolution bias")
        self.bias.copy_((self.bias - self.norm.running_mean) * scale + self.norm.bias)
        self.bn_bypassed.fill_(True)
        self._bn_is_folded = True

    @torch.no_grad()
    def apply_parameter_normalization(
        self, input_scale: torch.Tensor, output_scale: torch.Tensor
    ) -> None:
        """Apply Eq. (6): W_l <- W_l lambda_(l-1)/lambda_l, b_l <- b_l/lambda_l."""
        if not self._bn_is_folded:
            raise RuntimeError("Fold BatchNorm before ANN-to-SNN parameter normalization")
        if self._snn_is_normalized:
            raise RuntimeError("ANN-to-SNN parameter normalization was already applied")
        input_scale = input_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        output_scale = output_scale.to(device=self.weight.device, dtype=self.weight.dtype)
        if input_scale.shape != (self.in_channels,):
            raise ValueError("Input activation scale does not match spline input channels")
        if output_scale.shape != (self.out_channels,):
            raise ValueError("Output activation scale does not match spline output channels")
        input_scale = input_scale.clamp_min(1e-6)
        output_scale = output_scale.clamp_min(1e-6)
        self.weight.mul_(input_scale.view(1, -1, 1))
        self.weight.div_(output_scale.view(1, 1, -1))
        if self.root is not None:
            self.root.mul_(input_scale.view(-1, 1))
            self.root.div_(output_scale.view(1, -1))
        if self.bias is not None:
            self.bias.div_(output_scale)
        self.activation_max.copy_(output_scale)
        self.threshold.fill_(1.0)
        self.snn_normalized.fill_(True)
        self._snn_is_normalized = True


class ASGCNEncoder(nn.Module):
    """Equation-faithful ASGCN graph core adapted to a reconstruction decoder."""

    def __init__(
        self,
        hidden_dim: int = 64,
        graph_layers: int = 6,
        *,
        spline_kernel_size: int = 5,
        spline_degree: int = 1,
        spline_root_weight: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        graph_layers = int(graph_layers)
        if graph_layers < 1:
            raise ValueError("graph_layers must be at least 1 for ASGCN")
        self.hidden_dim = hidden_dim
        channels = [4] + [hidden_dim] * graph_layers
        self.layers = nn.ModuleList(
            [
                PaperSplineConv(
                    channels[index],
                    channels[index + 1],
                    kernel_size=spline_kernel_size,
                    degree=spline_degree,
                    root_weight=spline_root_weight,
                    bias=True,
                )
                for index in range(graph_layers)
            ]
        )
        self.register_buffer(
            "calibration_samples_seen",
            torch.zeros(graph_layers, dtype=torch.long),
            persistent=True,
        )

    def _basis_cache(self, graph: EventGraph) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute the fixed edge basis once for all layers and IF timesteps."""
        if graph.edge_attr.shape[0] == 0:
            return None
        return linear_open_bspline_basis(
            graph.edge_attr,
            self.layers[0].kernel_size,
        )

    def forward_ann(
        self, graph: EventGraph, return_activations: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden = graph.node_features
        activations: list[torch.Tensor] = []
        basis_cache = self._basis_cache(graph)
        for layer in self.layers:
            hidden, preactivation = layer(
                hidden,
                graph.edge_index,
                graph.edge_attr,
                basis_cache,
            )
            if return_activations:
                activations.append(torch.relu(preactivation))
        return hidden, activations

    def forward_snn(
        self,
        graph: EventGraph,
        simulation_steps: int = 16,
        dynamics: str = "literal_eq15",
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run explicit IF timesteps using literal Eq. (15) or a standard-IF control."""
        if isinstance(simulation_steps, bool) or int(simulation_steps) != simulation_steps:
            raise ValueError("simulation_steps must be an integer")
        simulation_steps = int(simulation_steps)
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be at least 1")
        if dynamics not in {"literal_eq15", "standard_if"}:
            raise ValueError("snn_dynamics must be 'literal_eq15' or 'standard_if'")
        if any(not layer._snn_is_normalized for layer in self.layers):
            raise RuntimeError("SNN inference requires Eq. (6) parameter normalization")
        node_count = int(graph.node_features.shape[0])
        if node_count == 0:
            empty = graph.node_features.new_empty((0, self.hidden_dim))
            zeros = [graph.node_features.new_zeros(()) for _ in self.layers]
            return empty, zeros

        membranes = [
            layer.threshold.to(graph.node_features).expand(node_count, -1).clone() * 0.5
            for layer in self.layers
        ]
        previous_spikes = [
            graph.node_features.new_zeros((node_count, layer.out_channels)) for layer in self.layers
        ]
        spike_sums = [torch.zeros_like(spikes) for spikes in previous_spikes]
        active_counts = [graph.node_features.new_zeros(()) for _ in self.layers]
        basis_cache = self._basis_cache(graph)

        for _ in range(simulation_steps):
            hidden = graph.node_features
            for index, layer in enumerate(self.layers):
                current = layer.affine(
                    hidden,
                    graph.edge_index,
                    graph.edge_attr,
                    basis_cache,
                )
                integrated = membranes[index] + current
                if dynamics == "literal_eq15":
                    # This is the paper's written +h_i^l(t-1) recurrence. It is
                    # intentionally separate from the standard rate-conversion IF
                    # control because the paper does not resolve their mismatch.
                    integrated = integrated + previous_spikes[index]
                threshold = layer.threshold.to(integrated).expand_as(integrated)
                spikes = torch.where(
                    integrated >= threshold, threshold, torch.zeros_like(integrated)
                )
                membranes[index] = integrated - spikes
                previous_spikes[index] = spikes
                spike_sums[index] = spike_sums[index] + spikes
                active_counts[index] = active_counts[index] + (spikes != 0).sum()
                hidden = spikes

        firing_rates = [
            active.to(graph.node_features.dtype) / float(simulation_steps * max(1, spikes.numel()))
            for active, spikes in zip(active_counts, spike_sums, strict=True)
        ]
        return spike_sums[-1] / float(simulation_steps), firing_rates

    @torch.no_grad()
    def update_activation_maxima(self, activations: list[torch.Tensor]) -> None:
        if len(activations) != len(self.layers):
            raise ValueError("Activation count does not match graph layer count")
        for index, (layer, activation) in enumerate(zip(self.layers, activations, strict=True)):
            if activation.numel() == 0:
                continue
            if activation.ndim != 2 or activation.shape[1] != layer.out_channels:
                raise ValueError("Calibration activation shape does not match graph layer")
            if not bool(torch.isfinite(activation).all()):
                raise FloatingPointError("Non-finite activation encountered during calibration")
            maxima = activation.amax(dim=0)
            layer.activation_max.copy_(torch.maximum(layer.activation_max, maxima))
            self.calibration_samples_seen[index].add_(1)

    @torch.no_grad()
    def reset_activation_maxima(self) -> None:
        for layer in self.layers:
            layer.activation_max.zero_()
        self.calibration_samples_seen.zero_()

    @torch.no_grad()
    def fold_batch_norm(self) -> None:
        self.eval()
        for layer in self.layers:
            layer.fold_batch_norm()

    @torch.no_grad()
    def apply_parameter_normalization(self) -> None:
        missing = torch.nonzero(self.calibration_samples_seen == 0).flatten().tolist()
        if missing:
            raise RuntimeError(
                "Cannot apply Eq. (6): no non-empty calibration activation for layer(s) "
                + ", ".join(str(index) for index in missing)
            )
        previous_scale = self.layers[0].weight.new_ones(self.layers[0].in_channels)
        for layer in self.layers:
            measured = layer.activation_max.detach().clone()
            # A ReLU channel that stayed identically zero has no usable lambda.
            # Keep it at unit scale instead of dividing its parameters by epsilon.
            output_scale = torch.where(measured > 0, measured, torch.ones_like(measured))
            layer.apply_parameter_normalization(previous_scale, output_scale)
            previous_scale = output_scale

    def calibration_summary(self) -> dict[str, list[int] | int]:
        dead_channels = [int((layer.activation_max <= 0).sum().item()) for layer in self.layers]
        valid_samples = [int(value) for value in self.calibration_samples_seen.tolist()]
        return {
            "valid_samples_per_layer": valid_samples,
            "minimum_valid_samples": min(valid_samples, default=0),
            "dead_channels_per_layer": dead_channels,
        }

    def output_activation_scale(self, reference: torch.Tensor) -> torch.Tensor:
        """Return lambda_L used to express spikes in the analog decoder's units."""
        if not self.layers[-1]._snn_is_normalized:
            raise RuntimeError("Output activation scale is available after Eq. (6) conversion")
        return self.layers[-1].activation_max.to(reference)
