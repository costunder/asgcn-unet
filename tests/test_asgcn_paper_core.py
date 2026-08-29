from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from asgcn_recon.graph import (
    ASGCNEncoder,
    EventGraph,
    PaperSplineConv,
    build_event_graph,
    build_radius_graph,
    linear_open_bspline_basis,
    prepare_event_nodes,
    uniformly_sample_events,
)
from asgcn_recon.model import ASGCNReconstructor


def _single_node_graph() -> EventGraph:
    return EventGraph(
        node_features=torch.zeros((1, 4)),
        positions=torch.zeros((1, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
    )


def _normalized_single_layer_encoder() -> ASGCNEncoder:
    encoder = ASGCNEncoder(
        hidden_dim=1,
        graph_layers=1,
        spline_kernel_size=2,
        spline_root_weight=True,
    ).eval()
    encoder.fold_batch_norm()
    encoder.layers[0].activation_max.fill_(1.0)
    encoder.calibration_samples_seen.fill_(1)
    encoder.apply_parameter_normalization()
    return encoder


def test_uniform_event_sampling_and_polarity_encoding() -> None:
    events = torch.tensor(
        [[index, 0, index, index % 2] for index in range(10)], dtype=torch.float32
    )
    sampled = uniformly_sample_events(events, factor=3)
    torch.testing.assert_close(sampled, events[[0, 3, 6, 9]])
    with pytest.raises(ValueError, match="at least 1"):
        uniformly_sample_events(events, factor=0)

    node_features, positions = prepare_event_nodes(events[:2], sensor_size=(2, 10))
    torch.testing.assert_close(node_features[:, 3], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(positions[:, 3], torch.tensor([0.0, 1.0]))

    graph = build_event_graph(
        events,
        (2, 10),
        event_sampling_factor=3,
        graph_radius=2.0,
        graph_position_dims=3,
        graph_chunk_size=2,
    )
    assert graph.node_features.shape == (4, 4)
    torch.testing.assert_close(graph.node_features[:, 0], sampled[:, 0] / 9.0)


def test_radius_graph_is_exactly_undirected_without_self_edges() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 1.0],
            [0.9, 0.0, 0.0, 0.0],
        ]
    )
    edge_index, edge_attr = build_radius_graph(
        positions,
        radius=0.5,
        position_dims=3,
        chunk_size=2,
    )

    torch.testing.assert_close(edge_index, torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(edge_attr, torch.tensor([[0.6], [0.6]]))
    assert edge_attr.shape == (2, 1)
    assert torch.all(edge_index[0] != edge_index[1])
    assert torch.all((0.0 <= edge_attr) & (edge_attr <= 1.0))

    boundary_edges, _ = build_radius_graph(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        radius=0.5,
        position_dims=3,
        chunk_size=1,
    )
    assert boundary_edges.shape == (2, 0), "The paper uses distance < D, not <= D"

    with pytest.raises(RuntimeError, match="max_graph_edges=5"):
        build_radius_graph(
            torch.zeros((4, 3)),
            radius=1.0,
            position_dims=3,
            chunk_size=2,
            max_edges=5,
        )


def test_linear_open_bspline_endpoints_and_partition_of_unity() -> None:
    pseudo = torch.tensor([[0.0], [0.125], [0.5], [0.875], [1.0]])
    indices, weights = linear_open_bspline_basis(pseudo, kernel_size=5)

    torch.testing.assert_close(indices[0], torch.tensor([0, 1]))
    torch.testing.assert_close(weights[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(indices[-1], torch.tensor([4, 0]))
    torch.testing.assert_close(weights[-1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(5))
    assert torch.all(weights >= 0)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        linear_open_bspline_basis(torch.tensor([[1.01]]), kernel_size=5)


def test_linear_bspline_exact_endpoint_matches_official_pyg_pseudo_gradient() -> None:
    """The official degree-1 backend wraps the inactive right basis at u=1."""
    layer = PaperSplineConv(
        1,
        1,
        kernel_size=2,
        root_weight=False,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[3.0]], [[5.0]]]))
    features = torch.tensor([[8.0]])
    edge_index = torch.tensor([[0], [0]])
    edge_attr = torch.tensor([[1.0]], requires_grad=True)

    output = layer.spline_aggregate(features, edge_index, edge_attr)
    output.sum().backward()

    torch.testing.assert_close(output, torch.tensor([[40.0]]))
    # At u=1 PyG uses bases [K-1, 0], so d/du = x * (W_0 - W_1).
    torch.testing.assert_close(edge_attr.grad, torch.tensor([[-16.0]]))


def test_spline_parameter_initialization_matches_official_pyg_bounds() -> None:
    layer = PaperSplineConv(4, 7, kernel_size=5, root_weight=True, bias=True)
    spline_bound = 1.0 / (5 * 4) ** 0.5
    root_bound = 1.0 / 4**0.5

    assert torch.all(layer.weight.abs() <= spline_bound)
    assert layer.root is not None
    assert torch.all(layer.root.abs() <= root_bound)
    torch.testing.assert_close(layer.bias, torch.zeros(7))


def test_spline_mean_aggregation_matches_hand_calculation_and_gradients() -> None:
    layer = PaperSplineConv(
        1,
        1,
        kernel_size=2,
        root_weight=False,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[3.0]], [[5.0]]]))
    features = torch.tensor([[2.0], [4.0], [8.0]], requires_grad=True)
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 2]])
    edge_attr = torch.tensor([[0.0], [1.0], [0.25]])

    output = layer.spline_aggregate(features, edge_index, edge_attr)
    torch.testing.assert_close(output, torch.tensor([[0.0], [23.0], [14.0]]))
    output.sum().backward()

    torch.testing.assert_close(features.grad, torch.tensor([[1.5], [3.5], [2.5]]))
    torch.testing.assert_close(layer.weight.grad, torch.tensor([[[4.0]], [[5.0]]]))


def test_batch_norm_folding_preserves_preactivation() -> None:
    generator = torch.Generator().manual_seed(33)
    layer = PaperSplineConv(2, 3, kernel_size=3, root_weight=True, bias=True).eval()
    with torch.no_grad():
        layer.norm.running_mean.copy_(torch.tensor([0.3, -0.2, 0.7]))
        layer.norm.running_var.copy_(torch.tensor([0.5, 2.0, 4.0]))
        layer.norm.weight.copy_(torch.tensor([1.2, -0.7, 0.4]))
        layer.norm.bias.copy_(torch.tensor([-0.1, 0.5, 0.2]))
    features = torch.rand((4, 2), generator=generator)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    edge_attr = torch.tensor([[0.0], [0.25], [0.75], [1.0]])

    before = layer.preactivation(features, edge_index, edge_attr)
    layer.fold_batch_norm()
    after = layer.preactivation(features, edge_index, edge_attr)

    torch.testing.assert_close(after, before, atol=1e-6, rtol=1e-5)
    assert layer.bn_bypassed.item() is True
    assert layer._bn_is_folded is True


def test_equation_6_scales_kernel_root_and_bias_per_feature() -> None:
    layer = PaperSplineConv(2, 2, kernel_size=2, root_weight=True, bias=True).eval()
    with pytest.raises(RuntimeError, match="Fold BatchNorm"):
        layer.apply_parameter_normalization(torch.ones(2), torch.ones(2))

    layer.fold_batch_norm()
    folded_weight = layer.weight.detach().clone()
    folded_root = layer.root.detach().clone()
    folded_bias = layer.bias.detach().clone()
    input_scale = torch.tensor([2.0, 4.0])
    output_scale = torch.tensor([5.0, 10.0])
    layer.apply_parameter_normalization(input_scale, output_scale)

    torch.testing.assert_close(
        layer.weight,
        folded_weight * input_scale.view(1, -1, 1) / output_scale.view(1, 1, -1),
    )
    torch.testing.assert_close(
        layer.root,
        folded_root * input_scale.view(-1, 1) / output_scale.view(1, -1),
    )
    torch.testing.assert_close(layer.bias, folded_bias / output_scale)
    torch.testing.assert_close(layer.activation_max, output_scale)
    torch.testing.assert_close(layer.threshold, torch.ones(2))
    assert layer.snn_normalized.item() is True


def test_equation_6_requires_nonempty_calibration_and_uses_unit_for_dead_channels() -> None:
    empty_encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2).eval()
    empty_encoder.fold_batch_norm()
    empty_encoder.reset_activation_maxima()
    _, activations = empty_encoder.forward_ann(
        EventGraph(
            node_features=torch.empty((0, 4)),
            positions=torch.empty((0, 4)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
        ),
        return_activations=True,
    )
    empty_encoder.update_activation_maxima(activations)
    assert empty_encoder.calibration_summary()["minimum_valid_samples"] == 0
    with pytest.raises(RuntimeError, match="no non-empty calibration"):
        empty_encoder.apply_parameter_normalization()

    encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2).eval()
    encoder.fold_batch_norm()
    encoder.reset_activation_maxima()
    encoder.layers[0].activation_max.copy_(torch.tensor([2.0, 0.0]))
    encoder.calibration_samples_seen.fill_(1)
    encoder.apply_parameter_normalization()
    torch.testing.assert_close(
        encoder.layers[0].activation_max,
        torch.tensor([2.0, 1.0]),
    )


def test_explicit_if_uses_half_threshold_initialization_and_threshold_spikes() -> None:
    encoder = _normalized_single_layer_encoder()
    encoder.layers[0].threshold.fill_(2.0)
    current = torch.ones((1, 1))

    with patch.object(encoder.layers[0], "affine", return_value=current) as affine:
        output, firing_rates = encoder.forward_snn(_single_node_graph(), simulation_steps=1)

    # v(0)=theta/2=1 and I(1)=1 reaches theta exactly; the emitted spike is theta=2.
    torch.testing.assert_close(output, torch.tensor([[2.0]]))
    torch.testing.assert_close(firing_rates[0], torch.tensor(1.0))
    assert affine.call_count == 1


def test_explicit_if_loop_keeps_soft_reset_residual() -> None:
    encoder = _normalized_single_layer_encoder()
    currents = [torch.tensor([[0.8]]), torch.tensor([[-0.2]])]

    with patch.object(encoder.layers[0], "affine", side_effect=currents) as affine:
        output, firing_rates = encoder.forward_snn(_single_node_graph(), simulation_steps=2)

    # t1: 0.5+0.8 -> spike, residual 0.3. t2: 0.3-0.2+previous spike -> spike.
    # A hard reset to zero would not fire at t2.
    torch.testing.assert_close(output, torch.tensor([[1.0]]))
    torch.testing.assert_close(firing_rates[0], torch.tensor(1.0))
    assert affine.call_count == 2


def test_snn_reuses_one_spline_basis_across_all_timesteps() -> None:
    encoder = _normalized_single_layer_encoder()
    graph = EventGraph(
        node_features=torch.zeros((2, 4)),
        positions=torch.zeros((2, 4)),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.25], [0.25]]),
    )

    with patch(
        "asgcn_recon.graph.linear_open_bspline_basis",
        wraps=linear_open_bspline_basis,
    ) as basis:
        encoder.forward_snn(graph, simulation_steps=16)

    assert basis.call_count == 1


def test_literal_equation_15_and_standard_if_are_explicitly_distinct() -> None:
    encoder = _normalized_single_layer_encoder()
    current = torch.tensor([[0.1]])

    with patch.object(encoder.layers[0], "affine", return_value=current):
        literal_output, literal_rates = encoder.forward_snn(
            _single_node_graph(),
            simulation_steps=100,
            dynamics="literal_eq15",
        )
    with patch.object(encoder.layers[0], "affine", return_value=current):
        standard_output, standard_rates = encoder.forward_snn(
            _single_node_graph(),
            simulation_steps=100,
            dynamics="standard_if",
        )

    # The written +h(t-1) recurrence self-reinjects every prior spike. This
    # regression test prevents it from being silently presented as standard IF.
    torch.testing.assert_close(literal_output, torch.tensor([[0.96]]))
    torch.testing.assert_close(literal_rates[0], torch.tensor(0.96))
    torch.testing.assert_close(standard_output, torch.tensor([[0.1]]))
    torch.testing.assert_close(standard_rates[0], torch.tensor(0.1))

    with pytest.raises(ValueError, match="snn_dynamics"):
        encoder.forward_snn(_single_node_graph(), dynamics="ambiguous")


def test_snn_rejects_unnormalized_encoder_and_invalid_steps() -> None:
    encoder = ASGCNEncoder(hidden_dim=1, graph_layers=1, spline_kernel_size=2)
    graph = _single_node_graph()
    with pytest.raises(RuntimeError, match=r"Eq\. \(6\)"):
        encoder.forward_snn(graph, simulation_steps=2)
    with pytest.raises(ValueError, match="simulation_steps"):
        encoder.forward_snn(graph, simulation_steps=0)
    with pytest.raises(ValueError, match="integer"):
        encoder.forward_snn(graph, simulation_steps=1.9)
    with pytest.raises(ValueError, match="integer"):
        encoder.forward_snn(graph, simulation_steps=True)


def test_real_equation_6_standard_if_lambda_boundary_matches_ann_activation() -> None:
    encoder = ASGCNEncoder(
        hidden_dim=1,
        graph_layers=1,
        spline_kernel_size=2,
        spline_root_weight=True,
    ).eval()
    graph = EventGraph(
        node_features=torch.tensor([[0.5, 0.0, 0.0, 0.0]]),
        positions=torch.zeros((1, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
    )
    with torch.no_grad():
        encoder.layers[0].weight.zero_()
        encoder.layers[0].root.zero_()
        encoder.layers[0].root[0, 0] = 1.0
        encoder.layers[0].bias.zero_()
        encoder.fold_batch_norm()
        encoder.reset_activation_maxima()
        ann_output, activations = encoder.forward_ann(graph, return_activations=True)
        encoder.update_activation_maxima(activations)
        encoder.apply_parameter_normalization()
        normalized_spikes, _ = encoder.forward_snn(
            graph,
            simulation_steps=8,
            dynamics="standard_if",
        )
        decoder_units = normalized_spikes * encoder.output_activation_scale(normalized_spikes)

    torch.testing.assert_close(ann_output, torch.tensor([[0.5]]))
    torch.testing.assert_close(decoder_units, ann_output)


def test_empty_and_single_node_graphs_are_finite_and_differentiable() -> None:
    empty = build_event_graph(
        torch.empty((0, 4)),
        (8, 8),
        event_sampling_factor=1,
        graph_radius=0.08,
        graph_position_dims=3,
        graph_chunk_size=4,
    )
    assert empty.node_features.shape == (0, 4)
    assert empty.edge_index.shape == (2, 0)
    assert empty.edge_attr.shape == (0, 1)

    single = build_event_graph(
        torch.tensor([[3.0, 4.0, 0.0, 1.0]]),
        (8, 8),
        event_sampling_factor=1,
        graph_radius=0.08,
        graph_position_dims=3,
        graph_chunk_size=4,
    )
    assert single.node_features.shape == (1, 4)
    assert single.edge_index.shape == (2, 0)

    encoder = ASGCNEncoder(hidden_dim=2, graph_layers=1, spline_kernel_size=2)
    empty_output, _ = encoder.forward_ann(empty)
    single_output, _ = encoder.forward_ann(single)
    assert empty_output.shape == (0, 2)
    assert torch.isfinite(single_output).all()
    single_output.sum().backward()
    assert encoder.layers[0].root.grad is not None
    assert torch.isfinite(encoder.layers[0].root.grad).all()

    snn_encoder = _normalized_single_layer_encoder()
    empty_snn, firing_rates = snn_encoder.forward_snn(empty, simulation_steps=3)
    assert empty_snn.shape == (0, 1)
    torch.testing.assert_close(firing_rates[0], torch.tensor(0.0))


def test_legacy_edge_mlp_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="legacy edge-MLP"):
        ASGCNReconstructor(architecture_version=1)

    with pytest.raises(ValueError, match="distance_over_radius"):
        ASGCNReconstructor(spline_pseudo="distance")


def test_snn_restores_last_layer_lambda_before_analog_decoder() -> None:
    model = ASGCNReconstructor(
        hidden_dim=2,
        graph_layers=1,
        spline_kernel_size=2,
        decoder_channels=4,
        recurrent=False,
    ).eval()
    sample = {
        "events": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "target": torch.zeros((1, 4, 4)),
        "sensor_size": (4, 4),
        "sample_id": "known-scale",
        "metadata": {"dataset_sampling_ratio": 1.0},
    }
    captured: dict[str, torch.Tensor] = {}

    def capture_raster(features, graph, sensor_size, downsample):
        del graph, sensor_size, downsample
        captured["features"] = features.detach().clone()
        return torch.zeros((1, 2, 1, 1))

    with (
        patch.object(
            model.encoder,
            "forward_snn",
            return_value=(torch.ones((1, 2)), [torch.tensor(0.0)]),
        ),
        patch.object(
            model.encoder,
            "output_activation_scale",
            return_value=torch.tensor([2.0, 3.0]),
        ),
        patch("asgcn_recon.model.rasterize_features", side_effect=capture_raster),
        patch.object(
            model.decoder,
            "forward",
            return_value=(torch.zeros((1, 1, 4, 4)), None),
        ),
    ):
        _, diagnostics = model.forward_sample(sample, inference_mode="snn")

    torch.testing.assert_close(captured["features"], torch.tensor([[2.0, 3.0]]))
    assert diagnostics["decoder_input_lambda_applied"] is True
    assert diagnostics["isolated_nodes"].item() == 1
    assert diagnostics["max_degree"].item() == 0
