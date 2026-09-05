"""CPU-only unit tests; synthetic inputs are not research evaluation results."""

from copy import deepcopy

import pytest
import torch

from asgcn_unet.graph import (
    ASGCNEncoder,
    build_event_graph,
    build_event_graph_batch,
    build_radius_graph,
)

SENSOR = (9, 13)
KWARGS = {
    "event_sampling_factor": 1,
    "graph_radius": 0.8,
    "graph_position_dims": 3,
    "graph_chunk_size": 3,
    "max_graph_edges": 500,
}


def samples():
    return [
        torch.empty((0, 4)),
        torch.tensor([
            [1, 1, 10, 1], [3, 2, 11, -1], [3, 2, 11, 0],
            [7, 5, 14, 1], [10, 8, 15, -1],
        ]),
        torch.empty((0, 4)),
        torch.tensor([[1, 1, 0, 1], [1, 1, 0, -1], [2, 1, 0, 1]]),
        torch.tensor([[0, 0, -7, -1]]),
        torch.empty((0, 4)),
    ]


def packed(values, **overrides):
    return build_event_graph_batch(
        torch.cat(values), tuple(len(value) for value in values), SENSOR,
        **(KWARGS | overrides),
    )


@pytest.mark.parametrize("factor", [1, 2, 3])
@pytest.mark.parametrize("dims", [1, 2, 3, 4])
@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_packed_graph_matches_independent_graphs(factor, dims, chunk):
    values = samples()
    options = KWARGS | {
        "event_sampling_factor": factor,
        "graph_position_dims": dims,
        "graph_chunk_size": chunk,
    }
    result = packed(values, **options)
    references = [build_event_graph(value, SENSOR, **options) for value in values]
    counts = tuple(len(graph.node_features) for graph in references)
    assert result.node_counts == counts
    assert result.edge_counts.tolist() == [graph.edge_index.shape[1] for graph in references]
    expected_indices = []
    offset = 0
    for graph in references:
        expected_indices.append(graph.edge_index + offset)
        offset += len(graph.node_features)
    for actual, expected in (
        (result.graph.node_features, torch.cat([g.node_features for g in references])),
        (result.graph.positions, torch.cat([g.positions for g in references])),
        (result.graph.edge_index, torch.cat(expected_indices, dim=1)),
        (result.graph.edge_attr, torch.cat([g.edge_attr for g in references])),
        (result.graph.in_degree, torch.cat([g.in_degree for g in references])),
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    source, destination = result.graph.edge_index
    assert torch.equal(result.node_batch[source], result.node_batch[destination])


def test_guard_applies_per_graph_not_to_aggregate_batch():
    dense = torch.zeros((5, 4))
    result = packed([dense, dense], max_graph_edges=20)
    assert result.edge_counts.tolist() == [20, 20]
    assert result.graph.edge_index.shape == (2, 40)
    with pytest.raises(RuntimeError, match="max_graph_edges=19"):
        packed([dense, dense], max_graph_edges=19)


def test_guard_catches_only_over_limit_sample_with_uneven_graphs():
    with pytest.raises(RuntimeError, match=r"batch sample\(s\) \[1\]"):
        packed([torch.zeros((2, 4)), torch.zeros((5, 4))], max_graph_edges=19)


def test_packed_radius_boundary_and_sample_identity():
    positions = torch.tensor([
        [0.0, 0.0], [0.5, 0.0], [0.25, 0.0],
        [0.0, 0.0], [0.5, 0.0], [0.25, 0.0],
    ])
    edge_index, edge_attr = build_radius_graph(
        positions, 0.5, position_dims=2, chunk_size=1,
        node_batch=torch.tensor([0, 0, 0, 1, 1, 1]), node_counts=(3, 3),
    )
    expected = torch.tensor([[0, 1, 2, 2], [2, 2, 0, 1]])
    torch.testing.assert_close(edge_index, torch.cat((expected, expected + 3), dim=1))
    torch.testing.assert_close(edge_attr, torch.full((8, 1), 0.5))


def test_empty_and_singleton_graphs_keep_batch_slots():
    result = packed([torch.empty((0, 4)) for _ in range(3)])
    assert result.node_counts == (0, 0, 0)
    assert result.graph.edge_index.shape == (2, 0)
    assert result.node_batch.shape == (0,)
    assert result.edge_counts.tolist() == [0, 0, 0]
    result = packed([torch.zeros((1, 4)) for _ in range(3)])
    assert result.graph.edge_index.shape == (2, 0)
    assert result.edge_counts.tolist() == [0, 0, 0]


def test_sampling_restarts_per_sample_and_ignores_unsampled_invalid_values():
    values = [torch.tensor([[0, 0, 1, 1], [float("nan"), 0, 2, 1], [1, 1, 3, 1]])]
    values.append(torch.tensor([[3, 4, -10, -1], [float("nan"), 0, 8, 1]]))
    result = packed(values, event_sampling_factor=2)
    assert result.node_counts == (2, 1)
    expected = torch.cat([
        build_event_graph(value, SENSOR, **(KWARGS | {"event_sampling_factor": 2})).positions
        for value in values
    ])
    torch.testing.assert_close(result.graph.positions, expected, atol=0, rtol=0)


@pytest.mark.parametrize("counts", [(1,), (-1, 3), (True, 1), (1.0, 1)])
def test_invalid_packed_counts_rejected(counts):
    with pytest.raises(ValueError, match="counts"):
        build_event_graph_batch(torch.zeros((2, 4)), counts, SENSOR, **KWARGS)


def test_timestamp_checks_apply_inside_each_sample():
    with pytest.raises(ValueError, match="monotonically"):
        packed([torch.tensor([[0, 0, 2, 1], [0, 0, 1, 1]])])
    result = packed([
        torch.tensor([[0, 0, 2, 1], [0, 0, 3, 1]]),
        torch.tensor([[0, 0, -2, 1], [0, 0, -1, 1]]),
    ])
    torch.testing.assert_close(result.graph.positions[:, 2], torch.tensor([0., 1., 0., 1.]))


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("steps", [1, 4])
def test_packed_snn_outputs_and_firing_rates_match_independent_samples(dynamics, steps):
    torch.manual_seed(45)
    values = samples()
    result = packed(values)
    encoder = ASGCNEncoder(hidden_dim=5, graph_layers=2).eval()
    encoder.reset_activation_maxima()
    _, activations = encoder.forward_ann(result.graph, return_activations=True)
    encoder.update_activation_maxima(activations, sample_count=3)
    encoder.fold_batch_norm()
    encoder.apply_parameter_normalization()
    hidden, rates = encoder.forward_snn(
        result.graph, steps, dynamics, node_batch=result.node_batch, batch_size=len(values)
    )
    references = [
        encoder.forward_snn(build_event_graph(value, SENSOR, **KWARGS), steps, dynamics)
        for value in values
    ]
    torch.testing.assert_close(hidden, torch.cat([value[0] for value in references]))
    for layer_index, per_sample in enumerate(rates):
        torch.testing.assert_close(
            per_sample, torch.stack([value[1][layer_index] for value in references])
        )


def test_packed_calibration_preserves_maxima_and_sample_count():
    torch.manual_seed(13)
    reference = ASGCNEncoder(hidden_dim=5, graph_layers=2).eval()
    reference.reset_activation_maxima()
    batched = deepcopy(reference)
    values = samples()
    for value in values:
        graph = build_event_graph(value, SENSOR, **KWARGS)
        _, activations = reference.forward_ann(graph, return_activations=True)
        reference.update_activation_maxima(activations)
    result = packed(values)
    _, activations = batched.forward_ann(result.graph, return_activations=True)
    batched.update_activation_maxima(activations, sample_count=3)
    torch.testing.assert_close(reference.calibration_samples_seen, batched.calibration_samples_seen)
    for single_layer, batch_layer in zip(reference.layers, batched.layers, strict=True):
        torch.testing.assert_close(
            single_layer.calibration_activation_max, batch_layer.calibration_activation_max
        )


def test_empty_packed_snn_has_per_sample_zero_statistics():
    result = packed([torch.empty((0, 4)), torch.empty((0, 4))])
    encoder = ASGCNEncoder(hidden_dim=5, graph_layers=2).eval()
    # Unit-test setup marks conversion only to exercise the empty-input contract.
    for layer in encoder.layers:
        layer._snn_is_normalized = True
    hidden, rates = encoder.forward_snn(
        result.graph, node_batch=result.node_batch, batch_size=2
    )
    assert hidden.shape == (0, 5)
    for rate in rates:
        torch.testing.assert_close(rate, torch.zeros(2))


@pytest.mark.parametrize("count", [-1, True, 0.5])
def test_calibration_rejects_invalid_sample_count(count):
    encoder = ASGCNEncoder(hidden_dim=5, graph_layers=2)
    with pytest.raises(ValueError, match="sample_count"):
        encoder.update_activation_maxima([torch.ones(2, 5)] * 2, sample_count=count)
