"""CPU-only synthetic unit fixtures, never dataset/quality evaluation results."""

from copy import deepcopy

import pytest
import torch

from asgcn_unet.ablation_encoders import (
    IdentityEventEncoder,
    PointwiseEventEncoder,
    PointwiseEventLayer,
    prepare_event_container,
    prepare_event_container_batch,
)
from asgcn_unet.graph import ASGCNEncoder, EventGraph, prepare_event_nodes


@pytest.fixture(autouse=True)
def cpu_unit_test_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def events(count, offset=0):
    index = torch.arange(count, dtype=torch.float32)
    return torch.stack((index % 7, index % 5, index * 7 + offset, index % 2), dim=1)


def container(count=24):
    return prepare_event_container(events(count), (5, 7), event_sampling_factor=1)


@pytest.mark.parametrize("factor", [1, 2, 3, 20])
@pytest.mark.parametrize("counts", [(0,), (1,), (7,), (0, 3, 0, 8, 1), (9, 11, 4)])
def test_normalization_and_sampling_match_original_without_radius(monkeypatch, factor, counts):
    def forbidden(*args, **kwargs):
        raise AssertionError("No-graph architecture must never construct a radius graph")

    monkeypatch.setattr("asgcn_unet.graph.build_radius_graph", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.build_event_graph", forbidden)
    monkeypatch.setattr("asgcn_unet.graph.build_event_graph_batch", forbidden)
    values = [events(count, offset=100 * index) for index, count in enumerate(counts)]
    packed = prepare_event_container_batch(
        torch.cat(values), counts, (5, 7), event_sampling_factor=factor
    )
    expected = [prepare_event_nodes(value[::factor], (5, 7)) for value in values]
    assert torch.equal(packed.graph.node_features, torch.cat([item[0] for item in expected]))
    assert torch.equal(packed.graph.positions, torch.cat([item[1] for item in expected]))
    assert packed.node_counts == tuple(len(item[0]) for item in expected)
    assert packed.node_batch.tolist() == [
        index for index, item in enumerate(expected) for _ in range(len(item[0]))
    ]
    assert packed.graph.edge_index.shape == (2, 0)
    assert packed.graph.edge_attr.shape == (0, 1)
    assert not packed.graph.in_degree.any()
    assert packed.edge_counts.tolist() == [0] * len(counts)
    for value, (features, positions) in zip(values, expected, strict=True):
        single = prepare_event_container(value, (5, 7), event_sampling_factor=factor)
        assert torch.equal(single.node_features, features)
        assert torch.equal(single.positions, positions)


@pytest.mark.parametrize("factor", [False, 0, -1, 1.5])
def test_invalid_sampling_fails(factor):
    with pytest.raises(ValueError):
        prepare_event_container(events(3), (5, 7), event_sampling_factor=factor)
    with pytest.raises(ValueError):
        prepare_event_container_batch(events(3), (3,), (5, 7), event_sampling_factor=factor)


@pytest.mark.parametrize("counts", [(2,), (-1, 4), (True, 2), (1.0, 2)])
def test_invalid_counts_fail(counts):
    with pytest.raises(ValueError):
        prepare_event_container_batch(events(3), counts, (5, 7), event_sampling_factor=1)


@pytest.mark.parametrize("damage", ["nan", "unordered", "shape", "sensor"])
def test_invalid_events_fail(damage):
    value, size = events(3), (5, 7)
    if damage == "nan":
        value[1, 0] = float("nan")
    elif damage == "unordered":
        value[1, 2] = -100
    elif damage == "shape":
        value = value[:, :3]
    else:
        size = (0, 7)
    with pytest.raises(ValueError):
        prepare_event_container(value, size, event_sampling_factor=1)
    with pytest.raises(ValueError):
        prepare_event_container_batch(value, (3,), size, event_sampling_factor=1)


def test_identity_is_exact_raw_feature_path_without_parameters():
    graph = container()
    encoder = IdentityEventEncoder()
    output, activations = encoder.forward_ann(graph, return_activations=True)
    assert output is graph.node_features
    assert activations == []
    assert list(encoder.parameters()) == []
    assert len(encoder.layers) == 0
    assert encoder.hidden_dim == 4
    assert encoder.supports_snn is False
    restored = IdentityEventEncoder()
    restored.load_state_dict(encoder.state_dict(), strict=True)
    assert torch.equal(restored.forward_ann(graph)[0], output)


@pytest.mark.parametrize(
    "method,args",
    [
        ("forward_snn", (None,)),
        ("fold_batch_norm", ()),
        ("reset_activation_maxima", ()),
        ("update_activation_maxima", ([],)),
        ("apply_parameter_normalization", ()),
        ("output_activation_scale", (torch.ones(4),)),
        ("calibration_summary", ()),
    ],
)
def test_identity_conversion_is_explicitly_unsupported(method, args):
    with pytest.raises(ValueError):
        getattr(IdentityEventEncoder(), method)(*args)


@pytest.mark.parametrize("kind", [IdentityEventEncoder, lambda: PointwiseEventEncoder(8, 3)])
def test_no_graph_encoders_reject_neighbor_topology(kind):
    graph = container()
    with_edges = EventGraph(
        graph.node_features, graph.positions, torch.tensor([[0], [1]]), torch.zeros(1, 1)
    )
    with pytest.raises(ValueError, match="edgeless"):
        kind().forward_ann(with_edges)


def test_pointwise_each_parameter_receives_gradient_and_optimizer_update():
    encoder = PointwiseEventEncoder(64, 6).eval()
    # Positive deterministic synthetic weights avoid randomly dead ReLU channels
    # so every affine and BN parameter must participate in this gradient test.
    with torch.no_grad():
        for layer in encoder.layers:
            layer.weight.fill_(0.02)
            layer.bias.fill_(0.1)
            layer.norm.weight.fill_(1.1)
            layer.norm.bias.fill_(0.2)
    assert len(encoder.layers) == 6
    assert encoder.hidden_dim == 64
    for layer in encoder.layers:
        assert layer.weight.ndim == 2
        assert not hasattr(layer, "root")
        assert not hasattr(layer, "kernel_size")
    original = {name: parameter.detach().clone() for name, parameter in encoder.named_parameters()}
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    output, activations = encoder.forward_ann(container(), True)
    assert len(activations) == 6
    assert output.shape == (24, 64)
    output.square().mean().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in encoder.named_parameters():
        assert not torch.equal(parameter, original[name]), name


def matched_root_only_reference():
    """Synthetic mathematical oracle: existing ASGCN root transform, no edges."""
    torch.manual_seed(712)
    pointwise = PointwiseEventEncoder(8, 3).eval()
    reference = ASGCNEncoder(8, 3, spline_root_weight=True).eval()
    with torch.no_grad():
        for actual, oracle in zip(pointwise.layers, reference.layers, strict=True):
            actual.bias.fill_(0.2)
            oracle.weight.zero_()
            oracle.root.copy_(actual.weight)
            oracle.bias.copy_(actual.bias)
            actual.norm.running_mean.copy_(torch.linspace(-0.1, 0.1, 8))
            actual.norm.running_var.copy_(torch.linspace(0.8, 1.2, 8))
            oracle.norm.load_state_dict(actual.norm.state_dict())
    return pointwise, reference


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
@pytest.mark.parametrize("steps", [1, 4, 8, 16, 32])
def test_same_ann_bn_conversion_and_if_as_original_root_transform(dynamics, steps):
    pointwise, reference = matched_root_only_reference()
    graph = container()
    torch.testing.assert_close(pointwise.forward_ann(graph)[0], reference.forward_ann(graph)[0])
    for encoder in (pointwise, reference):
        before = encoder.forward_ann(graph)[0]
        encoder.fold_batch_norm()
        torch.testing.assert_close(encoder.forward_ann(graph)[0], before, atol=2e-6, rtol=1e-5)
        encoder.reset_activation_maxima()
        encoder.update_activation_maxima(encoder.forward_ann(graph, True)[1], sample_count=3)
        encoder.apply_parameter_normalization()
    assert pointwise.calibration_summary() == reference.calibration_summary()
    for actual, oracle in zip(pointwise.layers, reference.layers, strict=True):
        torch.testing.assert_close(actual.weight, oracle.root)
        torch.testing.assert_close(actual.bias, oracle.bias)
        torch.testing.assert_close(actual.normalization_scale, oracle.normalization_scale)
        assert actual._bn_is_folded and actual._snn_is_normalized
    actual, actual_rates = pointwise.forward_snn(graph, steps, dynamics)
    expected, expected_rates = reference.forward_snn(graph, steps, dynamics)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_rates, expected_rates, strict=True))
    assert torch.equal(
        pointwise.output_activation_scale(actual), reference.output_activation_scale(expected)
    )


@pytest.mark.parametrize("dynamics", ["standard_if", "literal_eq15"])
def test_packed_ann_snn_matches_independent_evaluation(dynamics):
    encoder, _ = matched_root_only_reference()
    samples = [events(11), events(0), events(5, 100)]
    packed = prepare_event_container_batch(
        torch.cat(samples), tuple(map(len, samples)), (5, 7), event_sampling_factor=2
    )
    singles = [prepare_event_container(value, (5, 7), event_sampling_factor=2) for value in samples]
    actual = encoder.forward_ann(packed.graph)[0]
    expected = torch.cat([encoder.forward_ann(graph)[0] for graph in singles])
    torch.testing.assert_close(actual, expected)
    encoder.fold_batch_norm()
    encoder.reset_activation_maxima()
    encoder.update_activation_maxima(encoder.forward_ann(packed.graph, True)[1], sample_count=2)
    encoder.apply_parameter_normalization()
    actual, rates = encoder.forward_snn(
        packed.graph, 8, dynamics, node_batch=packed.node_batch, batch_size=3
    )
    single = [encoder.forward_snn(graph, 8, dynamics) for graph in singles]
    assert torch.equal(actual, torch.cat([item[0] for item in single]))
    for layer_index, rate in enumerate(rates):
        assert torch.equal(rate, torch.stack([item[1][layer_index] for item in single]))


def test_pooled_batch_norm_updates_once_per_layer():
    encoder = PointwiseEventEncoder(8, 3).train()
    packed = prepare_event_container_batch(
        torch.cat((events(7), events(9, 100))), (7, 9), (5, 7), event_sampling_factor=1
    )
    encoder.forward_ann(packed.graph)
    assert [layer.norm.num_batches_tracked.item() for layer in encoder.layers] == [1, 1, 1]


def test_converted_checkpoint_roundtrip_preserves_flags_and_predictions():
    encoder, _ = matched_root_only_reference()
    graph = container()
    encoder.fold_batch_norm()
    encoder.reset_activation_maxima()
    encoder.update_activation_maxima(encoder.forward_ann(graph, True)[1])
    encoder.apply_parameter_normalization()
    restored = PointwiseEventEncoder(8, 3).eval()
    restored.load_state_dict(deepcopy(encoder.state_dict()), strict=True)
    assert restored.calibration_summary() == encoder.calibration_summary()
    assert all(layer._bn_is_folded and layer._snn_is_normalized for layer in restored.layers)
    assert torch.equal(restored.forward_snn(graph, 8)[0], encoder.forward_snn(graph, 8)[0])


def test_conversion_rejects_unprepared_and_double_conversion():
    encoder = PointwiseEventEncoder(8, 3)
    with pytest.raises(RuntimeError, match="parameter normalization"):
        encoder.forward_snn(container())
    with pytest.raises(RuntimeError, match="no non-empty calibration"):
        encoder.apply_parameter_normalization()
    layer = encoder.layers[0]
    with pytest.raises(RuntimeError, match="Call eval"):
        layer.fold_batch_norm()
    encoder.fold_batch_norm()
    encoder.reset_activation_maxima()
    encoder.update_activation_maxima(encoder.forward_ann(container(), True)[1])
    encoder.apply_parameter_normalization()
    with pytest.raises(RuntimeError, match="already applied"):
        encoder.apply_parameter_normalization()


@pytest.mark.parametrize("hidden,layers", [(0, 6), (64, 0), (True, 6), (64, False), (2.5, 6)])
def test_architecture_sizes_must_be_explicit_positive_integers(hidden, layers):
    with pytest.raises(ValueError):
        PointwiseEventEncoder(hidden, layers)


def test_dead_channels_retain_unit_scale_without_fake_calibration():
    layer = PointwiseEventLayer(4, 3).eval()
    layer.fold_batch_norm()
    layer.calibration_activation_max.copy_(torch.tensor([0.0, 2.0, 4.0]))
    layer.apply_parameter_normalization(torch.ones(4), torch.tensor([1.0, 2.0, 4.0]))
    assert layer.dead_channel_mask.tolist() == [True, False, False]
    assert layer.normalization_scale.tolist() == [1.0, 2.0, 4.0]


def test_empty_event_window_has_valid_empty_encoder_output():
    encoder = PointwiseEventEncoder(8, 3)
    empty = container(0)
    output, activations = encoder.forward_ann(empty, True)
    assert output.shape == (0, 8)
    assert all(value.shape == (0, 8) for value in activations)
    encoder.reset_activation_maxima()
    encoder.update_activation_maxima(activations, sample_count=0)
    assert encoder.calibration_samples_seen.tolist() == [0, 0, 0]
    with pytest.raises(RuntimeError, match="no non-empty calibration"):
        encoder.apply_parameter_normalization()
