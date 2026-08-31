"""Exact IF recurrence regressions for invariant-current caching.

The reference intentionally retains the pre-optimization timestep implementation;
it is a test oracle, not an alternate experiment/inference implementation.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from asgcn_unet.graph import ASGCNEncoder, EventGraph


def _graph(nodes: int = 9, *, dtype: torch.dtype = torch.float32) -> EventGraph:
    generator = torch.Generator().manual_seed(241)
    features = torch.rand((nodes, 4), generator=generator, dtype=dtype) * 2 - 1
    pairs = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
    edges = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return EventGraph(
        features,
        features.clone(),
        edges,
        torch.linspace(0, 1, len(pairs), dtype=dtype).reshape(-1, 1),
    )


def _encoder(
    layers: int = 3,
    *,
    root: bool = True,
    dtype: torch.dtype = torch.float32,
) -> ASGCNEncoder:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(481)
        encoder = ASGCNEncoder(
            hidden_dim=5,
            graph_layers=layers,
            spline_kernel_size=3,
            spline_root_weight=root,
            spline_chunk_size=17,
        ).to(dtype=dtype).eval()
    with torch.no_grad():
        for index, layer in enumerate(encoder.layers):
            layer.bias.copy_(torch.linspace(-0.2, 0.8, 5, dtype=dtype))
            layer.norm.running_mean.fill_(0.03 * (index + 1))
            layer.norm.running_var.copy_(torch.linspace(0.6, 1.4, 5, dtype=dtype))
            layer.norm.weight.copy_(torch.linspace(0.7, 1.1, 5, dtype=dtype))
            layer.norm.bias.fill_(0.1)
            layer.calibration_activation_max.copy_(
                torch.linspace(0.8, 1.2, 5, dtype=dtype)
            )
        encoder.calibration_samples_seen.fill_(1)
        encoder.fold_batch_norm()
        encoder.apply_parameter_normalization()
        for layer in encoder.layers:
            layer.threshold.copy_(torch.linspace(0.5, 1.5, 5, dtype=dtype))
    return encoder


def _reference_snn(
    encoder: ASGCNEncoder,
    graph: EventGraph,
    steps: int,
    dynamics: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    node_count = int(graph.node_features.shape[0])
    if node_count == 0:
        return graph.node_features.new_empty((0, encoder.hidden_dim)), [
            graph.node_features.new_zeros(()) for _ in encoder.layers
        ]
    membranes = [
        layer.threshold.to(graph.node_features).expand(node_count, -1).clone() * 0.5
        for layer in encoder.layers
    ]
    previous_spikes = (
        [
            graph.node_features.new_zeros((node_count, layer.out_channels))
            for layer in encoder.layers
        ]
        if dynamics == "literal_eq15"
        else None
    )
    output_sum = graph.node_features.new_zeros(
        (node_count, encoder.layers[-1].out_channels)
    )
    active_counts = [graph.node_features.new_zeros(()) for _ in encoder.layers]
    basis_cache = encoder._basis_cache(graph)
    for _ in range(steps):
        hidden = graph.node_features
        for index, layer in enumerate(encoder.layers):
            current = layer.affine(
                hidden, graph.edge_index, graph.edge_attr, basis_cache, graph.in_degree
            )
            integrated = membranes[index] + current
            if previous_spikes is not None:
                integrated = integrated + previous_spikes[index]
            threshold = layer.threshold.to(integrated).expand_as(integrated)
            spikes = torch.where(
                integrated >= threshold, threshold, torch.zeros_like(integrated)
            )
            membranes[index] = integrated - spikes
            if previous_spikes is not None:
                previous_spikes[index] = spikes
            if index == len(encoder.layers) - 1:
                output_sum = output_sum + spikes
            active_counts[index] = active_counts[index] + (spikes != 0).sum()
            hidden = spikes
    rates = [
        active.to(graph.node_features.dtype)
        / float(steps * max(1, node_count * layer.out_channels))
        for active, layer in zip(active_counts, encoder.layers, strict=True)
    ]
    return output_sum / float(steps), rates


def _assert_exact(actual, expected) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _trace(function):
    original_where = torch.where
    spikes = []

    def traced_where(*args, **kwargs):
        result = original_where(*args, **kwargs)
        spikes.append(result.detach().clone())
        return result

    with patch("asgcn_unet.graph.torch.where", side_effect=traced_where):
        result = function()
    return result, spikes


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("steps", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("nodes", [0, 1, 9])
@pytest.mark.parametrize("root", [False, True])
def test_every_spike_and_rate_matches_original(dynamics, steps, nodes, root) -> None:
    encoder = _encoder(root=root)
    graph = _graph(nodes)
    with torch.no_grad():
        expected, expected_trace = _trace(
            lambda: _reference_snn(encoder, graph, steps, dynamics)
        )
        actual, actual_trace = _trace(
            lambda: encoder.forward_snn(graph, steps, dynamics)
        )
    _assert_exact(actual, expected)
    assert len(actual_trace) == len(expected_trace) == (3 * steps if nodes else 0)
    _assert_exact(actual_trace, expected_trace)


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("steps", [4, 8, 16, 32])
@pytest.mark.parametrize("layers", [1, 3])
def test_only_first_affine_is_cached(dynamics, steps, layers) -> None:
    encoder = _encoder(layers)
    with ExitStack() as stack:
        calls = [
            stack.enter_context(patch.object(layer, "affine", wraps=layer.affine))
            for layer in encoder.layers
        ]
        with torch.no_grad():
            encoder.forward_snn(_graph(), steps, dynamics)
    assert [call.call_count for call in calls] == [1] + [steps] * (layers - 1)


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("training", [False, True])
def test_folded_bn_state_is_unchanged(dynamics, training) -> None:
    encoder = _encoder().train(training)
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    with torch.no_grad():
        actual = encoder.forward_snn(_graph(), 8, dynamics)
        expected = _reference_snn(encoder, _graph(), 8, dynamics)
    _assert_exact(actual, expected)
    _assert_exact(encoder.state_dict(), before)


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_autocast_preserves_integration_and_output_dtype(dynamics, dtype) -> None:
    encoder = _encoder()
    graph = _graph(dtype=dtype)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = _reference_snn(encoder, graph, 8, dynamics)
        actual = encoder.forward_snn(graph, 8, dynamics)
    _assert_exact(actual, expected)


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
@pytest.mark.parametrize("layers", [1, 3])
def test_hard_threshold_autograd_is_preserved(dynamics, layers) -> None:
    reference = _encoder(layers, dtype=torch.float64)
    for layer in reference.layers:
        layer.threshold.requires_grad_(True)
    optimized = copy.deepcopy(reference)
    expected_graph = _graph(dtype=torch.float64)
    expected_graph.node_features.requires_grad_(True)
    actual_graph = copy.deepcopy(expected_graph)
    expected, _ = _reference_snn(reference, expected_graph, 8, dynamics)
    actual, _ = optimized.forward_snn(actual_graph, 8, dynamics)
    _assert_exact(actual, expected)
    loss_weights = torch.linspace(0.25, 1.25, actual.numel(), dtype=actual.dtype).reshape_as(
        actual
    )
    (expected * loss_weights).sum().backward()
    (actual * loss_weights).sum().backward()
    for old, new in zip(reference.layers, optimized.layers, strict=True):
        torch.testing.assert_close(new.threshold.grad, old.threshold.grad, rtol=1e-14, atol=1e-14)
    assert bool((optimized.layers[-1].threshold.grad > 0).any())
    # Hard spike decisions do not define input/weight gradients. Caching must
    # neither introduce a surrogate gradient nor detach a real threshold path.
    assert actual_graph.node_features.grad is expected_graph.node_features.grad is None
    assert all(parameter.grad is None for parameter in optimized.parameters())
    assert all(parameter.grad is None for parameter in reference.parameters())


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_exact_threshold_boundaries_and_signed_currents(dynamics) -> None:
    encoder = _encoder(1)
    encoder.layers[0].threshold.fill_(1.0)
    graph = _graph(1)
    # Select adjacent *integrated* values. Adjacent currents around 0.5 can
    # round back to exactly 1.0 when the half-threshold membrane is added.
    below = torch.nextafter(torch.tensor(1.0), torch.tensor(float("-inf"))) - 0.5
    above = torch.nextafter(torch.tensor(1.0), torch.tensor(float("inf"))) - 0.5
    currents = torch.tensor([[-0.5, 0.0, below, 0.5, above]])
    with patch.object(encoder.layers[0], "affine", return_value=currents):
        actual, trace = _trace(lambda: encoder.forward_snn(graph, 4, dynamics))
        expected, expected_trace = _trace(lambda: _reference_snn(encoder, graph, 4, dynamics))
    _assert_exact(actual, expected)
    _assert_exact(trace, expected_trace)
    # Equality fires; negative and zero currents remain below the initial
    # threshold at step one. Adjacent float values are compared without epsilon.
    _assert_exact(trace[0], torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]))


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_current_cache_does_not_outlive_a_forward(dynamics) -> None:
    encoder = _encoder(1)
    graph = _graph()
    with torch.no_grad():
        before = encoder.forward_snn(graph, 8, dynamics)
        graph.node_features.mul_(-2)
        encoder.layers[0].bias.add_(0.5)
        encoder.layers[0].threshold.mul_(1.25)
        after = encoder.forward_snn(graph, 8, dynamics)
        expected = _reference_snn(encoder, graph, 8, dynamics)
    _assert_exact(after, expected)
    assert not torch.equal(before[0], after[0])


@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_zero_branch_does_not_allocate_node_channel_tensors(dynamics) -> None:
    encoder = _encoder(1)
    with (
        patch.object(encoder.layers[0], "affine", return_value=torch.full((9, 5), 0.2)),
        patch("asgcn_unet.graph.torch.zeros_like", wraps=torch.zeros_like) as zero_factory,
    ):
        encoder.forward_snn(_graph(), 32, dynamics)
    assert zero_factory.call_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_cuda_autocast_matches_original(dynamics) -> None:
    encoder = _encoder().cuda()
    graph = _graph()
    graph = EventGraph(
        graph.node_features.cuda(),
        graph.positions.cuda(),
        graph.edge_index.cuda(),
        graph.edge_attr.cuda(),
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual = encoder.forward_snn(graph, 8, dynamics)
        expected = _reference_snn(encoder, graph, 8, dynamics)
    _assert_exact(actual, expected)
