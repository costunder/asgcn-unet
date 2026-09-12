"""Small synthetic CPU pooling tests; no production training/performance claims."""

import pytest
import torch

from asgcn_unet.graph import EventGraph, PaperSplineConv
from asgcn_unet.graph_pool import PoolState, pool_graph, update_pool
from asgcn_unet.implicit_radius import ImplicitRadiusIndex
from asgcn_unet.stream_graph import GraphUpdate, StreamGraph, evolve_stream_graph

CONFIG = {"spatial_cell_pixels": 4., "temporal_cell_seconds": .004, "edge_chunk_size": 3}
SENSOR = (33, 33)
SCALE = .05


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _update(previous, rows, *, storage="implicit_radius", cutoffs=(-1., -1.), radius=.25):
    values = torch.tensor(rows, dtype=torch.float64).reshape(-1, 5)
    positions = torch.stack((values[:, 0] / 32, values[:, 1] / 32, values[:, 2] / SCALE,
                             (values[:, 3] + 1) / 2), dim=1)
    return evolve_stream_graph(previous, positions.float(), positions, values[:, 2], values[:, 4].long(),
                               torch.tensor(cutoffs, dtype=torch.float64), radius=radius, max_graph_edges=None,
                               chunk_size=2, graph_storage=storage)


def _features(count, *, seed=12, dtype=torch.float64):
    return torch.randn((count, 64), dtype=dtype, generator=torch.Generator().manual_seed(seed))


def _raw_edges(raw):
    if isinstance(raw.graph, EventGraph):
        return raw.graph.edge_index, raw.graph.edge_attr
    chunks = list(raw.graph.iter_directed_neighbors())
    if not chunks:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float64)
    source, destination, pseudo = (torch.cat(parts) for parts in zip(*chunks))
    return torch.stack((source, destination)), pseudo


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
def test_pulse_only_tick_reuses_existing_quotient_without_sort_or_hash(monkeypatch, storage):
    from asgcn_unet import quotient_accumulator
    raw = _update(None, [[1, 1, .001, 1, 0], [2, 1, .002, -1, 0], [5, 1, .001, 1, 0]], storage=storage).state
    pool = pool_graph(raw, _features(3), CONFIG, SENSOR, SCALE)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unchanged quotient must not be rebuilt")
    monkeypatch.setattr(quotient_accumulator, "QuotientAccumulator", forbidden)
    features = pool.raw_features.clone()
    features[0] += .5
    update = GraphUpdate(raw, torch.arange(3), torch.zeros(3, dtype=torch.bool))
    changed, _ = update_pool(update, features, pool, CONFIG, SENSOR, SCALE)
    _assert_oracle(changed)
    assert changed.graph.graph.edge_index is pool.graph.graph.edge_index
    assert changed.graph.graph.edge_attr is pool.graph.graph.edge_attr
    assert changed.work["reused_quotient_topology"]
    assert changed.work["raw_edges_visited"] == 0
    assert not changed.work["topology_changed_nodes"].any()


def _assert_oracle(pool):
    raw, graph = pool.raw_graph, pool.graph.graph
    edges, pseudo = _raw_edges(raw)
    keys = torch.cat((raw.node_batch[:, None], torch.floor(raw.graph.positions[:, :3]
                      * raw.graph.positions.new_tensor((8., 8., SCALE / .004))).long()), dim=1)
    expected_keys, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    torch.testing.assert_close(pool.cluster_keys, expected_keys, atol=0, rtol=0)
    torch.testing.assert_close(pool.raw_to_cluster, inverse, atol=0, rtol=0)
    torch.testing.assert_close(pool.counts, counts, atol=0, rtol=0)
    for cluster in range(len(counts)):  # Small CPU oracle only, not implementation.
        selected = inverse == cluster
        torch.testing.assert_close(graph.node_features[cluster], pool.raw_features[selected].double().mean(0).to(pool.raw_features.dtype))
        torch.testing.assert_close(graph.positions[cluster], raw.graph.positions[selected].mean(0))
        torch.testing.assert_close(pool.graph.timestamps[cluster], raw.timestamps[selected].mean())
    contributors = {}
    for edge, value in zip(edges.t().tolist(), pseudo[:, 0].tolist()):
        pair = (int(inverse[edge[0]]), int(inverse[edge[1]]))
        if pair[0] != pair[1]:
            contributors.setdefault(pair, []).append(value)
    assert list(map(tuple, graph.edge_index.t().tolist())) == sorted(contributors)
    for number, pair in enumerate(sorted(contributors)):
        values = contributors[pair]
        assert pool.edge_refcounts[number] == len(values)
        torch.testing.assert_close(pool.edge_pseudo_sums[number, 0], torch.tensor(sum(values), dtype=torch.float64))
        torch.testing.assert_close(graph.edge_attr[number, 0], torch.tensor(sum(values) / len(values), dtype=graph.edge_attr.dtype))
    torch.testing.assert_close(graph.in_degree, torch.bincount(graph.edge_index[1], minlength=len(counts)))
    assert bool((pool.graph.node_batch[graph.edge_index[0]] == pool.graph.node_batch[graph.edge_index[1]]).all())


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_snapshot_means_quotient_counts_and_pseudo_match_existing_raw_edges(storage, dtype):
    raw = _update(None, [[1, 1, .001, 1, 0], [2, 1, .002, -1, 0],
                         [5, 1, .001, 1, 0], [6, 2, .002, 1, 0],
                         [1, 1, .001, 1, 1], [5, 1, .001, 1, 1]], storage=storage).state
    pool = pool_graph(raw, _features(6, dtype=dtype), CONFIG, SENSOR, SCALE)
    assert isinstance(pool, PoolState)
    assert pool.graph.graph.node_features.shape == (4, 64)
    _assert_oracle(pool)
    expected_cost = raw.graph.edge_index.shape[1] if storage == "materialized" else 0
    assert pool.work["materialized_edges_scanned"] == expected_cost
    if storage == "implicit_radius":
        assert not pool.work["whole_raw_edge_reenumeration"]
    centroid_distance = torch.linalg.vector_norm(pool.graph.graph.positions[0, :3] - pool.graph.graph.positions[1, :3]) / .25
    assert not torch.isclose(pool.graph.graph.edge_attr[0, 0], centroid_distance, rtol=1e-6, atol=1e-8)


def test_arbitrary_materialized_multiedges_self_loops_and_missing_pairs_are_not_radius_rebuilt():
    positions = torch.tensor([[0., 0, 0, 0], [.03, 0, 0, 0], [.13, 0, 0, 0], [.3, 0, 0, 0]], dtype=torch.float64)
    edges = torch.tensor([[0, 0, 0, 1, 2, 3], [0, 1, 2, 2, 0, 3]])
    pseudo = torch.tensor([[0.], [.2], [.3], [.7], [.8], [0.]], dtype=torch.float64)
    raw = StreamGraph(EventGraph(torch.zeros((4, 4)), positions, edges, pseudo),
                      torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.float64))
    pool = pool_graph(raw, _features(4), CONFIG, SENSOR, SCALE)
    assert pool.graph.graph.edge_index.tolist() == [[0, 1], [1, 0]]
    assert pool.edge_refcounts.tolist() == [2, 1]
    torch.testing.assert_close(pool.graph.graph.edge_attr[:, 0], torch.tensor([.5, .8], dtype=torch.float64))
    assert pool.graph.graph.in_degree.tolist() == [1, 1, 0]
    _assert_oracle(pool)


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
def test_incremental_append_expire_replacement_and_empty_clusters_match_fresh_snapshot(storage):
    steps = [
        ([[1, 1, .001, 1, 0], [2, 1, .002, -1, 0], [5, 1, .002, 1, 0],
          [6, 1, .003, -1, 0], [1, 1, .001, 1, 1]], (-1., -1.)),
        ([[3, 1, .003, 1, 0], [5, 2, .006, -1, 0], [5, 1, .005, 1, 1]], (.002, .001)),
        ([], (.002, .002)),
        ([[1, 1, .010, -1, 0], [2, 1, .010, 1, 0]], (.005, .006)),
        ([], (.02, .02)),
        ([[1, 1, .03, -1, 1], [5, 1, .03, 1, 1]], (.02, .02)),
    ]
    raw, previous = None, None
    for number, (rows, cutoffs) in enumerate(steps):
        update = _update(raw, rows, storage=storage, cutoffs=cutoffs)
        features = _features(len(update.state.timestamps), seed=number)
        state, pooled_update = update_pool(update, features, previous, CONFIG, SENSOR, SCALE)
        fresh = pool_graph(update.state, features, CONFIG, SENSOR, SCALE)
        _assert_oracle(state)
        for field in ("cluster_keys", "counts", "raw_to_cluster", "feature_sums", "position_sums", "timestamp_sums", "edge_refcounts", "edge_pseudo_sums"):
            torch.testing.assert_close(getattr(state, field), getattr(fresh, field), rtol=1e-11, atol=1e-12, msg=field)
        torch.testing.assert_close(state.graph.graph.edge_index, fresh.graph.graph.edge_index)
        torch.testing.assert_close(state.graph.graph.edge_attr, fresh.graph.graph.edge_attr)
        if previous is not None:
            expected_old = []
            old_keys = list(map(tuple, previous.cluster_keys.tolist()))
            for key in map(tuple, state.cluster_keys.tolist()):
                expected_old.append(old_keys.index(key) if key in old_keys else -1)
            assert pooled_update.old_indices.tolist() == expected_old
        assert pooled_update.changed_nodes.shape == state.counts.shape
        raw, previous = update.state, state


def test_feature_only_update_uses_deltas_and_never_requeries_raw_topology(monkeypatch):
    first = _update(None, [[1, 1, .001, 1, 0], [2, 1, .001, 1, 0], [5, 1, .001, 1, 0], [24, 1, .001, 1, 0]])
    features = _features(4)
    initial, _ = update_pool(first, features, None, CONFIG, SENSOR, SCALE)
    update = _update(first.state, [], cutoffs=(-1., -1.))
    changed_features = features.clone()
    changed_features[0] += .125
    original_features = initial.feature_sums.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("Feature-only pooling must reuse topology without any raw edge query")

    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", forbidden)
    state, change = update_pool(update, changed_features, initial, CONFIG, SENSOR, SCALE)
    assert state.work["raw_edges_visited"] == state.work["raw_query_nodes"] == 0
    assert state.work["feature_rows_updated"] == 1
    assert change.changed_nodes.tolist() == [True, True, False]
    assert state.work["topology_changed_nodes"].tolist() == [False, False, False]
    assert state.work["input_changed_sources"].tolist() == [True, False, False]
    torch.testing.assert_close(initial.feature_sums, original_features, rtol=0, atol=0)
    torch.testing.assert_close(state.graph.graph.node_features[0], changed_features[:2].mean(0))


def test_implicit_topology_queries_only_added_removed_incident_nodes(monkeypatch):
    first = _update(None, [[1, 1, .001, 1, 0], [2, 1, .002, 1, 0], [5, 1, .002, 1, 0], [6, 1, .002, 1, 0]])
    initial = pool_graph(first.state, _features(4), CONFIG, SENSOR, SCALE)
    update = _update(first.state, [[3, 1, .003, 1, 0]], cutoffs=(.002, -1.))
    original = ImplicitRadiusIndex.iter_directed_neighbors
    sizes = []

    def selected_only(index, destinations=None, **kwargs):
        assert destinations is not None, "Incremental pooling must not re-enumerate unchanged raw edges"
        sizes.append(destinations.numel())
        return original(index, destinations, **kwargs)

    monkeypatch.setattr(ImplicitRadiusIndex, "iter_directed_neighbors", selected_only)
    state, _ = update_pool(update, _features(4), initial, CONFIG, SENSOR, SCALE)
    assert sizes == [1, 1]
    assert state.work["raw_query_nodes"] == 2
    assert state.work["materialized_edges_scanned"] == 0


def test_repeated_equal_pulses_still_seed_root_and_outgoing_cluster_neighbors():
    first = _update(None, [[1, 1, .001, 1, 0], [5, 1, .001, 1, 0], [24, 1, .001, 1, 0]])
    features = torch.ones((3, 64))
    initial = pool_graph(first.state, features, CONFIG, SENSOR, SCALE)
    identity = GraphUpdate(first.state, torch.arange(3), torch.zeros(3, dtype=torch.bool))
    unchanged, quiet = update_pool(identity, features, initial, CONFIG, SENSOR, SCALE)
    assert not bool(quiet.changed_nodes.any())
    assert not bool(unchanged.work["topology_changed_nodes"].any())
    assert not bool(unchanged.work["input_changed_sources"].any())
    for active in (torch.tensor([0]), torch.tensor([True, False, False])):
        state, pulse = update_pool(identity, features, unchanged, CONFIG, SENSOR, SCALE, active_sources=active)
        assert pulse.changed_nodes.tolist() == [True, True, False]
        assert state.work["topology_changed_nodes"].tolist() == [False, False, False]
        assert state.work["input_changed_sources"].tolist() == [True, False, False]
        torch.testing.assert_close(state.graph.graph.node_features, initial.graph.graph.node_features, rtol=0, atol=0)
        assert state.work["raw_edges_visited"] == 0


def test_removed_contributor_changes_edge_mean_even_when_quotient_degree_is_unchanged():
    first = _update(None, [[1, 1, .001, 1, 0], [2, 1, .002, 1, 0], [5, 1, .002, 1, 0]])
    initial = pool_graph(first.state, torch.ones((3, 64)), CONFIG, SENSOR, SCALE)
    update = _update(first.state, [], cutoffs=(.002, -1.))
    state, change = update_pool(update, torch.ones((2, 64)), initial, CONFIG, SENSOR, SCALE)
    torch.testing.assert_close(state.graph.graph.in_degree, initial.graph.graph.in_degree, rtol=0, atol=0)
    assert not torch.equal(state.graph.graph.edge_attr, initial.graph.graph.edge_attr)
    assert change.changed_nodes.tolist() == [True, True]
    assert state.work["topology_changed_nodes"].tolist() == [True, True]
    assert not bool(state.work["input_changed_sources"].any())
    _assert_oracle(state)


def test_full_snapshot_mean_gradient_reaches_all_prefix_features_and_post_pool_spline_parameters():
    raw = _update(None, [[1, 1, .001, 1, 0], [2, 1, .001, 1, 0], [5, 1, .001, 1, 0], [6, 1, .001, 1, 0]]).state
    features = _features(4, dtype=torch.float32).requires_grad_()
    state = pool_graph(raw, features, CONFIG, SENSOR, SCALE)
    layer = PaperSplineConv(64, 64, spline_backend="torch").eval()
    graph = state.graph.graph
    output, _ = layer(graph.node_features, graph.edge_index, graph.edge_attr, in_degree=graph.in_degree)
    loss = (output - .3).square().mean()
    loss.backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    assert bool((features.grad.abs().sum(1) > 0).all())
    torch.testing.assert_close(features.grad[0], features.grad[1], rtol=0, atol=0)
    torch.testing.assert_close(features.grad[2], features.grad[3], rtol=0, atol=0)
    for parameter in layer.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    previous = layer.weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=.01).step()
    assert not torch.equal(previous, layer.weight)


def test_snapshot_double_gradcheck_and_zero_valued_features_remain_real_topology():
    raw = _update(None, [[1, 1, .001, 1, 0], [2, 1, .001, 1, 0], [5, 1, .001, 1, 0]]).state
    features = torch.zeros((3, 64), dtype=torch.float64, requires_grad=True)
    state = pool_graph(raw, features, CONFIG, SENSOR, SCALE)
    assert state.graph.graph.edge_index.shape == (2, 2)
    assert state.edge_refcounts.tolist() == [2, 2]
    assert not bool(state.graph.graph.node_features.any())
    operation = lambda value: pool_graph(raw, value, CONFIG, SENSOR, SCALE).graph.graph.node_features
    assert torch.autograd.gradcheck(operation, (features,), fast_mode=True)


@pytest.mark.parametrize("storage", ["implicit_radius", "materialized"])
def test_empty_graph_and_all_expired_graph(storage):
    empty = _update(None, [], storage=storage)
    state, change = update_pool(empty, torch.empty((0, 64)), None, CONFIG, SENSOR, SCALE)
    assert state.graph.graph.node_features.shape == (0, 64)
    assert state.graph.graph.edge_index.shape == (2, 0)
    assert change.old_indices.numel() == change.changed_nodes.numel() == 0
    first = _update(empty.state, [[1, 1, .001, 1, 0]], storage=storage)
    state, _ = update_pool(first, torch.ones((1, 64)), state, CONFIG, SENSOR, SCALE)
    expired = _update(first.state, [], storage=storage, cutoffs=(.1, .1))
    state, _ = update_pool(expired, torch.empty((0, 64)), state, CONFIG, SENSOR, SCALE)
    assert state.counts.numel() == state.edge_refcounts.numel() == 0


@pytest.mark.parametrize("wrong", [{}, {"spatial_cell_pixels": 4, "temporal_cell_seconds": 0},
                                   {"spatial_cell_pixels": True, "temporal_cell_seconds": .004},
                                   {**CONFIG, "radius": .2}, {**CONFIG, "edge_chunk_size": 0}])
def test_invalid_or_unapproved_pool_contract_fails(wrong):
    raw = _update(None, []).state
    with pytest.raises(ValueError):
        pool_graph(raw, torch.empty((0, 64)), wrong, SENSOR, SCALE)


def test_changed_config_and_rewritten_retained_geometry_fail_without_mutating_state():
    first = _update(None, [[1, 1, .001, 1, 0], [5, 1, .001, 1, 0]])
    features = _features(2)
    state = pool_graph(first.state, features, CONFIG, SENSOR, SCALE)
    identity = GraphUpdate(first.state, torch.arange(2), torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="contract changed"):
        update_pool(identity, features, state, {**CONFIG, "spatial_cell_pixels": 8}, SENSOR, SCALE)
    with pytest.raises(ValueError, match="duplicated"):
        update_pool(GraphUpdate(first.state, torch.tensor([0, 0]), identity.changed_nodes), features,
                    state, CONFIG, SENSOR, SCALE)
    with pytest.raises(ValueError, match="active_sources"):
        update_pool(identity, features, state, CONFIG, SENSOR, SCALE, active_sources=torch.tensor([2]))
    changed_raw = _update(None, [[2, 1, .001, 1, 0], [5, 1, .001, 1, 0]])
    with pytest.raises(ValueError, match="Retained raw geometry"):
        update_pool(GraphUpdate(changed_raw.state, torch.arange(2), identity.changed_nodes), features,
                    state, CONFIG, SENSOR, SCALE)
    changed_radius = _update(None, [[1, 1, .001, 1, 0], [5, 1, .001, 1, 0]], radius=.1)
    with pytest.raises(ValueError, match="radius geometry contract"):
        update_pool(GraphUpdate(changed_radius.state, torch.arange(2), identity.changed_nodes), features,
                    state, CONFIG, SENSOR, SCALE)
    torch.testing.assert_close(state.raw_features, features, rtol=0, atol=0)


def test_materialized_old_old_edge_changes_are_rejected_instead_of_reusing_stale_contributors():
    first = _update(None, [[1, 1, .001, 1, 0], [5, 1, .001, 1, 0]], storage="materialized")
    features = _features(2)
    state = pool_graph(first.state, features, CONFIG, SENSOR, SCALE)
    old = first.state.graph
    changed_graph = EventGraph(old.node_features, old.positions, old.edge_index, old.edge_attr * .5)
    changed = StreamGraph(changed_graph, first.state.node_batch, first.state.timestamps)
    update = GraphUpdate(changed, torch.arange(2), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="Retained materialized raw edge attributes"):
        update_pool(update, features, state, CONFIG, SENSOR, SCALE)


def test_centroid_only_change_is_structural_but_same_feature_is_not_an_input_pulse():
    first = _update(None, [[1, 1, .001, 1, 0], [2, 1, .001, 1, 0]])
    state = pool_graph(first.state, torch.ones((2, 64)), CONFIG, SENSOR, SCALE)
    update = _update(first.state, [[3, 1, .001, 1, 0]])
    pooled, changed = update_pool(update, torch.ones((3, 64)), state, CONFIG, SENSOR, SCALE)
    assert pooled.graph.graph.edge_index.numel() == 0
    assert pooled.work["topology_changed_nodes"].tolist() == [True]
    assert pooled.work["input_changed_sources"].tolist() == [False]
    assert changed.changed_nodes.tolist() == [True]


def test_cold_start_separates_new_topology_from_explicit_active_pulse_sources():
    update = _update(None, [[1, 1, .001, 1, 0], [5, 1, .001, 1, 0]])
    state, changed = update_pool(update, torch.ones((2, 64)), None, CONFIG, SENSOR, SCALE,
                                 active_sources=torch.tensor([1]))
    assert state.work["topology_changed_nodes"].tolist() == [True, True]
    assert state.work["input_changed_sources"].tolist() == [False, True]
    assert changed.changed_nodes.tolist() == [True, True]
    for key in ("topology_changed_nodes", "input_changed_sources"):
        assert state.work[key].dtype == torch.bool and state.work[key].device == state.counts.device


@pytest.mark.parametrize("seed", list(range(5)))
def test_small_random_stream_updates_match_fresh_means_counts_and_edge_attributes(seed):
    generator = torch.Generator().manual_seed(seed)
    raw, previous = None, None
    for step in range(12):
        amount = (step * 3 + seed) % 7
        pixels = torch.randint(0, 16, (amount, 2), generator=generator)
        times = step * .002 + torch.randint(0, 3, (amount,), generator=generator).double() * .0003
        batches = torch.randint(0, 3, (amount,), generator=generator)
        rows = torch.cat((pixels, times[:, None], torch.ones((amount, 1)), batches[:, None]), dim=1)
        update = _update(raw, rows.tolist(), cutoffs=(step * .002 - .006,) * 3)
        features = _features(len(update.state.timestamps), seed=step + seed)
        state, _ = update_pool(update, features, previous, CONFIG, SENSOR, SCALE)
        fresh = pool_graph(update.state, features, CONFIG, SENSOR, SCALE)
        for field in ("cluster_keys", "counts", "raw_to_cluster", "edge_refcounts"):
            torch.testing.assert_close(getattr(state, field), getattr(fresh, field), rtol=0, atol=0)
        for field in ("feature_sums", "position_sums", "timestamp_sums", "edge_pseudo_sums"):
            torch.testing.assert_close(getattr(state, field), getattr(fresh, field), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(state.graph.graph.edge_index, fresh.graph.graph.edge_index, rtol=0, atol=0)
        torch.testing.assert_close(state.graph.graph.edge_attr, fresh.graph.graph.edge_attr, rtol=1e-11, atol=1e-12)
        raw, previous = update.state, state
