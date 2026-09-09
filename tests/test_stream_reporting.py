"""CPU reporting regressions with explicit synthetic counters, not research results."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from asgcn_unet.batching import SequenceBatchSampler, pack_samples
from asgcn_unet.evaluation_batches import evaluation_frames
from asgcn_unet.stream_reporting import aggregate_stream_execution
from tests.test_batching import _model
from tests.test_evaluation_batches import DiagnosticDataset


def _execution(multiplier=1):
    return {"arrival_updates": 2 * multiplier, "readout_updates": multiplier,
            "incoming_events": 7 * multiplier, "training_dense_snapshot": False,
            "updated_nodes_per_layer": [3 * multiplier, 4 * multiplier],
            "message_edges_per_layer": [5 * multiplier, 6 * multiplier],
            "projected_sources_per_layer": [2 * multiplier, 3 * multiplier],
            "topology_indexed_edges": 8 * multiplier}


def test_batch_shared_work_counted_once_not_per_lane_and_originals_unchanged():
    execution = _execution()
    diagnostics = [{"stream_execution": execution} for _ in range(16)]
    saved = copy.deepcopy(diagnostics)
    first = aggregate_stream_execution(None, diagnostics)
    assert first == {"scope": "batch_once", "physical_batches": 1, "frames": 16, **execution}
    second = aggregate_stream_execution(first, [{"stream_execution": _execution(2)} for _ in range(3)])
    assert second["physical_batches"] == 2 and second["frames"] == 19
    assert second["arrival_updates"] == 6 and second["incoming_events"] == 21
    assert second["message_edges_per_layer"] == [15, 18]
    assert second["topology_indexed_edges"] == 24
    assert second["training_dense_snapshot"] is False
    assert first["message_edges_per_layer"] == [5, 6]
    assert diagnostics == saved


def test_static_diagnostics_do_not_create_stream_fields():
    assert aggregate_stream_execution(None, [{"nodes": 5}, {"nodes": 3}]) is None


def test_missing_optional_counters_are_not_fabricated():
    value = {"arrival_updates": 0, "readout_updates": 0, "incoming_events": 7, "training_dense_snapshot": True}
    total = aggregate_stream_execution(None, [{"stream_execution": value}])
    assert "message_edges_per_layer" not in total and "topology_indexed_edges" not in total


@pytest.mark.parametrize("field,value", [
    ("arrival_updates", True), ("incoming_events", -1), ("topology_indexed_edges", float("nan")),
    ("training_dense_snapshot", 0), ("updated_nodes_per_layer", [1, True]),
    ("message_edges_per_layer", [1]), ("projected_sources_per_layer", []),
])
def test_invalid_counters_are_not_summed(field, value):
    execution = _execution()
    execution[field] = value
    with pytest.raises((TypeError, ValueError)):
        aggregate_stream_execution(None, [{"stream_execution": execution}])


def test_disagreeing_lanes_and_static_stream_mixing_fail():
    with pytest.raises(ValueError, match="Per-lane"):
        aggregate_stream_execution(None, [{"stream_execution": _execution()}, {"stream_execution": _execution(2)}])
    with pytest.raises(ValueError, match="missing and present"):
        aggregate_stream_execution(None, [{"stream_execution": _execution()}, {}])
    total = aggregate_stream_execution(None, [{"stream_execution": _execution()}])
    with pytest.raises(ValueError, match="static"):
        aggregate_stream_execution(total, [{}])


def test_mode_and_layer_coverage_changes_fail_without_mutating_total():
    total = aggregate_stream_execution(None, [{"stream_execution": _execution()}])
    original = copy.deepcopy(total)
    mode = _execution()
    mode["training_dense_snapshot"] = True
    with pytest.raises(ValueError, match="training_dense_snapshot"):
        aggregate_stream_execution(total, [{"stream_execution": mode}])
    missing = _execution()
    del missing["topology_indexed_edges"]
    with pytest.raises(ValueError, match="coverage"):
        aggregate_stream_execution(total, [{"stream_execution": missing}])
    shorter = _execution()
    for key in ("updated_nodes_per_layer", "message_edges_per_layer", "projected_sources_per_layer"):
        shorter[key] = shorter[key][:1]
    with pytest.raises(ValueError, match="lengths changed"):
        aggregate_stream_execution(total, [{"stream_execution": shorter}])
    assert total == original


@pytest.mark.parametrize("streaming", [False, True])
def test_evaluation_aggregates_after_actual_forward_once_including_partial_tail(streaming):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = DiagnosticDataset(lengths=(2, 1))
        sampler = SequenceBatchSampler(dataset, 2)
        plan = list(sampler)
        loader = DataLoader(dataset, batch_sampler=plan, collate_fn=pack_samples)
        model = _model(recurrent=True).eval()
        calls = []
        def run_forward(samples, contexts, timer):
            prediction, diagnostics = model.forward_batch(samples, [item[0] for item in contexts], timing=timer)
            calls.append(len(samples))
            if streaming:
                # These are explicit reporting-only fixture counters, not claims
                # about this static fixture model's executed operations.
                measured = _execution(len(calls))
                for detail in diagnostics:
                    detail["stream_execution"] = measured
            return prediction, diagnostics
        statistics = {}
        rows = list(evaluation_frames(
            loader, plan, device=torch.device("cpu"), run_forward=run_forward,
            independent_sequences=True, statistics=statistics, timing_steps=2, timing_warmup=0,
            final_sequence_indices=sampler.final_sequence_indices,
        ))
        assert calls == [2, 1] and len(rows) == 3
        if streaming:
            total = statistics["stream_execution"]
            assert total["scope"] == "batch_once" and total["physical_batches"] == 2
            assert total["frames"] == 3 and total["incoming_events"] == 21
            assert total["message_edges_per_layer"] == [15, 18]
        else:
            assert "stream_execution" not in statistics
    finally:
        torch.set_num_threads(previous_threads)
