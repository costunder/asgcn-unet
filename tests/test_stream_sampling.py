"""CPU-only synthetic tests; no production data/model reductions."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from asgcn_unet.batching import pack_samples
from asgcn_unet.stream_sampling import replace_record_groups, sample_stream_batch


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _sample(name, times, *, first_id=0, frame=0):
    events = torch.tensor([[index % 8, index % 8, time, 1 if index % 2 else -1]
                           for index, time in enumerate(times)], dtype=torch.float64).reshape(-1, 4)
    groups = []
    for index, time in enumerate(times):
        if index and time == times[index - 1]:
            groups[-1] += 1
        else:
            groups.append(1)
    return {
        "sample_id": f"{name}/{frame}", "sensor_size": (8, 8),
        "events": events, "target": torch.full((1, 8, 8), 0.25),
        "event_ids": torch.tensor([[0, first_id + index] for index in range(len(times))],
                                  dtype=torch.long).reshape(-1, 2),
        "metadata": {"sequence_id": name, "sequence_index": frame, "group": "synthetic",
                     "stream_time": {"schema": "physical_seconds_v1", "sequence_origin_seconds": 0.,
                                     "interval_start_seconds": 0., "interval_end_seconds": 10.,
                                     "arrival_group_counts": groups}},
    }


def test_r1_preserves_packed_identity_every_event_and_raw_counter():
    packed = pack_samples([_sample("a", [1, 1, 2]), _sample("b", [])])
    result = sample_stream_batch(packed, [7, 4], factor=1)
    assert result.packed is packed and result.keep_mask.tolist() == [True] * 3
    assert result.next_offsets == (10, 4) and result.arrival_group_counts == ((2, 1), ())


def test_independent_lanes_use_global_ordinals_and_recompute_equal_time_groups():
    packed = pack_samples([_sample("a", [1, 1, 1, 2, 2, 3, 3, 3]),
                           _sample("b", [1, 1, 2, 2, 2], first_id=100)])
    metadata_before = deepcopy([sample["metadata"] for sample in packed])
    result = sample_stream_batch(packed, [0, 2], factor=3)
    assert result.packed.event_counts == (3, 2)
    assert result.packed.event_ids[:, 1].tolist() == [0, 3, 6, 101, 104]
    assert result.arrival_group_counts == ((1, 1, 1), (1, 1))
    assert result.next_offsets == (8, 7)
    assert result.packed.targets is packed.targets and result.packed.sensor_size == packed.sensor_size
    assert [sample["metadata"] for sample in packed] == metadata_before
    torch.testing.assert_close(result.packed.events, packed.events[result.keep_mask], rtol=0, atol=0)
    for index, sample in enumerate(result.packed):
        assert sample["metadata"]["stream_time"]["arrival_group_counts"] == result.arrival_group_counts[index]
        assert sample["metadata"]["sequence_id"] == packed[index]["metadata"]["sequence_id"]


def test_no_selected_events_keeps_frame_target_and_advances_raw_offset():
    packed = pack_samples([_sample("a", [1, 1], first_id=1), _sample("b", [])])
    result = sample_stream_batch(packed, [1, 8], factor=4)
    assert len(result.packed) == 2 and result.packed.event_counts == (0, 0)
    assert result.packed.events.shape == (0, 4) and result.packed.event_ids.shape == (0, 2)
    assert result.packed.targets is packed.targets
    assert result.next_offsets == (3, 8) and result.arrival_group_counts == ((), ())
    next_frame = sample_stream_batch(pack_samples([_sample("a", [2, 2], first_id=3, frame=1)]),
                                    [result.next_offsets[0]], factor=4)
    assert next_frame.packed.event_ids[:, 1].tolist() == [4] and next_frame.next_offsets == (5,)


def test_all_empty_lanes_preserve_raw_offsets_without_inventing_arrivals():
    packed = pack_samples([_sample("a", []), _sample("b", [])])
    result = sample_stream_batch(packed, [3, 8], factor=7)
    assert result.packed.event_counts == (0, 0) and result.keep_mask.shape == (0,)
    assert result.next_offsets == (3, 8) and result.arrival_group_counts == ((), ())
    assert result.packed.targets is packed.targets


def test_packed_selection_matches_independent_lanes_without_shared_phase():
    samples = [_sample("a", [1] * 9), _sample("b", [2] * 6, first_id=100), _sample("c", [])]
    offsets = [8, 1, 23]
    batched = sample_stream_batch(pack_samples(samples), offsets, factor=4)
    singles = [sample_stream_batch(pack_samples([sample]), [prior], factor=4)
               for sample, prior in zip(samples, offsets, strict=True)]
    torch.testing.assert_close(batched.packed.events, torch.cat([result.packed.events for result in singles]))
    torch.testing.assert_close(batched.packed.event_ids, torch.cat([result.packed.event_ids for result in singles]))
    assert batched.next_offsets == tuple(result.next_offsets[0] for result in singles)
    assert batched.arrival_group_counts == tuple(result.arrival_group_counts[0] for result in singles)


@pytest.mark.parametrize("factor", [1, 2, 3, 7, 100])
def test_event_identity_selection_is_invariant_to_frame_partition(factor):
    times = [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 5]
    full = sample_stream_batch(pack_samples([_sample("a", times)]), [0], factor=factor)
    parts, offset = [], 0
    for number, (start, stop) in enumerate([(0, 2), (2, 2), (2, 5), (5, 9), (9, 12)]):
        result = sample_stream_batch(pack_samples([_sample("a", times[start:stop], first_id=start, frame=number)]),
                                     [offset], factor=factor)
        parts.append(result.packed.event_ids)
        offset = result.next_offsets[0]
    torch.testing.assert_close(torch.cat(parts), full.packed.event_ids, rtol=0, atol=0)
    assert offset == len(times)


def test_large_python_counters_do_not_overflow_tensor_ordinals():
    prior = 2**100 + 9
    packed = pack_samples([_sample("a", [1] * 11)])
    result = sample_stream_batch(packed, [prior], factor=7)
    assert result.keep_mask.tolist() == [(prior + index) % 7 == 0 for index in range(11)]
    assert result.next_offsets == (prior + 11,)


def test_int64_factor_boundary_does_not_add_overflowing_residues():
    factor = torch.iinfo(torch.long).max
    packed = pack_samples([_sample("a", [1] * 5)])
    result = sample_stream_batch(packed, [factor - 2], factor=factor)
    assert result.keep_mask.tolist() == [False, False, True, False, False]


@pytest.mark.parametrize("prior", [[], [0, 1], [-1], [True], [0.0], [None], "0"])
def test_invalid_offsets_fail_explicitly(prior):
    with pytest.raises(ValueError, match="prior_offsets"):
        sample_stream_batch(pack_samples([_sample("a", [1])]), prior, factor=2)


@pytest.mark.parametrize("factor", [0, -1, True, 1.0, None, 2**63])
def test_invalid_factor_is_not_replaced_with_default(factor):
    with pytest.raises(ValueError, match="factor"):
        sample_stream_batch(pack_samples([_sample("a", [1])]), [0], factor=factor)


@pytest.mark.parametrize("groups", [[0, 2], [True, 1], [1], None])
def test_invalid_raw_group_metadata_cannot_authorize_sampling(groups):
    sample = _sample("a", [1, 1])
    sample["metadata"]["stream_time"]["arrival_group_counts"] = groups
    with pytest.raises((ValueError, TypeError), match="arrival_group_counts"):
        sample_stream_batch(pack_samples([sample]), [0], factor=3)


def test_selection_preserves_gradients_for_only_selected_raw_rows():
    packed = pack_samples([_sample("a", [1] * 6), _sample("b", [2] * 4)])
    packed.events.requires_grad_(True)
    result = sample_stream_batch(packed, [1, 0], factor=3)
    result.packed.events.square().sum().backward()
    expected = 2 * packed.events.detach() * result.keep_mask[:, None]
    torch.testing.assert_close(packed.events.grad, expected, rtol=0, atol=0)


def test_record_helper_changes_only_group_counts():
    records = [(('synthetic', 'a'), 2, 1., 2., 0., (2, 3)), (('synthetic', 'b'), 4, 3., 4., 0., (1,))]
    original = deepcopy(records)
    updated = replace_record_groups(records, ((1, 1), ()))
    assert records == original and updated is not records
    assert [row[:5] for row in updated] == [row[:5] for row in original]
    assert [row[5] for row in updated] == [(1, 1), ()]
    with pytest.raises(ValueError, match="One sampled"):
        replace_record_groups(records, ())
    with pytest.raises(ValueError, match="six-field"):
        replace_record_groups([(0,)], [()])
