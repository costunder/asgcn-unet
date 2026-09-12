"""Small CPU collision/streaming witnesses, not accelerator performance evidence."""

import pytest
import torch

from asgcn_unet.quotient_accumulator import QuotientAccumulator


def empty():
    return QuotientAccumulator(torch.empty((0, 2), dtype=torch.long), torch.empty(0, dtype=torch.long),
                               torch.empty((0, 1), dtype=torch.float64))


@pytest.mark.parametrize("collisions", [False, True])
def test_collision_resolution_growth_duplicate_sums_and_removal(monkeypatch, collisions):
    if collisions:
        monkeypatch.setattr(QuotientAccumulator, "_slots", lambda self, pairs: pairs.new_zeros(len(pairs)))
    table = empty()
    expected = {}
    for step in range(8):
        pairs = torch.tensor([[step, other] for other in range(6)] + [[step, 0], [2**60, step]], dtype=torch.long)
        counts = torch.ones(len(pairs), dtype=torch.long)
        sums = torch.arange(len(pairs), dtype=torch.float64)[:, None] / 16
        table.add(pairs, counts, sums)
        for pair, value in zip(map(tuple, pairs.tolist()), sums[:, 0].tolist()):
            old_count, old_sum = expected.get(pair, (0, 0.))
            expected[pair] = old_count + 1, old_sum + value
    deleted = next(iter(expected))
    count, value = expected.pop(deleted)
    table.add(torch.tensor([deleted]), torch.tensor([-count]), torch.tensor([[-value]], dtype=torch.float64))
    work = {}
    pairs, counts, sums = table.finish(work)
    assert list(map(tuple, pairs.tolist())) == sorted(expected)
    assert counts.tolist() == [expected[pair][0] for pair in sorted(expected)]
    torch.testing.assert_close(sums[:, 0], torch.tensor([expected[pair][1] for pair in sorted(expected)], dtype=torch.float64))
    assert work["quotient_final_sorts"] == 1 and table.rehashes < table.chunks


def test_many_chunks_do_not_sort_accumulated_keys_again(monkeypatch):
    table = empty()
    original_unique = torch.unique
    examined = []
    def unique(value, *args, **kwargs):
        if value.ndim == 2:
            examined.append(len(value))
        return original_unique(value, *args, **kwargs)
    monkeypatch.setattr(torch, "unique", unique)
    for step in range(64):
        pairs = torch.tensor([[step, step + 1], [step, step + 2]], dtype=torch.long)
        table.add(pairs, torch.ones(2, dtype=torch.long), torch.ones((2, 1), dtype=torch.float64))
    work = {}
    pairs, _, _ = table.finish(work)
    assert len(pairs) == 128
    assert max(examined) == 2
    assert work["quotient_final_sorts"] == 1
    assert table.capacity <= 4 * (len(pairs) + 2)


def test_zero_and_negative_counts_are_not_silently_forgiven():
    table = empty()
    table.add(torch.tensor([[1, 2], [1, 2], [2, 3]]), torch.tensor([1, -1, -1]),
              torch.tensor([[.1], [-.1], [-.2]], dtype=torch.float64))
    pairs, counts, sums = table.finish({})
    assert pairs.tolist() == [[2, 3]] and counts.tolist() == [-1]
    torch.testing.assert_close(sums, torch.tensor([[-.2]], dtype=torch.float64))
