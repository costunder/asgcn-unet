"""Bounded tensor hash accumulation of coarse-edge witnesses, never raw E storage.

Only each incoming chunk is coalesced. Existing Q keys are rehashed on geometric
capacity growth and sorted once on export, not sorted afresh for every E chunk.
Hash collisions are resolved by full two-column key equality, never truncated.
"""

import torch


class QuotientAccumulator:
    def __init__(self, pairs, counts, sums):
        self.device = pairs.device
        self.capacity = 0
        self.used = 0
        self.rehashes = 0
        self.probe_rounds = 0
        self.chunks = 0
        self.keys = pairs.new_empty((0, 2))
        self.counts = counts.new_empty(0)
        self.sums = sums.new_empty((0, 1))
        self._reserve(len(pairs))
        self._insert_unique(pairs, counts, sums)

    def _slots(self, pairs):
        # Signed int64 wraparound is intentional hash arithmetic. Stored full
        # keys, not hashes, decide equality; large node IDs cannot alias edges.
        mixed = (pairs[:, 0] * -7046029254386353131) ^ (pairs[:, 1] * -4417276706812531889)
        mixed = (mixed ^ (mixed >> 30)) * -4658895280553007687
        mixed = (mixed ^ (mixed >> 27)) * -7723592293110705685
        return (mixed ^ (mixed >> 31)) & (self.capacity - 1)

    def _reserve(self, maximum_keys):
        capacity = max(16, self.capacity)
        while capacity < 2 * maximum_keys:
            capacity *= 2
        if capacity == self.capacity:
            return
        occupied = self.keys[:, 0] >= 0
        keys, counts, sums = self.keys[occupied], self.counts[occupied], self.sums[occupied]
        self.capacity = capacity
        self.keys = torch.full((capacity, 2), -1, dtype=torch.long, device=self.device)
        self.counts = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.sums = torch.zeros((capacity, 1), dtype=torch.float64, device=self.device)
        self.used = 0
        self.rehashes += 1
        self._insert_unique(keys, counts, sums)

    def _insert_unique(self, pairs, counts, sums):
        pending = torch.arange(len(pairs), device=self.device)
        slots = self._slots(pairs)
        rounds = 0
        while pending.numel():
            rounds += 1
            if rounds > self.capacity:
                raise RuntimeError("Quotient hash table unexpectedly exhausted; no edges were dropped")
            self.probe_rounds += 1
            current = self.keys[slots]
            matched = (current == pairs[pending]).all(dim=1)
            matched_slots = slots[matched]
            self.counts.index_add_(0, matched_slots, counts[pending[matched]])
            self.sums.index_add_(0, matched_slots, sums[pending[matched]])
            empty = torch.nonzero(current[:, 0] < 0, as_tuple=True)[0]
            # Concurrent candidates for one empty slot choose one winner.
            # The temporary reduction is chunk-sized, not table/Q-sized.
            unique_slots, inverse = torch.unique(slots[empty], return_inverse=True)
            winners = torch.full((len(unique_slots),), len(pending), dtype=torch.long, device=self.device)
            winners.scatter_reduce_(0, inverse, empty, reduce="amin", include_self=True)
            chosen = pending[winners]
            self.keys[unique_slots] = pairs[chosen]
            self.counts[unique_slots] = counts[chosen]
            self.sums[unique_slots] = sums[chosen]
            self.used += len(chosen)
            done = matched.clone()
            done[winners] = True
            pending = pending[~done]
            slots = (slots[~done] + 1) & (self.capacity - 1)

    def add(self, pairs, counts, sums):
        if not len(pairs):
            return
        self.chunks += 1
        unique, inverse = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
        compact_counts = counts.new_zeros(len(unique)).index_add_(0, inverse, counts)
        compact_sums = sums.new_zeros((len(unique), 1)).index_add_(0, inverse, sums)
        self._reserve(self.used + len(unique))
        self._insert_unique(unique, compact_counts, compact_sums)

    def finish(self, work):
        live = (self.keys[:, 0] >= 0) & (self.counts != 0)
        pairs, counts, sums = self.keys[live], self.counts[live], self.sums[live]
        order = torch.argsort(pairs[:, 1], stable=True)
        order = order[torch.argsort(pairs[order, 0], stable=True)]
        work["quotient_accumulator"] = "tensor_open_addressed_full_pair_equality_v1"
        for key, value in (("quotient_chunks", self.chunks), ("quotient_rehashes", self.rehashes),
                           ("quotient_probe_rounds", self.probe_rounds), ("quotient_final_sorts", 1)):
            work[key] = work.get(key, 0) + value
        work["quotient_peak_table_capacity"] = max(work.get("quotient_peak_table_capacity", 0), self.capacity)
        return pairs[order], counts[order], sums[order]
