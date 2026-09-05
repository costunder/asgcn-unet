"""Disjoint graph batches and chronological, independent sequence scheduling."""

from __future__ import annotations

import random
import zipfile
from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import torch
from PIL import Image
from torch.utils.data import Sampler, Subset

from .graph import EventGraph


class PackedSampleBatch:
    """Pack events/targets before transfer; sample dictionaries only expose views.

    This deliberately does not inherit ``list`` or ``Sequence``: DataLoader's
    pin-memory dispatcher must call our method once for each packed allocation,
    rather than recursively pinning the individual sample views again.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        events: torch.Tensor,
        event_counts: tuple[int, ...],
        targets: torch.Tensor | None,
    ) -> None:
        self.events = events
        self.event_counts = event_counts
        self.targets = targets
        self.sensor_size = tuple(samples[0]["sensor_size"])
        self._samples = []
        offset = 0
        for index, (sample, count) in enumerate(zip(samples, event_counts, strict=True)):
            view = dict(sample)
            view["events"] = events[offset : offset + count]
            if targets is not None:
                view["target"] = targets[index]
            self._samples.append(view)
            offset += count

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def to(self, device: torch.device) -> PackedSampleBatch:
        return PackedSampleBatch(
            self._samples,
            self.events.to(device, non_blocking=True),
            self.event_counts,
            None if self.targets is None else self.targets.to(device, non_blocking=True),
        )

    def pin_memory(self) -> PackedSampleBatch:
        return PackedSampleBatch(
            self._samples,
            self.events.pin_memory(),
            self.event_counts,
            None if self.targets is None else self.targets.pin_memory(),
        )


def pack_samples(samples: list[dict[str, Any]] | PackedSampleBatch) -> PackedSampleBatch:
    """Collate one geometry bucket without padding/dropping any event or target."""
    if isinstance(samples, PackedSampleBatch):
        return samples
    if not samples:
        raise ValueError("Cannot pack an empty sample batch")
    sensor_size = tuple(samples[0]["sensor_size"])
    if any(tuple(sample["sensor_size"]) != sensor_size for sample in samples):
        raise ValueError("A packed batch must contain one shared sensor_size")
    events = [sample["events"] for sample in samples]
    reference = events[0]
    if any(
        value.ndim != 2 or value.shape[1:] != reference.shape[1:]
        or value.device != reference.device or value.dtype != reference.dtype
        for value in events
    ):
        raise ValueError("Packed events must share feature shape, device and dtype")
    have_targets = ["target" in sample for sample in samples]
    if any(have_targets) and not all(have_targets):
        raise ValueError("Packed samples must either all have targets or all omit them")
    targets = torch.stack([sample["target"] for sample in samples]) if all(have_targets) else None
    return PackedSampleBatch(
        samples,
        torch.cat(events, dim=0),
        tuple(int(value.shape[0]) for value in events),
        targets,
    )


def move_batch(
    samples: list[dict[str, Any]] | PackedSampleBatch, device: torch.device
) -> PackedSampleBatch:
    """One event transfer and one target transfer for the physical mini-batch."""
    return pack_samples(samples).to(device)


def pack_calibration_samples(
    samples: list[dict[str, Any]] | PackedSampleBatch,
) -> PackedSampleBatch:
    """CPU calibration collate without stacking or retaining target images.

    Calibration observes only encoder events. Shallow dictionary views preserve
    every event and metadata field, while the caller's target tensors and sample
    dictionaries remain unchanged. A module-level callable also supports spawn
    DataLoader workers without initializing CUDA in the decoding processes.
    """
    if any(sample["events"].device.type != "cpu" for sample in samples):
        raise ValueError("Calibration collation requires CPU event tensors")
    views = [
        {key: value for key, value in sample.items() if key != "target"}
        for sample in samples
    ]
    return pack_samples(views)


def sequence_key(sample: dict[str, Any]) -> tuple[str, str]:
    """Identify a recurrent stream without merging distinct files of one scene."""
    metadata = sample.get("metadata", sample)
    if not isinstance(metadata, dict):
        raise TypeError("Sequence batching requires dictionary sample metadata")
    identity = metadata.get("sequence_id") or metadata.get("scene")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Sequence batching requires a nonempty scene or sequence_id")
    source = metadata.get("source_file", "")
    if not isinstance(source, str):
        raise TypeError("Sequence batching source_file must be a string")
    return identity, source


def concatenate_graphs(graphs: list[EventGraph]) -> EventGraph:
    """Form a disjoint union; no radius query is performed across sample borders."""
    if not graphs:
        raise ValueError("Cannot concatenate an empty graph batch")
    device = graphs[0].node_features.device
    dtype = graphs[0].node_features.dtype
    offset = 0
    edges = []
    degrees = []
    for graph in graphs:
        if graph.node_features.device != device or graph.node_features.dtype != dtype:
            raise ValueError("Batched graphs must share node dtype and device")
        if (graph.positions.device != device or graph.edge_index.device != device
                or graph.edge_attr.device != device):
            raise ValueError("Batched graph topology and nodes must share a device")
        if (graph.positions.dtype != graphs[0].positions.dtype
                or graph.edge_attr.dtype != graphs[0].edge_attr.dtype):
            raise ValueError("Batched graph positions and edge attributes must share their dtypes")
        edges.append(graph.edge_index + offset)
        assert graph.in_degree is not None
        degrees.append(graph.in_degree)
        offset += graph.node_features.shape[0]
    return EventGraph(
        node_features=torch.cat([graph.node_features for graph in graphs], dim=0),
        positions=torch.cat([graph.positions for graph in graphs], dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_attr=torch.cat([graph.edge_attr for graph in graphs], dim=0),
        in_degree=torch.cat(degrees, dim=0),
    )


class SequenceBatchSampler(Sampler[list[int]]):
    """Refill independent sequence lanes without shuffling frames within a stream.

    Every selected dataset index appears exactly once per epoch, including empty
    event intervals and final partial batches. A batch contains at most one frame
    from each stream and one post-crop sensor shape. Shape changes do not create a
    new concurrent copy of a stream; the training loop must reset its recurrent
    state at the same shape/index discontinuities as ordinary framewise training.

    EventHDR image dimensions are read from HDF5 metadata only, never GT pixels.
    ``shuffle_sequences`` changes only the deterministic lane admission order.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        shuffle_sequences: bool = False,
        seed: int = 2026,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Sequence batch_size must be a positive integer")
        if not isinstance(shuffle_sequences, bool):
            raise TypeError("shuffle_sequences must be a boolean")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("Sequence sampler seed must be an integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_sequences = shuffle_sequences
        self.seed = seed
        self.epoch = 0
        base = dataset
        selected = list(range(len(dataset)))
        while isinstance(base, Subset):
            selected = [int(base.indices[index]) for index in selected]
            base = base.dataset
        if len(set(selected)) != len(selected):
            raise ValueError("Sequence batching does not permit duplicate selected dataset indices")
        records = getattr(base, "samples", None)
        if not isinstance(records, list) or len(records) != len(base):
            raise TypeError("Sequence batching requires an indexed dataset.samples list")
        crop_size = getattr(base, "crop_size", None)
        self.sample_count = len(selected)
        self._streams: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
        shapes: list[tuple[int, int]] = []
        previous_indices: dict[tuple[str, str], int] = {}
        with ExitStack() as stack:
            handles: dict[Path, h5py.File] = {}
            zip_handles: dict[Path, zipfile.ZipFile] = {}
            for sample_index, original_index in enumerate(selected):
                record = records[original_index]
                if not isinstance(record, dict):
                    raise TypeError("Sequence batching sample records must be dictionaries")
                key = sequence_key(record)
                index = record.get("sequence_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError("Sequence batching requires nonnegative integer sequence_index")
                if key in previous_indices and index <= previous_indices[key]:
                    raise ValueError("Each sequence must be indexed in strictly chronological order")
                previous_indices[key] = index
                self._streams.setdefault(key, []).append(sample_index)
                shape = record.get("sensor_size") or record.get("shape")
                if shape is None:
                    path = Path(record["path"])
                    if path.suffix.lower() == ".zip":
                        if path not in zip_handles:
                            zip_handles[path] = stack.enter_context(zipfile.ZipFile(path))
                        with zip_handles[path].open(record["target_name"]) as member, Image.open(member) as image:
                            shape = (image.height, image.width)
                    else:
                        if path not in handles:
                            handles[path] = stack.enter_context(h5py.File(path, "r"))
                        image = handles[path]["images"][record["image_key"]]
                        if not isinstance(image, h5py.Dataset) or image.ndim not in (2, 3):
                            raise ValueError("Sequence batching requires HxW or HxWxC image metadata")
                        shape = image.shape[:2]
                if not isinstance(shape, (tuple, list)) or len(shape) != 2:
                    raise ValueError("Sequence batching sensor_size must contain height and width")
                height, width = (int(value) for value in shape)
                if crop_size is not None:
                    height = min(height, int(crop_size[0]))
                    width = min(width, int(crop_size[1]))
                if height < 1 or width < 1:
                    raise ValueError("Sequence batching sensor dimensions must be positive")
                shapes.append((height, width))
        self._shapes = tuple(shapes)
        # Last selected chronological index, not the last index in the source
        # file: Subsets may end early. Evict state after this sample succeeds.
        self.final_sequence_indices = MappingProxyType(previous_indices)
        self.sequence_count = len(self._streams)
        self._batches = self._schedule()

    @property
    def sample_sensor_sizes(self) -> tuple[tuple[int, int], ...]:
        """Post-crop geometry indexed by the sampler's selected dataset indices."""
        return self._shapes

    @property
    def sequence_indices(self) -> tuple[tuple[int, ...], ...]:
        """Chronological dataset indices, available without decoding events/targets."""
        return tuple(tuple(indices) for indices in self._streams.values())

    def _schedule(self) -> tuple[tuple[int, ...], ...]:
        streams = [tuple(indices) for indices in self._streams.values()]
        if self.shuffle_sequences:
            random.Random(self.seed + self.epoch).shuffle(streams)
        pending = deque(deque(indices) for indices in streams)
        lanes: list[deque[int] | None] = [None] * min(self.batch_size, len(streams))
        batches: list[tuple[int, ...]] = []
        next_lane = 0
        while pending or any(lanes):
            for lane in range(len(lanes)):
                if not lanes[lane] and pending:
                    lanes[lane] = pending.popleft()
            # Round-robin shape selection prevents a long sequence from starving
            # lanes with a different resolution. Matching lanes advance together.
            active = next(
                (next_lane + offset) % len(lanes)
                for offset in range(len(lanes))
                if lanes[(next_lane + offset) % len(lanes)]
            )
            assert lanes[active] is not None
            shape = self._shapes[lanes[active][0]]
            selected = []
            for lane in lanes:
                if lane and self._shapes[lane[0]] == shape:
                    selected.append(lane.popleft())
            batches.append(tuple(selected))
            next_lane = (active + 1) % len(lanes)
        return tuple(batches)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Sequence sampler epoch must be a nonnegative integer")
        self.epoch = epoch
        self._batches = self._schedule()

    def __iter__(self) -> Iterator[list[int]]:
        return (list(batch) for batch in self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class ShapeBatchSampler(SequenceBatchSampler):
    """Non-recurrent encoder calibration can batch frames from the same stream."""

    def _schedule(self) -> tuple[tuple[int, ...], ...]:
        buckets: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        for index, shape in enumerate(self._shapes):
            buckets.setdefault(shape, []).append(index)
        return tuple(
            tuple(indices[start : start + self.batch_size])
            for indices in buckets.values()
            for start in range(0, len(indices), self.batch_size)
        )
