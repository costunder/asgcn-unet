"""Explicit tiny CPU smoke/regression fixtures; not research quality results."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from asgcn_unet.batching import SequenceBatchSampler, pack_samples
from asgcn_unet.evaluation_batches import evaluation_frames
from asgcn_unet.metrics import batch_frame_metrics, frame_metrics
from tests.test_batching import _model, _sample


@pytest.fixture(autouse=True, scope="module")
def _cpu_fixture_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class DiagnosticDataset:
    def __init__(self, lengths=(3, 2, 1)):
        self.values = []
        self.samples = []
        for stream, length in enumerate(lengths):
            for frame in range(length):
                sample = _sample(str(stream), count=0 if frame == 1 else 7)
                # Explicit synthetic target for this CPU-only regression fixture.
                sample["target"] = torch.full((1, *sample["sensor_size"]), 0.2 + 0.1 * frame)
                sample["sample_id"] = f"{stream}/{frame}"
                sample["metadata"] = {
                    "scene": str(stream), "sequence_id": str(stream),
                    "sequence_index": frame, "source_file": f"{stream}.h5",
                }
                self.values.append(sample)
                self.samples.append({**sample["metadata"], "sensor_size": sample["sensor_size"]})

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


@pytest.mark.parametrize("mode", ["ann", "snn"])
@pytest.mark.parametrize("dynamics", ["literal_eq15", "standard_if"])
def test_batched_quality_preserves_chronology_empty_frames_metrics_and_tail(mode, dynamics):
    dataset = DiagnosticDataset()

    def collect(batch_size):
        model = _model(recurrent=True).eval()
        model.snn_dynamics = dynamics
        if mode == "snn":
            # Use actual fixture activations and the public conversion path;
            # SNN safety guards remain enabled in both reference and packed runs.
            model.calibrate_batch(pack_samples(dataset.values))
            model.fold_batch_norm()
            model.apply_parameter_normalization()
        sampler = SequenceBatchSampler(dataset, batch_size)
        plan = list(sampler)
        loader = DataLoader(dataset, batch_sampler=plan, collate_fn=pack_samples)

        def run_forward(samples, contexts, timer):
            if batch_size == 1:
                prediction, detail = model.forward_sample(
                    samples[0], inference_mode=mode, simulation_steps=4,
                    recurrent_state=contexts[0][0],
                )
                return prediction, [detail]
            return model.forward_batch(
                samples, [context[0] for context in contexts],
                inference_mode=mode, simulation_steps=4, timing=timer,
            )

        statistics = {}
        rows = evaluation_frames(
            loader, plan, device=torch.device("cpu"), run_forward=run_forward,
            independent_sequences=True, final_sequence_indices=sampler.final_sequence_indices,
            statistics=statistics, timing_steps=2, timing_warmup=0,
        )
        result = {}
        for index, _sample_value, prediction, _target, metrics, details, latency in rows:
            assert index not in result
            result[index] = (prediction.clone(), metrics, details)
            assert latency > 0
        assert sorted(result) == list(range(len(dataset)))
        assert statistics["frames"] == len(dataset)
        assert statistics["latency_scope"] == "physical_batch_completion_not_amortized"
        assert statistics["throughput_frames_per_second"] > 0
        return result, statistics

    reference, _ = collect(1)
    batched, statistics = collect(2)
    assert "2" in statistics["physical_batch_histogram"]
    for index, (prediction, metrics, details) in batched.items():
        expected, expected_metrics, expected_details = reference[index]
        torch.testing.assert_close(prediction, expected, rtol=3e-5, atol=3e-6)
        assert metrics == pytest.approx(expected_metrics, rel=3e-5, abs=3e-6)
        for key in ("nodes", "edges", "isolated_nodes", "max_degree", "isolate_ratio"):
            assert details[key] == expected_details[key]
    for index in (0, 3, 5):
        assert "temporal_l1" not in batched[index][1]
    for index in (1, 2, 4):
        assert "temporal_l1" in batched[index][1]


def test_vectorized_metrics_equal_individual_frame_reductions():
    generator = torch.Generator().manual_seed(43)
    prediction = torch.rand(4, 1, 17, 21, generator=generator)
    target = torch.rand(4, 1, 17, 21, generator=generator)
    expected = [frame_metrics(prediction[i:i + 1], target[i:i + 1]) for i in range(4)]
    actual = batch_frame_metrics(prediction, target)
    for left, right in zip(actual, expected, strict=True):
        assert left == pytest.approx(right, rel=2e-6, abs=2e-6)


def test_loader_missing_or_extra_frames_fail_explicitly():
    dataset = DiagnosticDataset()
    model = _model().eval()

    def run_forward(samples, contexts, timer):
        return model.forward_batch(samples, timing=timer)

    for loader_plan, declared in [([[0]], [[0], [1]]), ([[0], [1]], [[0]])]:
        loader = DataLoader(dataset, batch_sampler=loader_plan, collate_fn=pack_samples)
        with pytest.raises(RuntimeError, match="before|extra"):
            list(evaluation_frames(
                loader, declared, device=torch.device("cpu"), run_forward=run_forward,
                independent_sequences=True, statistics={}, timing_steps=1, timing_warmup=0,
            ))
