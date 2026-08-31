from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from scripts import bench
from tests.test_batch_preflight import _batch_config


def _dataset(tmp_path, *, streams=4, frames=8):
    config = _batch_config(tmp_path, batch_size=1, streams=streams, frames=frames)
    return config, bench.build_dataset(config["dataset"], split="train")


def _selection(dataset):
    return bench.select_windows(dataset, streams=4, warmup=2, frames=4, seed=17)


def _args(tmp_path):
    return Namespace(
        worker=False,
        config=str(tmp_path / "config.json"),
        output=str(tmp_path / "bench"),
        batches=[1, 4],
        backends=["torch"],
        chunks=None,
        streams=4,
        warmup_per_stream=2,
        frames_per_stream=4,
        repeats=1,
        threads=1,
        trace_steps=0,
    )


def test_same_real_frames_and_warmup_membership_across_batch_sizes(tmp_path):
    _, dataset = _dataset(tmp_path)
    try:
        selection = _selection(dataset)
        bench.validate_selection(dataset, selection)
        common = None
        for size in (1, 2, 4):
            subset, _sampler, schedule, phases = bench.batch_schedule(dataset, selection, size)
            grouped = {
                phase: sorted(
                    subset.indices[index]
                    for batch, label in zip(schedule, phases, strict=True)
                    if label == phase
                    for index in batch
                )
                for phase in ("warmup", "measure")
            }
            assert len(grouped["warmup"]) == 8
            assert len(grouped["measure"]) == 16
            assert common is None or grouped == common
            common = grouped
            assert all(len(batch) == size for batch in schedule)
    finally:
        dataset.close()


def test_window_plan_is_metadata_only_and_seeded(tmp_path, monkeypatch):
    _, dataset = _dataset(tmp_path)
    try:

        def forbidden(*args):
            pytest.fail("Selection decoded a sample")

        monkeypatch.setattr(type(dataset), "__getitem__", forbidden)
        selected = _selection(dataset)
        assert selected == _selection(dataset)
        assert selected["sha256"] == bench.digest(
            {key: value for key, value in selected.items() if key != "sha256"}
        )
        assert len({(e["source_file"], e["stream"]) for e in selected["entries"]}) == 4
    finally:
        dataset.close()


@pytest.mark.parametrize("change", ["hash", "identity", "phase"])
def test_selection_tampering_rejected(tmp_path, change):
    _, dataset = _dataset(tmp_path)
    try:
        selected = _selection(dataset)
        if change == "hash":
            selected["sha256"] = "0" * 64
        else:
            selected["entries"][0]["image_key" if change == "identity" else "phase"] = "forged"
            selected["sha256"] = bench.digest(
                {key: value for key, value in selected.items() if key != "sha256"}
            )
        with pytest.raises(ValueError, match="selection|identity"):
            bench.validate_selection(dataset, selected)
    finally:
        dataset.close()


def test_insufficient_real_windows_do_not_fabricate_or_repeat_frames(tmp_path):
    _, dataset = _dataset(tmp_path, streams=3, frames=4)
    try:
        with pytest.raises(ValueError, match="cannot form this comparison"):
            _selection(dataset)
    finally:
        dataset.close()


def test_mixed_or_partial_batches_are_rejected(tmp_path):
    _, dataset = _dataset(tmp_path)
    try:
        selected = _selection(dataset)
        with pytest.raises(ValueError, match="divide"):
            bench.batch_schedule(dataset, selected, 3)
        selected["entries"][0]["phase"] = "measure"
        with pytest.raises(ValueError, match="must never mix"):
            bench.batch_schedule(dataset, selected, 4)
    finally:
        dataset.close()


def test_real_cpu_training_loop_keeps_all_frames_and_temporal_context(tmp_path, monkeypatch):
    config, dataset = _dataset(tmp_path, streams=2, frames=4)
    config["model"]["recurrent"] = True
    selection = bench.select_windows(dataset, streams=2, warmup=1, frames=2, seed=0)
    config = bench.variant_config(config, 2, "torch", None)
    calls = []
    original = bench._step

    def observed(model, optimizer, scaler, criterion, samples, contexts, *args, **kwargs):
        calls.append((len(samples), sum(value[1] is not None for value in contexts)))
        return original(model, optimizer, scaler, criterion, samples, contexts, *args, **kwargs)

    monkeypatch.setattr(bench, "_step", observed)
    try:
        subset, sampler, schedule, phases = bench.batch_schedule(dataset, selection, 2)
        report = bench.exercise(config, subset, sampler, schedule, phases, torch.device("cpu"))
        assert calls == [(2, 0), (2, 2), (2, 2)]
        assert report["warmup_frames"] == 2
        assert report["measured_frames"] == 4
        assert report["optimizer_steps_measured"] == 2
        assert report["frames_per_second"] > 0
        assert report["host_decode_ms"] > 0
        assert report["peak_allocated_mib"] is None
        assert report["cuda_measured"] is False
        assert not (tmp_path / "train").exists()
    finally:
        dataset.close()


def test_backend_parity_checks_real_prediction_and_gradients(tmp_path):
    config, dataset = _dataset(tmp_path, streams=2, frames=4)
    try:
        reference = bench._numeric_snapshot(config, dataset, [0], torch.device("cpu"))
        candidate = bench._numeric_snapshot(config, dataset, [0], torch.device("cpu"))
        assert bench.compare_snapshots(reference, candidate, amp=False)["passed"]
        norm = candidate["gradient_norm"]
        candidate["gradient_norm"] = norm + 1
        with pytest.raises(ValueError, match="pre-clipping gradient"):
            bench.compare_snapshots(reference, candidate, amp=False)
        candidate["gradient_norm"] = norm
        candidate["tensors"]["prediction"] += 1
        with pytest.raises(ValueError, match="prediction"):
            bench.compare_snapshots(reference, candidate, amp=False)
    finally:
        dataset.close()


@pytest.mark.parametrize("edge_count", [0, 2])
def test_numerical_gate_cannot_validate_an_unexecuted_spline(tmp_path, monkeypatch, edge_count):
    config, dataset = _dataset(tmp_path, streams=2, frames=4)
    try:
        selection = bench.select_windows(dataset, streams=2, warmup=1, frames=2, seed=7)
        subset, _, schedule, _ = bench.batch_schedule(dataset, selection, 1)

        def snapshot(*args):
            return {
                "topology": [(2, edge_count)], "tensors": {}, "loss": {"total": 0.1},
                "gradient_norm": 0.0, "amp": {},
            }

        monkeypatch.setattr(bench, "_numeric_snapshot", snapshot)
        if edge_count:
            report = bench.numerical_gate(config, config, subset, schedule, torch.device("cpu"))
            assert report["nonempty_spline_exercised"] is True
            assert report["measurements"][0]["topology"] == [{"nodes": 2, "edges": 2}]
        else:
            with pytest.raises(ValueError, match="spline backend was not exercised"):
                bench.numerical_gate(config, config, subset, schedule, torch.device("cpu"))
    finally:
        dataset.close()


def test_cuda_only_refusal_creates_no_benchmark_or_train_output(tmp_path, monkeypatch):
    config, dataset = _dataset(tmp_path)
    dataset.close()
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never substitutes CPU"):
        bench.run(_args(tmp_path))
    assert not (tmp_path / "bench").exists()
    assert not (tmp_path / "train").exists()


@pytest.mark.parametrize("output", ["bench", "train", "train/nested"])
def test_existing_or_training_output_is_never_overwritten(tmp_path, monkeypatch, output):
    config, dataset = _dataset(tmp_path)
    dataset.close()
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = _args(tmp_path)
    args.output = str(tmp_path / output)
    (tmp_path / output).mkdir(parents=True)
    marker = tmp_path / output / "keep.txt"
    marker.write_text("untouched")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises((FileExistsError, ValueError)):
        bench.run(args)
    assert marker.read_text() == "untouched"


def test_source_stat_change_detected_without_touching_training_cache(tmp_path):
    _, dataset = _dataset(tmp_path)
    try:
        selection = _selection(dataset)
        before = bench._file_signatures(dataset, selection)
        assert before == bench._file_signatures(dataset, selection)
        assert not (tmp_path / "train" / ".data_hash_cache.json").exists()
    finally:
        dataset.close()


def test_redaction_removes_accounts_hostnames_and_configured_roots(tmp_path, monkeypatch):
    from tests.test_repo_hygiene import _private_unix_path, _private_windows_path

    monkeypatch.setattr(bench.socket, "gethostname", lambda: "private-gpu-host")
    text = f"private-gpu-host {_private_unix_path()} {_private_windows_path()}"
    cleaned = bench.redact(text, {}, tmp_path)
    assert "research-node-user" not in cleaned
    assert "private-gpu-host" not in cleaned


def test_variant_config_does_not_mutate_training_config(tmp_path):
    config, dataset = _dataset(tmp_path)
    dataset.close()
    before = copy.deepcopy(config)
    variant = bench.variant_config(config, 4, "triton", 128)
    assert config == before
    assert variant["model"]["spline_backend"] == "triton"
    assert variant["model"]["spline_chunk_size"] == 128
    assert variant["train"]["batch_size"] == 4


def test_all_requested_batch_sizes_keep_the_same_nonempty_measured_corpus(tmp_path):
    _, dataset = _dataset(tmp_path, streams=16, frames=4)
    try:
        selection = bench.select_windows(dataset, streams=16, warmup=1, frames=2, seed=11)
        expected = {
            entry["dataset_index"] for entry in selection["entries"] if entry["phase"] == "measure"
        }
        for size in (1, 4, 8, 16):
            subset, _, schedule, phases = bench.batch_schedule(dataset, selection, size)
            actual = [
                subset.indices[index]
                for batch, phase in zip(schedule, phases, strict=True)
                if phase == "measure"
                for index in batch
            ]
            assert len(actual) == len(set(actual)) == 32
            assert set(actual) == expected
    finally:
        dataset.close()


def test_trace_profiler_is_closed_when_training_raises(tmp_path, monkeypatch):
    config, dataset = _dataset(tmp_path, streams=1, frames=4)
    selection = bench.select_windows(dataset, streams=1, warmup=1, frames=2, seed=11)
    calls = []

    class Recorder:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args):
            calls.append("exit")

    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: Recorder())
    original = bench._step

    def fail_after_warmup(*args, **kwargs):
        if kwargs["step"] > 0:
            raise RuntimeError("training failed while tracing")
        return original(*args, **kwargs)

    monkeypatch.setattr(bench, "_step", fail_after_warmup)
    try:
        subset, sampler, schedule, phases = bench.batch_schedule(dataset, selection, 1)
        with pytest.raises(RuntimeError, match="training failed while tracing"):
            bench.exercise(
                config,
                subset,
                sampler,
                schedule,
                phases,
                torch.device("cpu"),
                trace_path=tmp_path / "trace.json",
                trace_steps=1,
            )
        assert calls == ["enter", "exit"]
    finally:
        dataset.close()


def test_oom_in_numerical_gate_cannot_be_reported_as_throughput_success(tmp_path, monkeypatch):
    config, dataset = _dataset(tmp_path)
    selection = _selection(dataset)
    signatures = bench._file_signatures(dataset, selection)
    dataset.close()
    output = tmp_path / "bench" / "trial.json"
    source = {"test": "fixed"}
    monkeypatch.setattr(bench, "_source_contract", lambda: source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(bench, "_runtime_provenance", lambda device: {})
    monkeypatch.setattr(bench, "_training_protocol", lambda config, device: {})

    def oom(*args):
        raise torch.cuda.OutOfMemoryError("test CUDA out of memory")

    monkeypatch.setattr(bench, "numerical_gate", oom)
    job = {
        "config": config,
        "output": str(output),
        "variant": {"batch_size": 1},
        "selection": selection,
        "source": source,
        "threads": 1,
        "reference_config": config,
        "file_signatures": signatures,
        "trace_steps": 0,
    }
    assert bench.worker(job) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["failure_category"] == "cuda_out_of_memory"
    assert "measurement" not in report
    assert not (tmp_path / "train").exists()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        bench.worker(job)
    assert output.read_bytes() == original


def test_aggregate_keeps_failed_repeat_visible_instead_of_declaring_success():
    variant = {"batch_size": 4, "backend": "torch", "spline_chunk_size": 64}
    rows = [
        {
            "variant": variant,
            "status": "passed",
            "measurement": {
                "frames_per_second": fps,
                "peak_allocated_mib": 10,
                "peak_reserved_mib": 20,
            },
        }
        for fps in (10, 20)
    ]
    rows.append({"variant": variant, "status": "failed", "measurement": None})
    summary = bench.aggregate(rows)[0]
    assert summary["frames_per_second_median"] == 15
    assert summary["passed_repeats"] == 2
    assert summary["requested_repeats"] == 3
    assert summary["all_repeats_passed"] is False


@pytest.mark.parametrize("location", ["data", "source", "other_run"])
def test_protected_data_source_and_nonconfigured_run_directories(tmp_path, location):
    config, dataset = _dataset(tmp_path)
    dataset.close()
    output = {
        "data": tmp_path / "hdr" / "bench",
        "source": bench.PROJECT / "src" / "bench",
        "other_run": bench.PROJECT / "runs" / "batch" / "benchmark",
    }[location]
    with pytest.raises(ValueError, match="separate|dedicated"):
        bench.protect_output(output.resolve(), config)


def test_backend_gate_rejects_equal_norm_small_element_sign_flip():
    gradient = torch.full((1_000_000,), 0.001)
    reference = {
        "tensors": {"gradient/weight": gradient},
        "topology": [(1, 0)],
        "gradient_norm": 1.0,
        "loss": {"total": 0.1},
    }
    candidate = {**reference, "tensors": {"gradient/weight": -gradient}}
    # Elementwise AMP tolerance alone accepts this entire reversed direction.
    assert torch.allclose(gradient, -gradient, atol=0.002, rtol=0.02)
    with pytest.raises(ValueError, match="gradient L2 direction"):
        bench.compare_snapshots(reference, candidate, amp=True)


@pytest.mark.parametrize("failure", [False, True])
def test_raw_profiler_metadata_is_never_published_and_temporary_is_removed(tmp_path, failure):
    from tests.test_repo_hygiene import _private_unix_path

    final = tmp_path / "public.trace.json"
    temporary = []

    class Recorder:
        def export_chrome_trace(self, path):
            temporary.append(Path(path))
            Path(path).write_text(
                "not-json"
                if failure
                else json.dumps(
                    {"traceName": _private_unix_path(), "traceEvents": []}
                )
            )

    if failure:
        with pytest.raises(json.JSONDecodeError):
            bench.export_trace(Recorder(), final, {})
        assert not final.exists()
    else:
        bench.export_trace(Recorder(), final, {})
        assert "research-node-user" not in final.read_text()
    assert temporary[0] != final
    assert not temporary[0].exists()
