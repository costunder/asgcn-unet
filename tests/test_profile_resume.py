from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
import torch

import asgcn_unet.preflight as profile
from asgcn_unet.data import build_dataset
from asgcn_unet.scan import ScanInUseError, ScanJournal, canonical_hash
from asgcn_unet.utils import save_json
from tests.fixtures import make_eventhdr
from tests.test_gpu_preflight import _config


def _record(index: int) -> dict:
    return {"dataset_index": index, "sample_id": f"sample-{index}"}


def test_scan_journal_commits_bounded_blocks_and_resumes(tmp_path) -> None:
    directory = tmp_path / "profile.scan"
    contract = {"config": 1}
    journal = ScanJournal(directory, contract, resume=False, block_size=2)
    journal.append(_record(0))
    assert json.loads((directory / "index.json").read_text())["samples_committed"] == 0
    journal.append(_record(1))
    first_bytes = (directory / "000000.json").read_bytes()
    journal.append(_record(2))
    journal.flush()
    journal.close()
    resumed = ScanJournal(directory, contract, resume=True, block_size=2)
    try:
        assert resumed.records == [_record(0), _record(1), _record(2)]
        resumed.append(_record(3))
        resumed.flush()
        assert (directory / "000000.json").read_bytes() == first_bytes
        assert resumed.committed == 4
    finally:
        resumed.close()


def test_scan_journal_rejects_concurrent_writer_and_releases_lock(tmp_path) -> None:
    directory = tmp_path / "profile.scan"
    first = ScanJournal(directory, {}, resume=False)
    try:
        with pytest.raises(ScanInUseError, match="Another process"):
            ScanJournal(directory, {}, resume=True)
    finally:
        first.close()
    second = ScanJournal(directory, {}, resume=True)
    second.close()


def test_scan_journal_lock_is_released_after_process_exits_without_cleanup(tmp_path) -> None:
    directory = tmp_path / "profile.scan"
    code = (
        "import os,sys; from pathlib import Path; "
        "from asgcn_unet.scan import ScanJournal; "
        "journal=ScanJournal(Path(sys.argv[1]),{},resume=False,block_size=1); "
        "journal.append({'dataset_index':0}); os._exit(0)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    reopened = ScanJournal(directory, {}, resume=True)
    try:
        assert reopened.records == [{"dataset_index": 0}]
    finally:
        reopened.close()


def test_scan_journal_ignores_uncommitted_orphan_block(tmp_path) -> None:
    directory = tmp_path / "profile.scan"
    first = ScanJournal(directory, {}, resume=False, block_size=1)
    first.append(_record(0))
    first.close()
    save_json(directory / "000001.json", [_record(999)])
    resumed = ScanJournal(directory, {}, resume=True, block_size=1)
    try:
        assert resumed.records == [_record(0)]
        resumed.append(_record(1))
    finally:
        resumed.close()
    final = ScanJournal(directory, {}, resume=True)
    try:
        assert final.records == [_record(0), _record(1)]
    finally:
        final.close()


@pytest.mark.parametrize("change", ["contract", "block_hash", "count", "path", "start"])
def test_scan_journal_rejects_mismatches_and_releases_failed_constructor_lock(
    tmp_path, change
) -> None:
    directory = tmp_path / "profile.scan"
    journal = ScanJournal(directory, {"data": "sealed"}, resume=False, block_size=1)
    journal.append(_record(0))
    journal.close()
    index_path = directory / "index.json"
    valid = json.loads(index_path.read_text())
    invalid = copy.deepcopy(valid)
    if change == "contract":
        invalid["contract"]["data"] = "changed"
    elif change == "block_hash":
        invalid["blocks"][0]["sha256"] = "0" * 64
    elif change == "count":
        invalid["samples_committed"] = 2
    elif change == "path":
        invalid["blocks"][0]["file"] = "../outside.json"
    else:
        invalid["blocks"][0]["start"] = 1
    save_json(index_path, invalid)
    with pytest.raises(ValueError, match="journal"):
        ScanJournal(directory, {"data": "sealed"}, resume=True)
    save_json(index_path, valid)
    reopened = ScanJournal(directory, {"data": "sealed"}, resume=True)
    reopened.close()


@pytest.fixture
def fast_profile(monkeypatch):
    calls: list[int] = []

    def measured(model, criterion, optimizer, scaler, raw, topology, config, device, step, **kw):
        calls.append(topology["dataset_index"])
        return {
            "dataset_index": topology["dataset_index"],
            "sample_id": topology["sample_id"],
            "nodes": topology["model_sampled_events"],
            "actual_directed_edges": topology["actual_directed_edges"],
            "loss": {"total": 1.0},
            "gradient_norm": 0.1,
            "step_time_ms": 1.0,
            "peak_allocated_mib": 100.0,
            "peak_reserved_mib": 120.0,
            "amp_enabled": False,
            "amp": {"scale_before": 1.0, "scale_after": 1.0, "retries": 0},
            "temporal_loss_applied": False,
            "temporal_context_sample_id": None,
        }

    monkeypatch.setattr(profile, "_gpu_step", measured)
    # Cache mechanics are independent of concurrent edits by other test workers.
    monkeypatch.setattr(
        profile,
        "_topology_implementation_contract",
        lambda device: {"implementation": "test-sealed", "device_type": device.type},
    )
    return calls


def _run(config, output, **kwargs):
    return profile.training_preflight(
        config,
        output,
        profile_samples=1,
        top_density_count=2,
        require_cuda=False,
        **kwargs,
    )


def test_interrupted_scan_commits_prefix_then_resumes_without_rescanning(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    output = tmp_path / "profile.json"
    original = profile._sample_topology
    scans = []
    interrupted = False

    def interrupt_once(sample, model_config, index):
        nonlocal interrupted
        scans.append(index)
        if index == 2 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original(sample, model_config, index)

    monkeypatch.setattr(profile, "_sample_topology", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        _run(config, output)
    failed = json.loads(output.read_text())
    assert failed["status"] == "interrupted"
    assert failed["scan_samples_committed"] == 2
    assert fast_profile == []
    report = _run(config, output, resume_scan=True)
    assert report["passed"] is True
    assert scans == [0, 1, 2, 2, 3]
    assert report["scan_provenance"]["reused_samples"] == 2
    assert report["scan_provenance"]["new_samples"] == 2


def test_gpu_probe_failure_retains_complete_scan_for_explicit_resume(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    output = tmp_path / "profile.json"
    measured = profile._gpu_step

    def failure(*args, **kwargs):
        raise FloatingPointError("Non-finite gradient in first sample")

    monkeypatch.setattr(profile, "_gpu_step", failure)
    failed = _run(config, output)
    assert failed["passed"] is False
    assert failed["topology"]["scan_complete"] is True
    assert failed["scan_samples_committed"] == 4
    assert "first sample" in failed["training_probe"]["failure"]["message"]
    monkeypatch.setattr(profile, "_gpu_step", measured)
    monkeypatch.setattr(
        profile,
        "_sample_topology",
        lambda *args: pytest.fail("Completed topology must not be scanned twice"),
    )
    resumed = _run(config, output, resume_scan=True)
    assert resumed["passed"] is True
    assert resumed["scan_provenance"]["new_samples"] == 0
    assert fast_profile[0] == 0


def test_concurrent_resume_refusal_preserves_existing_failure_report(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    output = tmp_path / "profile.json"

    def failure(*args, **kwargs):
        raise FloatingPointError("first sample backward failed")

    monkeypatch.setattr(profile, "_gpu_step", failure)
    _run(config, output)
    before = output.read_bytes()
    directory = output.with_suffix(".scan")
    index = json.loads((directory / "index.json").read_text())
    owner = ScanJournal(directory, index["contract"], resume=True)
    try:
        with pytest.raises(ScanInUseError, match="Another process"):
            _run(config, output, resume_scan=True)
        assert output.read_bytes() == before
    finally:
        owner.close()


def test_explicit_reuse_preserves_report_and_reruns_numerical_and_dense_probes(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    original_path = tmp_path / "original.json"
    original = _run(config, original_path)
    original_bytes = original_path.read_bytes()
    fast_profile.clear()
    monkeypatch.setattr(
        profile, "_sample_topology", lambda *args: pytest.fail("Explicit reuse must not rescan")
    )
    report = _run(config, tmp_path / "new.json", reuse_report=original_path)
    assert report["passed"] is True
    assert report["topology"] == original["topology"]
    assert report["scan_provenance"]["reused_samples"] == 4
    assert report["scan_provenance"]["origin"]["gpu_measurements_reused"] is False
    assert original_path.read_bytes() == original_bytes
    assert fast_profile[0] == 0
    assert len(fast_profile) == 1 + len(report["training_probe"]["numerical_probes"])
    with pytest.raises(ValueError, match="different output"):
        _run(config, original_path, reuse_report=original_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(config, tmp_path / "other.json", resume_scan=True, reuse_report=original_path)


def _legacy(report, path):
    legacy = copy.deepcopy(report)
    source_hash, commit = min(profile.LEGACY_TOPOLOGY_SOURCES)
    legacy["schema"] = "asgcn_training_preflight_v1"
    legacy["status"] = "passed"
    legacy["passed"] = legacy["report_eligible"] = True
    legacy["request"]["require_cuda"] = True
    legacy["checks"]["cuda_available"] = legacy["checks"]["cuda_oom_free"] = True
    legacy["source_provenance"] = {
        "source_tree_sha256": source_hash,
        "git_commit": commit,
        "git_source_dirty": False,
    }
    legacy["output"] = profile._artifact_path_label(path)
    legacy.pop("topology_contract", None)
    legacy.pop("scan_provenance", None)
    legacy["training_probe"].pop("numerical_selection", None)
    legacy["training_probe"].pop("numerical_probes", None)
    return legacy


def test_audited_legacy_report_migrates_only_topology_with_cpu_origin(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    generated = _run(config, tmp_path / "generated.json")
    path = tmp_path / "legacy.json"
    legacy = _legacy(generated, path)
    save_json(path, legacy)
    before = path.read_bytes()
    fast_profile.clear()
    monkeypatch.setattr(
        profile, "_sample_topology", lambda *args: pytest.fail("Legacy topology must be preserved")
    )
    migrated = _run(config, tmp_path / "migrated.json", reuse_report=path)
    assert migrated["passed"] is True
    assert migrated["schema"] == "asgcn_training_preflight_v2"
    assert migrated["scan_provenance"]["origin"]["record_device"] == "cpu"
    assert migrated["scan_provenance"]["new_samples"] == 0
    assert fast_profile[0] == 0
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["source", "dirty", "summary", "data", "config", "identity"])
def test_legacy_reuse_rejects_untrusted_or_inconsistent_inputs(
    tmp_path, fast_profile, change
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    generated = _run(config, tmp_path / "generated.json")
    path = tmp_path / "legacy.json"
    legacy = _legacy(generated, path)
    if change == "source":
        legacy["source_provenance"]["source_tree_sha256"] = "0" * 64
    elif change == "dirty":
        legacy["source_provenance"]["git_source_dirty"] = True
    elif change == "summary":
        legacy["topology"]["totals"]["raw_events"] += 1
    elif change == "data":
        legacy["data_provenance"]["content"]["sha256"] = "0" * 64
    elif change == "config":
        legacy["config_provenance"]["config"]["seed"] += 1
    else:
        legacy["topology"]["samples"][0]["scene"] = "wrong-source"
    save_json(path, legacy)
    fast_profile.clear()
    rejected = _run(config, tmp_path / "rejected.json", reuse_report=path)
    assert rejected["passed"] is False
    assert rejected["scan_samples_committed"] == 0
    assert fast_profile == []


def test_topology_transfer_uses_selected_cuda_and_never_silent_cpu_fallback() -> None:
    transfers = []

    class Events:
        device = torch.device("cuda:0")
        is_cuda = True

        def to(self, *, device, non_blocking):
            transfers.append((device, non_blocking))
            return self

    class Dataset:
        def get_topology_sample(self, index):
            assert index == 7
            return {"events": Events(), "sample_id": "test"}

        def __getitem__(self, index):
            pytest.fail("Topology scan must not request a target image")

    profile._scan_sample(Dataset(), 7, torch.device("cuda"))
    assert transfers == [(torch.device("cuda"), True)]
    Events.device = torch.device("cpu")
    Events.is_cuda = False
    with pytest.raises(RuntimeError, match="selected execution device"):
        profile._scan_sample(Dataset(), 7, torch.device("cuda"))


def test_numerical_selection_covers_first_empty_and_sparse_without_duplicates() -> None:
    records = [
        {
            "dataset_index": i,
            "sample_id": str(i),
            "model_sampled_events": n,
            "actual_directed_edges": e,
        }
        for i, n, e in ((0, 0, 0), (1, 8, 24), (2, 1, 0), (3, 0, 0))
    ]
    chosen = profile._numerical_selection(records)
    assert [entry["dataset_index"] for entry in chosen] == [0, 2]
    assert chosen[0]["reasons"] == ["first_chronological", "first_empty"]
    assert chosen[1]["reasons"] == ["sparsest_nonempty"]


def test_record_identity_must_match_real_dataset_even_if_summaries_are_self_consistent(
    tmp_path, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    report = _run(config, tmp_path / "profile.json")
    records = copy.deepcopy(report["topology"]["samples"])
    records[0]["sample_id"] = "invented-but-unique"
    dataset = build_dataset(config["dataset"], split="train")
    try:
        with pytest.raises(ValueError, match="identity differs"):
            profile._validate_topology_records(
                records, len(dataset), config["model"], dataset=dataset, complete=True
            )
    finally:
        dataset.close()


def test_verifier_rejects_jointly_rewritten_sample_identity_summary_and_probes(
    tmp_path, monkeypatch, fast_profile
) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    output = tmp_path / "profile.json"
    report = _run(config, output)
    report["status"] = "passed"
    report["passed"] = report["report_eligible"] = True
    report["request"]["require_cuda"] = True
    report["checks"]["cuda_available"] = report["checks"]["cuda_oom_free"] = True
    records = report["topology"]["samples"]
    records[0]["sample_id"] = "self-consistent-but-not-the-real-sample"
    report["topology"] = profile._topology_summary(
        records,
        dataset_size=len(records),
        data_max_events=config["dataset"]["max_events"],
        model_config=config["model"],
        top_density_count=2,
    )
    probe = report["training_probe"]
    for key in ("selected_samples", "steps", "numerical_selection", "numerical_probes"):
        for item in probe[key]:
            if item["dataset_index"] == 0:
                item["sample_id"] = records[0]["sample_id"]
    save_json(output, report)
    assert profile._require_verified_report_contract(report, output) is report
    monkeypatch.setattr(profile.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(profile, "resolve_device", lambda value: torch.device("cuda"))
    monkeypatch.setattr(profile, "_current_source_contract", lambda: report["source_provenance"])
    monkeypatch.setattr(profile, "_runtime_provenance", lambda device: report["runtime_provenance"])
    monkeypatch.setattr(
        profile, "_topology_implementation_contract", lambda device: report["topology_contract"]
    )
    monkeypatch.setattr(
        profile, "_training_protocol", lambda config, device: probe["training_protocol"]
    )
    with pytest.raises(ValueError, match="identity differs"):
        profile.verify_training_preflight(config, output)


@pytest.mark.parametrize(
    ("effective", "fresh", "amp", "mode"),
    [
        (True, True, {"scale_before": 65536.0, "scale_after": 32768.0, "retries": 1}, True),
        (False, True, {"scale_before": 1.0, "scale_after": 1.0, "retries": 0}, False),
        (True, False, {"scale_before": 128.0, "scale_after": 256.0, "retries": 0}, True),
    ],
)
def test_probe_amp_contract_accepts_valid_finite_scale_history(effective, fresh, amp, mode) -> None:
    profile._validate_probe_amp({"amp_enabled": mode, "amp": amp}, effective, fresh=fresh)


@pytest.mark.parametrize(
    ("effective", "fresh", "amp", "mode"),
    [
        (True, True, {"scale_before": 2.0, "scale_after": 2.0, "retries": 0}, True),
        (True, False, {"scale_before": 2.0, "scale_after": 0.0, "retries": 0}, True),
        (True, False, {"scale_before": float("inf"), "scale_after": 1.0, "retries": 1}, True),
        (True, False, {"scale_before": 2.0, "scale_after": 1.0, "retries": 17}, True),
        (True, False, {"scale_before": 2.0, "scale_after": 1.0, "retries": True}, True),
        (False, False, {"scale_before": 1.0, "scale_after": 1.0, "retries": 1}, False),
        (False, False, {"scale_before": 2.0, "scale_after": 2.0, "retries": 0}, False),
        (False, False, {"scale_before": 1.0, "scale_after": 1.0, "retries": 0}, True),
    ],
)
def test_probe_amp_contract_rejects_false_or_unbounded_diagnostics(
    effective, fresh, amp, mode
) -> None:
    with pytest.raises(ValueError, match="AMP|GradScaler"):
        profile._validate_probe_amp({"amp_enabled": mode, "amp": amp}, effective, fresh=fresh)


def test_successful_profile_cannot_be_overwritten_by_resume(tmp_path, fast_profile) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root)
    config = _config(root)
    output = tmp_path / "profile.json"
    _run(config, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="failed/interrupted"):
        _run(config, output, resume_scan=True)
    assert output.read_bytes() == original


def test_journal_contract_digest_is_canonical_and_does_not_accept_nonfinite_values() -> None:
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        canonical_hash({"value": float("nan")})
