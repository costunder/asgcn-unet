"""CPU-only recovery CLI tests; CUDA preflight is mocked, not measured."""

from __future__ import annotations

import copy
import importlib.util
import json
import shlex
import shutil
from pathlib import Path

import pytest

from asgcn_unet import resources, stream_preflight

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recovery(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "stream_recovery_test", PROJECT / "scripts" / "recover_streaming_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "synthetic checkout with spaces"
    baseline = root / "configs" / "ablations"
    baseline.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='synthetic-cpu-test'\n", encoding="utf-8")
    for kind in ("train", "hdr", "aid"):
        shutil.copyfile(PROJECT / "configs" / "ablations" / f"graph_unet-{kind}.json",
                        baseline / f"graph_unet-{kind}.json")
    report = stream_preflight.prepare_streaming_experiment(
        root, "runs/prepared", window_seconds=0.02, time_scale_seconds=0.1,
        hdr_timestamp_scale_to_seconds=1, aid_timestamp_scale_to_seconds=1,
        hdr_interval_timestamp_scale_to_seconds=1, aid_interval_timestamp_scale_to_seconds=1e-6,
    )
    monkeypatch.setattr(module, "PROJECT", root)
    monkeypatch.setattr(resources, "collect_runtime_resources",
                        lambda **kwargs: {"cpu": {"effective_cpu_limit": 2.5}})
    monkeypatch.setattr("torch.set_num_threads", lambda count: None)
    return module, root / report["output_root"]


def _arguments(root, *extra):
    return ["--experiment-root", str(root), "--use-measured-edge-guard",
            "--reserve-vram-mib", "1024", *extra]


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result(root):
    paths = sorted(root.glob("preflight-recovery-*/recovery.json"))
    assert len(paths) == 1
    return _read(paths[0])


def _mock_preflight(monkeypatch, module, *, passed=True, eligible=True, change=None):
    calls = []

    def probe(config, output, **kwargs):
        calls.append((copy.deepcopy(config), output, kwargs))
        saved = copy.deepcopy(config)
        saved["model"]["max_graph_edges"] = 3_000_000
        if change:
            change(saved, output, kwargs)
        if passed:
            module._write_new_json(Path(kwargs["measured_guard_config_output"]), saved)
        result = {
            "passed": passed, "report_eligible": eligible,
            "status": "passed" if passed else "failed",
            "topology": {"max_readout_directed_edges": 2_500_000,
                         "max_prefix_union_directed_edges_upper_bound": 3_000_000},
            "failure": None if passed else {
                "stage": "cuda_memory_budget", "type": "RuntimeError", "message": "Synthetic memory refusal"},
        }
        module._write_new_json(Path(output), result)
        return result

    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", probe)
    return calls


def test_success_derives_matching_configs_without_changing_inputs(recovery, monkeypatch, capsys):
    module, root = recovery
    originals = {path: path.read_bytes() for path in (root / "configs").glob("*.json")}
    calls = _mock_preflight(monkeypatch, module)
    assert module.main(_arguments(root)) == 0
    result = _result(root)
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert result["training_executed"] is False and result["calibration_executed"] is False
    assert result["request"]["cpu_threads"] == 2
    assert result["request"]["requested_cpu_threads"] == 4
    config, output, options = calls[0]
    assert options["require_cuda"] is True and options["reserve_vram_mib"] == 1024
    assert config["train"]["batch_size"] == 16 and config["train"]["epochs"] == 40
    assert config["model"]["graph_layers"] == 6 and config["model"]["hidden_dim"] == 64
    assert config["dataset"]["max_events"] is None
    assert Path(options["measured_guard_config_output"]).parent.parent == output.parent
    train = _read(result["configs"]["train"])
    assert train["output"]["run_dir"] == str(root / "train")
    assert Path(train["dataset"]["root"]).is_absolute()
    assert {entry["field"] for entry in result["changes"]["train"]} == {"model.max_graph_edges"}
    for kind in ("hdr", "aid"):
        derived = _read(result["configs"][kind])
        assert derived["model"] == train["model"]
        assert derived["eval"]["max_graph_edges_override"] is None
        assert Path(derived["eval"]["output_dir"]) == output.parent / "eval" / kind
    assert any("7475202" in warning for warning in result["warnings"])
    commands = [shlex.split(command) for command in result["next_commands"]]
    assert len(commands) == 2 and commands[0][4] == "train" and commands[1][4] == "calibrate"
    assert commands[0][6] == result["configs"]["train"]
    assert commands[0][8] == result["preflight_report"]
    assert commands[1][-2:] == ["--samples", "all"]
    text = capsys.readouterr().out
    assert text.count(result["next_commands"][0]) == 1
    assert text.count(result["next_commands"][1]) == 1
    assert '"topology"' not in text


def test_retry_always_creates_a_new_recovery_directory(recovery, monkeypatch):
    module, root = recovery
    _mock_preflight(monkeypatch, module, passed=False, eligible=False)
    assert module.main(_arguments(root)) == 1
    old = {path: path.read_bytes() for path in root.glob("preflight-recovery-*/*") if path.is_file()}
    assert module.main(_arguments(root)) == 1
    assert len(list(root.glob("preflight-recovery-*"))) == 2
    assert all(path.read_bytes() == content for path, content in old.items())


@pytest.mark.parametrize("passed,eligible", [(False, False), (True, False)])
def test_failed_or_cpu_smoke_never_provides_training_commands(recovery, monkeypatch, capsys, passed, eligible):
    module, root = recovery
    _mock_preflight(monkeypatch, module, passed=passed, eligible=eligible)
    assert module.main(_arguments(root)) == 1
    result = _result(root)
    assert result["next_commands"] == [] and not result["report_eligible"]
    assert not Path(result["configs"]["hdr"]).exists()
    text = capsys.readouterr()
    assert "prefix-union upper bound=3000000" in text.err
    assert "asgcn_unet.cli train" not in text.out


def test_nonempty_original_run_is_preserved_and_gets_no_fresh_commands(recovery, monkeypatch):
    module, root = recovery
    run = root / "train"
    run.mkdir()
    sentinel = run / "existing-user-result.txt"
    sentinel.write_text("original", encoding="utf-8")
    _mock_preflight(monkeypatch, module)
    assert module.main(_arguments(root)) == 0
    assert sentinel.read_text(encoding="utf-8") == "original"
    result = _result(root)
    assert result["next_commands"] == []
    assert any("nonempty" in warning for warning in result["warnings"])


def test_explicit_opt_in_is_required_before_probe_or_output(recovery, monkeypatch):
    module, root = recovery
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no probe"))
    with pytest.raises(SystemExit) as error:
        module.main(["--experiment-root", str(root), "--reserve-vram-mib", "1024"])
    assert error.value.code == 2
    assert not list(root.glob("preflight-recovery-*"))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_vram_reserve_must_be_explicit_finite_positive(recovery, value):
    module, root = recovery
    with pytest.raises(SystemExit):
        module.main(["--experiment-root", str(root), "--use-measured-edge-guard",
                     "--reserve-vram-mib", value])
    assert not list(root.glob("preflight-recovery-*"))


@pytest.mark.parametrize("change", ["other-output", "dataset-overlap", "model-mismatch"])
def test_unrelated_or_incompatible_config_refused_before_probe(recovery, monkeypatch, change):
    module, root = recovery
    path = root / "configs" / "aid.json"
    config = _read(path)
    if change == "other-output":
        config["eval"]["output_dir"] = "runs/some-other-study/eval"
    elif change == "dataset-overlap":
        config["dataset"]["root"] = str(root / "data")
    else:
        config["model"]["graph_radius"] = 0.07
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no probe"))
    assert module.main(_arguments(root)) == 1
    assert not list(root.glob("preflight-recovery-*"))


def test_evaluation_config_collision_does_not_overwrite_or_print_commands(recovery, monkeypatch):
    module, root = recovery
    sentinel = []

    def precreate(config, output, kwargs):
        path = Path(kwargs["measured_guard_config_output"]).with_name("hdr.json")
        module._write_new_json(path, {"existing": "preserve"})
        sentinel.append(path)

    _mock_preflight(monkeypatch, module, change=precreate)
    assert module.main(_arguments(root)) == 1
    assert _read(sentinel[0]) == {"existing": "preserve"}
    assert _result(root)["next_commands"] == []


def test_unexpected_probe_config_change_is_rejected(recovery, monkeypatch):
    module, root = recovery
    _mock_preflight(monkeypatch, module,
                    change=lambda config, *_: config["model"].update(graph_radius=0.07))
    assert module.main(_arguments(root)) == 1
    assert _result(root)["next_commands"] == []


def test_existing_json_is_never_overwritten(recovery, tmp_path):
    module, _ = recovery
    path = tmp_path / "existing.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._write_new_json(path, {"replacement": True})
    assert path.read_text(encoding="utf-8") == "original"


def test_unbounded_input_guard_becomes_a_finite_measured_guard(recovery, monkeypatch):
    module, root = recovery
    for path in (root / "configs").glob("*.json"):
        config = _read(path)
        config["model"]["max_graph_edges"] = None
        path.write_text(json.dumps(config), encoding="utf-8")
    _mock_preflight(monkeypatch, module)
    assert module.main(_arguments(root)) == 0
    for path in _result(root)["configs"].values():
        assert _read(path)["model"]["max_graph_edges"] == 3_000_000


@pytest.mark.parametrize("error,code", [(RuntimeError("synthetic probe failure"), 1),
                                       (KeyboardInterrupt(), 130)])
def test_raised_failure_recovers_saved_count_maxima(recovery, monkeypatch, capsys, error, code):
    module, root = recovery

    def interrupt(config, output, **kwargs):
        module._write_new_json(Path(output), {
            "topology": {"max_readout_directed_edges": 2_500_000,
                         "max_prefix_union_directed_edges_upper_bound": 3_000_000},
            "failure": {"stage": "stateful_training_probe"},
        })
        raise error

    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", interrupt)
    assert module.main(_arguments(root)) == code
    result = _result(root)
    assert result["failure"]["max_prefix_union_directed_edges_upper_bound"] == 3_000_000
    assert result["failure"]["stage"] == "stateful_training_probe"
    assert result["next_commands"] == []
    assert "prefix-union upper bound=3000000" in capsys.readouterr().err


def _saved_recovery(module, root, *, readout_edges=424096036, union_edges=440677422):
    """Synthetic saved metadata only; no event dataset or GPU is used."""
    from asgcn_unet.engine import _artifact_path_label, _public_config

    prior = root / "preflight-recovery-prior"
    prior.mkdir()
    configs, sources = module._load_experiment_configs(root)
    public = _public_config(configs["train"])
    source = {"source_tree_sha256": "a" * 64, "git_commit": "b" * 40, "git_source_dirty": False}
    row = {"dataset_index": 0, "readout_nodes": 25000, "readout_directed_edges": readout_edges,
           "prefix_union_nodes_upper_bound": 26000, "prefix_union_directed_edges_upper_bound": union_edges}
    topology = {
        "samples": [row, None], "scanned_samples": 1, "dataset_samples": 2, "scan_complete": False,
        "arrival_prefix_peak_measured": False,
        "prefix_bound_kind": "previous_live_window_union_all_current_frame_arrivals",
        **{f"max_{key}": value for key, value in row.items() if key != "dataset_index"},
    }
    profile_path = prior / "stream-profile.json"
    profile = {
        "schema": "asgcn_streaming_training_preflight_v1", "synthetic_cpu_test_only": True,
        "status": "interrupted", "stage": "count_only_topology", "passed": False, "report_eligible": False,
        "output": _artifact_path_label(profile_path), "source_provenance": source,
        "input_config_provenance": {"config": public, "sha256": stream_preflight._digest(public)},
        "config_provenance": {"config": copy.deepcopy(public), "sha256": stream_preflight._digest(public)},
        "data_provenance": {"content": {"sha256": "c" * 64}}, "topology": topology,
        "runtime_provenance": {"gpu": {"name": "synthetic MIG", "total_memory_mib": 9728.0}},
    }
    profile["commitment_sha256"] = stream_preflight._digest(profile)
    module._write_new_json(profile_path, profile)
    metadata = {
        "schema": "asgcn_streaming_preflight_recovery_v1", "experiment_root": str(root),
        "output_root": str(prior), "preflight_report": str(profile_path), "source_configs": sources,
        "configs": {kind: str(prior / "configs" / f"{kind}.json") for kind in ("train", "hdr", "aid")},
        "status": "interrupted", "report_eligible": False,
        "training_executed": False, "calibration_executed": False,
    }
    module._write_new_json(prior / "recovery.json", metadata)
    return prior, source


def _resumed_result(root, prior):
    paths = [path for path in root.glob("preflight-recovery-*/recovery.json") if path.parent != prior]
    assert len(paths) == 1
    return _read(paths[0])


def _change_saved_profile(prior, change, *, reseal=True):
    path = prior / "stream-profile.json"
    value = _read(path)
    value.pop("commitment_sha256", None)
    change(value)
    if reseal:
        value["commitment_sha256"] = stream_preflight._digest(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_resume_old_partial_detects_impossible_saved_allocation_without_cuda_or_rescan(recovery, monkeypatch, capsys):
    module, root = recovery
    prior, source = _saved_recovery(module, root)
    original = {path: path.read_bytes() for path in prior.iterdir()}
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract",
                        lambda: {**source, "source_tree_sha256": "d" * 64})
    monkeypatch.setattr("torch.cuda.is_available", lambda: pytest.fail("evidence inspection must not query CUDA"))
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no rescan"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    evidence = result["resume_evidence"]
    assert evidence["source_matches_current"] is False
    assert evidence["current_data_revalidated"] is False
    assert evidence["current_device_revalidated"] is False
    assert evidence["report_eligible"] is False and evidence["exact_resume_performed"] is False
    assert evidence["infeasible_on_saved_device"] is True
    assert evidence["saved_device_budget_after_requested_reserve_mib"] == 8704
    assert evidence["single_readout_storage_floor"]["graph_and_basis_bytes"] == 424096036 * 56 + 25000 * 72
    assert not Path(result["preflight_report"]).exists() and result["next_commands"] == []
    assert all(path.read_bytes() == content for path, content in original.items())
    core = dict(result)
    assert core.pop("commitment_sha256") == stream_preflight._digest(core)
    output = capsys.readouterr()
    assert "recorded allocation allows at most 8,704.00 MiB" in output.out
    assert "no committed raw scanner checkpoint" in output.err


def test_large_conservative_union_is_not_used_as_actual_graph_memory(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root, readout_edges=6)
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no raw state"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    assert result["resume_evidence"]["infeasible_on_saved_device"] is False
    assert result["resume_evidence"]["single_readout_storage_floor"]["graph_and_basis_bytes"] == 6 * 56 + 25000 * 72
    assert "no committed raw scanner checkpoint" in result["failure"]["message"]


@pytest.mark.parametrize("corruption", ["commitment", "cached-count", "source-contract", "model", "data-hash", "eligible"])
def test_resume_rejects_corrupt_or_incompatible_saved_evidence_before_probe(recovery, monkeypatch, corruption):
    module, root = recovery
    prior, source = _saved_recovery(module, root)
    def change(profile):
        if corruption == "cached-count":
            profile["topology"]["max_readout_directed_edges"] += 2
        elif corruption == "source-contract":
            profile["source_provenance"]["source_tree_sha256"] = "invalid"
        elif corruption == "model":
            value = profile["config_provenance"]["config"]
            value["model"]["graph_radius"] = 0.07
            profile["config_provenance"]["sha256"] = stream_preflight._digest(value)
        elif corruption == "data-hash":
            profile["data_provenance"]["content"]["sha256"] = "invalid"
        elif corruption == "eligible":
            profile["report_eligible"] = True
    _change_saved_profile(prior, change, reseal=corruption != "commitment")
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no unsafe probe"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    assert result["failure"]["stage"] == "inspect_saved_recovery_evidence"
    assert result["next_commands"] == []


def test_resume_rejects_changed_raw_configuration_bytes(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root)
    path = root / "configs" / "train.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("changed identity"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    assert "raw configuration" in _resumed_result(root, prior)["failure"]["message"]


def test_resume_checkpoint_candidate_is_forwarded_only_after_integrity_inspection(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root, readout_edges=6, union_edges=8)
    module._write_new_json(prior / "scan-checkpoint" / "latest.json", {"synthetic": True})
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    inspected = []
    monkeypatch.setattr(module, "_inspect_resume_checkpoint", lambda path, resource: inspected.append(path))
    calls = _mock_preflight(monkeypatch, module, passed=False, eligible=False)
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    assert inspected == [prior / "scan-checkpoint"]
    assert calls[0][2]["resume_checkpoint"] == prior / "scan-checkpoint"
    assert calls[0][2]["scan_checkpoint_dir"] == Path(result["output_root"]) / "scan-checkpoint"
    assert result["resume_evidence"]["checkpoint_integrity_verified"] is True
    assert result["resume_evidence"]["exact_resume_performed"] is False
    assert not result["report_eligible"] and not result["next_commands"]


def test_checkpoint_source_migration_is_not_inferred_from_matching_config(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root)
    module._write_new_json(prior / "scan-checkpoint" / "latest.json", {"synthetic": True})
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract",
                        lambda: {**source, "source_tree_sha256": "d" * 64})
    monkeypatch.setattr(module, "_inspect_resume_checkpoint", lambda *a: pytest.fail("no cross-source restore"))
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no rescan"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    assert "source differs" in _resumed_result(root, prior)["failure"]["message"]


def test_saved_derived_configuration_content_must_match_its_report(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root)
    configs, _ = module._load_experiment_configs(root)
    changed = copy.deepcopy(configs["train"])
    changed["model"]["graph_radius"] = 0.07
    module._write_new_json(prior / "configs" / "train.json", changed)
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no changed config"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    assert "config file differs" in _resumed_result(root, prior)["failure"]["message"]


def test_identical_source_bytes_do_not_require_git_metadata_migration(recovery, monkeypatch):
    module, root = recovery
    prior, source = _saved_recovery(module, root, readout_edges=6, union_edges=8)
    module._write_new_json(prior / "scan-checkpoint" / "latest.json", {"synthetic": True})
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract",
                        lambda: {**source, "git_commit": "f" * 40})
    monkeypatch.setattr(module, "_inspect_resume_checkpoint", lambda *a: None)
    calls = _mock_preflight(monkeypatch, module, passed=False, eligible=False)
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    assert result["resume_evidence"]["source_matches_current"] is True
    assert result["resume_evidence"]["source_metadata_matches_current"] is False
    assert len(calls) == 1


def test_committed_running_metadata_exists_before_preflight(recovery, monkeypatch):
    module, root = recovery

    def inspect_startup(config, output, kwargs):
        metadata = _read(Path(output).parent / "recovery.json")
        core = dict(metadata)
        assert core.pop("commitment_sha256") == stream_preflight._digest(core)
        assert metadata["status"] == "running" and metadata["report_eligible"] is False
        assert metadata["next_commands"] == []

    _mock_preflight(monkeypatch, module, change=inspect_startup)
    assert module.main(_arguments(root)) == 0
    final = _result(root)
    core = dict(final)
    assert core.pop("commitment_sha256") == stream_preflight._digest(core)
    assert final["status"] == "passed"


def test_external_change_to_owned_metadata_is_not_overwritten(recovery, monkeypatch, capsys):
    module, root = recovery
    sentinel = []

    def external_change(config, output, kwargs):
        path = Path(output).parent / "recovery.json"
        path.write_text('{"external": "preserve"}', encoding="utf-8")
        sentinel.append(path)

    _mock_preflight(monkeypatch, module, change=external_change)
    assert module.main(_arguments(root)) == 1
    assert _read(sentinel[0]) == {"external": "preserve"}
    assert "asgcn_unet.cli train" not in capsys.readouterr().out


@pytest.mark.parametrize("sealed", [True, False])
def test_running_recovery_requires_committed_metadata_before_checkpoint_resume(recovery, monkeypatch, sealed):
    module, root = recovery
    prior, source = _saved_recovery(module, root, readout_edges=6, union_edges=8)
    path = prior / "recovery.json"
    metadata = _read(path)
    metadata["status"] = "running"
    if sealed:
        metadata["commitment_sha256"] = stream_preflight._digest(metadata)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _change_saved_profile(prior, lambda value: value.update(status="running"))
    module._write_new_json(prior / "scan-checkpoint" / "latest.json", {"synthetic": True})
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(module, "_inspect_resume_checkpoint", lambda *args: None)
    calls = _mock_preflight(monkeypatch, module, passed=False, eligible=False)
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    result = _resumed_result(root, prior)
    assert len(calls) == int(sealed)
    assert result["report_eligible"] is False and result["next_commands"] == []
    if not sealed:
        assert "Running recovery metadata requires" in result["failure"]["message"]


def test_implicit_saved_graph_never_uses_materialized_edge_storage_floor(recovery, monkeypatch, capsys):
    module, root = recovery
    for path in (root / "configs").glob("*.json"):
        config = _read(path)
        config["model"]["graph_storage"] = "implicit_radius"
        path.write_text(json.dumps(config), encoding="utf-8")
    prior, source = _saved_recovery(module, root)
    monkeypatch.setattr("asgcn_unet.engine._current_source_contract", lambda: source)
    monkeypatch.setattr(stream_preflight, "streaming_training_preflight", lambda *a, **k: pytest.fail("no raw state"))
    assert module.main(_arguments(root, "--resume-from", str(prior))) == 1
    evidence = _resumed_result(root, prior)["resume_evidence"]
    floor = evidence["single_readout_storage_floor"]
    assert floor["graph_storage"] == "implicit_radius" and floor["edge_storage_materialized"] is False
    assert floor["graph_and_basis_bytes"] == 25000 * 72
    assert evidence["infeasible_on_saved_device"] is False
    output = capsys.readouterr().out
    assert "excluding index/scratch/training" in output and "cannot fit" not in output
