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
