"""Synthetic CPU/read-only profile checks; no production training or GPU runs."""

import builtins
import copy
import importlib.util
import math
import os
from collections import Counter
from pathlib import Path

import pytest

from asgcn_unet.ablation_comparison import render_bcd_t4
from asgcn_unet.ablation_suite import collect_summary

PROJECT = Path(__file__).resolve().parents[1]
PROFILE = "bcd-throughput-t4"


def _cli():
    spec = importlib.util.spec_from_file_location("synthetic_bcd_cli", PROJECT / "scripts/run_ablations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return collect_summary(PROJECT, profile=PROFILE)


def _forbidden(*args, **kwargs):
    raise AssertionError("Read-only test started a subprocess, checkpoint, or GPU operation")


def test_plan_selects_only_bcd_and_t4_with_no_execution(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli.subprocess, "run", _forbidden)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-assigned-device")
    before_env = dict(os.environ)
    configs = list((PROJECT / "configs/ablations").glob("*.json"))
    before_configs = {path: path.read_bytes() for path in configs}
    assert cli.main(["--profile", PROFILE]) == 0
    output = capsys.readouterr().out
    assert "PLAN ONLY" in output
    assert "train=2, calibrate=2, evaluate=12, benchmark=12" in output
    assert "T=8/16/32 are deferred" in output
    assert "A/E are not scheduled" in output
    assert "single-frame latency" in output
    commands = [line for line in output.splitlines() if line.startswith("[")]
    assert not any("[unet:" in line or "[transformer:" in line for line in commands)
    for line in commands:
        if "--simulation-steps" in line:
            assert "--simulation-steps 4" in line
    assert before_env == dict(os.environ)
    assert before_configs == {path: path.read_bytes() for path in configs}


def test_default_full_plan_keeps_all_families_and_sweep(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli.subprocess, "run", _forbidden)
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "Execution profile: full" in output
    assert "train=4, calibrate=2, evaluate=40, benchmark=40" in output
    assert "[unet:train]" in output and "[transformer:train]" in output
    assert "--simulation-steps 32" in output


@pytest.mark.parametrize("family", ["unet", "transformer"])
def test_throughput_profile_refuses_a_e_before_execution(monkeypatch, family, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "execute_commands", _forbidden)
    assert cli.main(["--profile", PROFILE, "--families", family, "--stage", "all", "--execute"]) == 1
    assert "not permitted" in capsys.readouterr().err


@pytest.mark.parametrize("details", [False, True])
def test_throughput_summary_is_read_only_and_not_an_empty_ae_table(monkeypatch, capsys, details):
    cli = _cli()
    rows = _rows()
    calls = []

    def collect(project, families, *, profile):
        calls.append((families, profile))
        return rows

    monkeypatch.setattr(cli, "collect_summary", collect)
    monkeypatch.setattr(cli, "plan_commands", _forbidden)
    monkeypatch.setattr(cli, "execute_commands", _forbidden)
    monkeypatch.setattr(cli.subprocess, "run", _forbidden)
    assert cli.main(["--profile", PROFILE, "--stage", "summary", *(["--details"] if details else [])]) == 0
    output = capsys.readouterr().out
    assert calls == [(("pointwise_unet", "graph_unet"), PROFILE)]
    assert "## B/C/D throughput-first profile: T=4" in output
    assert "Primary comparison: A" not in output
    assert ("## Selected experiments and stored provenance" in output) is details
    assert "not completion of the full sweep" in output
    assert "missing quality; missing benchmark" in output


def test_explicit_execute_dispatches_two_shared_trainings_and_resume(monkeypatch, capsys):
    cli = _cli()
    calls = []
    rows = _rows()

    def execute(commands, project, **kwargs):
        calls.append((commands, kwargs))

    monkeypatch.setattr(cli, "execute_commands", execute)
    monkeypatch.setattr(cli, "collect_summary", lambda *args, **kwargs: rows)
    monkeypatch.setattr(cli.subprocess, "run", _forbidden)
    assert cli.main(["--profile", PROFILE, "--stage", "all", "--execute", "--resume"]) == 0
    assert len(calls) == 1
    commands, options = calls[0]
    counts = Counter(command.stage for command in commands)
    assert counts["train"] == counts["calibrate"] == 2
    assert counts["evaluate"] == counts["benchmark"] == 12
    assert options["execute"] is options["resume"] is True
    assert "## B/C/D throughput-first" in capsys.readouterr().out


def test_plan_never_accepts_execute(monkeypatch, capsys):
    cli = _cli()
    monkeypatch.setattr(cli, "execute_commands", _forbidden)
    assert cli.main(["--profile", PROFILE, "--stage", "plan", "--execute"]) == 1
    assert "plan never executes" in capsys.readouterr().err


def test_render_separates_compute_benchmark_and_measured_batch_throughput():
    rows = _rows()
    rows[0].update(fps=17.25, eval_compute_fps=92.5, eval_end_to_end_fps=8.5, eval_batch_size=16, eval_num_workers=4)
    before = copy.deepcopy(rows)
    output = render_bcd_t4(rows)
    assert "17.25000" in output and "92.50000" in output and "8.50000" in output
    assert "Batch compute frames/s" in output and "End-to-end frames/s" in output
    assert "ceiling" in output
    assert rows == before


@pytest.mark.parametrize("field", ["fps", "micro_psnr", "vram_mib", "eval_compute_fps", "eval_end_to_end_fps"])
@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_nonfinite_or_unrecorded_metrics_are_unavailable(field, value):
    rows = _rows()
    rows[0][field] = value
    output = render_bcd_t4(rows)
    assert "| nan |" not in output.lower() and "| inf |" not in output.lower()
    assert "N/A" in output


@pytest.mark.parametrize("replacement", [
    {"family": "unet", "group": "A", "mode": "ann"},
    {"family": "graph_transformer", "group": "E", "mode": "ann"},
    {"mode": "snn_literal_eq15_T32"},
    {"dataset": "unrecognized"},
])
def test_render_refuses_wrong_family_mode_or_dataset(replacement):
    rows = _rows()
    rows[0].update(replacement)
    with pytest.raises(ValueError, match="Unexpected"):
        render_bcd_t4(rows)


def test_render_refuses_duplicate_results():
    rows = _rows()
    with pytest.raises(ValueError, match="Duplicate"):
        render_bcd_t4(rows + [rows[0]])


def test_report_render_does_not_import_torch_or_engine(monkeypatch):
    rows = _rows()
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy"} or "engine" in name:
            raise AssertionError(f"Read-only summary imported heavy runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert "B/C/D" in render_bcd_t4(rows)
    assert "No selected results available" in render_bcd_t4([])
