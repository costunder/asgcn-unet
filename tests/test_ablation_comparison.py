"""Synthetic A/E comparison fixtures only; never real quality/performance evidence."""

from __future__ import annotations

import builtins
import copy
import importlib.util
from pathlib import Path

import pytest

from asgcn_unet.ablation_comparison import compare_a_e, render_a_e

PROJECT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _row(family="unet", dataset="hdr", **overrides):
    """Arbitrary small values, explicitly not a measured trained model."""
    row = {
        "family": family,
        "group": "A" if family == "unet" else "E",
        "dataset": dataset,
        "mode": "ann",
        "run": f"synthetic-only/{family}/{dataset}/ann",
        "frames": 10,
        "parameters": 100,
        "micro_psnr": 10.0,
        "micro_ssim": 0.5,
        "macro_psnr": 12.0,
        "macro_ssim": 0.6,
        "mean_ms": 20.0,
        "fps": 50.0,
        "vram_mib": 80.0,
        "training_settings_match": True,
        "quality_eligible": True,
        "benchmark_eligible": True,
        "quality_model_contract_valid": True,
        "benchmark_model_contract_valid": True,
        "quality_mode_valid": True,
        "benchmark_mode_valid": True,
        "benchmark_contracts_verified": True,
        "benchmark_checkpoint_match": True,
        "io_excluded": True,
        "quality_dataset_hash_verified": False,
    }
    for field in (
        "quality_dataset_claimed_sha256",
        "quality_runtime_sha256",
        "quality_source_sha256",
        "quality_precision_sha256",
        "benchmark_evaluation_dataset_sha256",
        "benchmark_runtime_sha256",
        "benchmark_source_sha256",
        "benchmark_precision_sha256",
    ):
        row[field] = DIGEST
    row.update(overrides)
    return row


def _pair(**e_overrides):
    return [_row(), _row("transformer", **e_overrides)]


def _measurements(pair):
    return {row["metric"]: row for row in pair["measurements"]}


def test_raw_metrics_delta_e_minus_a_and_ratio_e_divided_by_a():
    pair = compare_a_e(
        _pair(
            parameters=150,
            micro_psnr=13.0,
            micro_ssim=0.7,
            macro_psnr=11.0,
            macro_ssim=0.55,
            mean_ms=30.0,
            fps=25.0,
            vram_mib=40.0,
        )
    )[0]
    values = _measurements(pair)
    assert pair["quality_issues"] == pair["benchmark_issues"] == []
    assert values["PSNR-micro (dB)"] == {
        "metric": "PSNR-micro (dB)",
        "a": 10.0,
        "e": 13.0,
        "delta": 3.0,
        "ratio": None,
    }
    assert values["SSIM-micro"]["delta"] == pytest.approx(0.2)
    assert values["PSNR-macro (dB)"]["delta"] == -1.0
    assert values["Parameters"]["ratio"] == 1.5
    assert values["Latency (ms)"]["delta"] == 10.0
    assert values["Latency (ms)"]["ratio"] == 1.5
    assert values["FPS"]["delta"] == -25.0
    assert values["FPS"]["ratio"] == 0.5
    assert values["VRAM (MiB)"]["ratio"] == 0.5


def test_legacy_graph_transformer_cannot_be_relabelled_as_E():
    rows = [_row(), _row("graph_transformer", group="E", micro_psnr=999.0)]
    pair = compare_a_e(rows)[0]
    assert pair["e_run"] is None
    assert _measurements(pair)["PSNR-micro (dB)"]["e"] is None
    assert "A or E result missing" in pair["quality_issues"]
    assert "graph_transformer results are not E" in render_a_e(rows)


@pytest.mark.parametrize(
    "override", [{"group": "E-ANN-control"}, {"mode": "snn_literal_eq15_T4"}, {"group": "D"}]
)
def test_wrong_group_or_snn_mode_cannot_enter_primary_E_pair(override):
    pair = compare_a_e([_row(), _row("transformer", **override)])[0]
    assert pair["e_run"] is None
    assert _measurements(pair)["FPS"]["delta"] is None


def test_datasets_are_paired_separately():
    rows = _pair(micro_psnr=13.0)
    rows += [
        _row(dataset="aid", micro_psnr=20.0),
        _row("transformer", dataset="aid", micro_psnr=25.0),
    ]
    result = {row["dataset"]: _measurements(row) for row in compare_a_e(rows)}
    assert result["hdr"]["PSNR-micro (dB)"]["delta"] == 3.0
    assert result["aid"]["PSNR-micro (dB)"]["delta"] == 5.0


def test_other_families_hardware_and_global_mismatch_flags_do_not_pollute_pair():
    rows = _pair(mean_ms=10.0)
    for row in rows:
        row["benchmark_runtime_matched"] = False
        row["quality_runtime_matched"] = False
    rows += [
        _row("pointwise_unet", group="B", benchmark_runtime_sha256=OTHER_DIGEST),
        _row("graph_unet", group="D", quality_runtime_sha256=OTHER_DIGEST),
    ]
    pair = compare_a_e(rows)[0]
    assert pair["quality_issues"] == pair["benchmark_issues"] == []
    assert _measurements(pair)["Latency (ms)"]["ratio"] == 0.5


def test_ssim_zero_baseline_allows_valid_absolute_difference():
    rows = [_row(micro_ssim=0.0), _row("transformer", micro_ssim=0.2)]
    metric = _measurements(compare_a_e(rows)[0])["SSIM-micro"]
    assert metric["a"] == 0.0
    assert metric["delta"] == 0.2
    assert metric["ratio"] is None


@pytest.mark.parametrize(
    "field,label",
    [
        ("mean_ms", "Latency (ms)"),
        ("fps", "FPS"),
        ("vram_mib", "VRAM (MiB)"),
        ("parameters", "Parameters"),
    ],
)
def test_zero_denominator_never_divides_by_zero(field, label):
    rows = [_row(**{field: 0.0}), _row("transformer", **{field: 3.0})]
    metric = _measurements(compare_a_e(rows)[0])[label]
    assert metric["ratio"] is None
    assert metric["delta"] == 3.0


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), float("-inf"), True, "12.0"])
def test_nonfinite_missing_or_nonnumeric_values_do_not_become_zero(missing):
    rows = _pair(micro_psnr=missing, mean_ms=missing)
    metrics = _measurements(compare_a_e(rows)[0])
    for label in ("PSNR-micro (dB)", "Latency (ms)"):
        assert metrics[label]["e"] is None
        assert metrics[label]["delta"] is None
        assert metrics[label]["ratio"] is None
    assert "N/A" in render_a_e(rows)


@pytest.mark.parametrize(
    "field",
    [
        "quality_runtime_sha256",
        "quality_dataset_claimed_sha256",
        "quality_source_sha256",
        "quality_precision_sha256",
    ],
)
def test_quality_provenance_failure_does_not_hide_valid_benchmark_delta(field):
    pair = compare_a_e(_pair(**{field: OTHER_DIGEST, "micro_psnr": 15.0, "mean_ms": 10.0}))[0]
    values = _measurements(pair)
    assert values["PSNR-micro (dB)"]["a"] == 10.0
    assert values["PSNR-micro (dB)"]["e"] == 15.0
    assert values["PSNR-micro (dB)"]["delta"] is None
    assert values["Latency (ms)"]["delta"] == -10.0
    assert pair["quality_issues"] and not pair["benchmark_issues"]


@pytest.mark.parametrize(
    "field",
    [
        "benchmark_runtime_sha256",
        "benchmark_evaluation_dataset_sha256",
        "benchmark_source_sha256",
        "benchmark_precision_sha256",
    ],
)
def test_benchmark_provenance_failure_does_not_hide_valid_quality_delta(field):
    pair = compare_a_e(_pair(**{field: OTHER_DIGEST, "micro_psnr": 15.0, "mean_ms": 10.0}))[0]
    values = _measurements(pair)
    assert values["PSNR-micro (dB)"]["delta"] == 5.0
    assert values["Latency (ms)"]["e"] == 10.0
    assert values["Latency (ms)"]["delta"] is None
    assert not pair["quality_issues"] and pair["benchmark_issues"]


@pytest.mark.parametrize("kind", ["quality", "benchmark"])
def test_false_eligibility_blocks_only_its_comparison_columns(kind):
    pair = compare_a_e(_pair(**{f"{kind}_eligible": False, "micro_psnr": 15.0, "mean_ms": 10.0}))[0]
    values = _measurements(pair)
    assert values["PSNR-micro (dB)"]["delta"] == (None if kind == "quality" else 5.0)
    assert values["Latency (ms)"]["delta"] == (None if kind == "benchmark" else -10.0)


@pytest.mark.parametrize(
    "flag",
    [
        "benchmark_model_contract_valid",
        "benchmark_mode_valid",
        "benchmark_contracts_verified",
        "benchmark_checkpoint_match",
        "io_excluded",
    ],
)
def test_failed_benchmark_validation_flags_block_ratios(flag):
    metric = _measurements(compare_a_e(_pair(**{flag: False}))[0])["FPS"]
    assert metric["delta"] is metric["ratio"] is None


@pytest.mark.parametrize("bad_digest", [None, "not-a-hash", "a" * 63, "g" * 64, "A" * 64])
def test_equal_but_invalid_hash_strings_are_not_matching_provenance(bad_digest):
    rows = _pair()
    for row in rows:
        row["quality_runtime_sha256"] = bad_digest
    pair = compare_a_e(rows)[0]
    assert pair["quality_issues"]
    assert _measurements(pair)["PSNR-micro (dB)"]["delta"] is None


@pytest.mark.parametrize("family", ["unet", "transformer"])
def test_duplicate_A_or_E_results_are_explicit_errors(family):
    with pytest.raises(ValueError, match="Ambiguous"):
        compare_a_e(_pair() + [_row(family)])


def test_missing_A_is_not_filled_by_another_model_or_zero():
    rows = [_row("transformer", micro_psnr=15.0), _row("graph_unet", group="C", micro_psnr=2.0)]
    metric = _measurements(compare_a_e(rows)[0])["PSNR-micro (dB)"]
    assert metric["a"] is None
    assert metric["e"] == 15.0
    assert metric["delta"] is None


def test_read_only_render_does_not_mutate_rows_and_discloses_limits():
    rows = _pair()
    before = copy.deepcopy(rows)
    result = render_a_e(rows)
    assert rows == before
    assert "E - A" in result and "E / A" in result
    assert "stored identity claims" in result
    assert "not original-data revalidation" in result
    assert "Training duration is not derived" in result
    assert "No A/E results available" in render_a_e([])


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "synthetic_ablation_cli_test", PROJECT / "scripts/run_ablations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("details", [False, True])
def test_summary_cli_only_reads_collect_results_and_details_are_explicit(
    monkeypatch, capsys, details
):
    cli = _load_script()
    rows = _pair()
    collected = []

    def collect(project, families):
        collected.append((project, families))
        return rows

    def forbidden(*args, **kwargs):
        raise AssertionError("Summary started a plan, process, checkpoint, or GPU operation")

    monkeypatch.setattr(cli, "collect_summary", collect)
    monkeypatch.setattr(cli, "plan_commands", forbidden)
    monkeypatch.setattr(cli, "execute_commands", forbidden)
    monkeypatch.setattr(cli.subprocess, "run", forbidden)
    monkeypatch.setattr(cli, "render_summary", lambda actual: "SYNTHETIC_SUPPLEMENTARY_SENTINEL")
    assert cli.main(["--stage", "summary", *(["--details"] if details else [])]) == 0
    output = capsys.readouterr().out
    assert len(collected) == 1
    assert "Primary comparison: A (U-Net only) vs E (Transformer only)" in output
    assert ("SYNTHETIC_SUPPLEMENTARY_SENTINEL" in output) is details


def test_comparison_module_import_and_render_do_not_import_torch_or_engine(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy"} or "engine" in name:
            raise AssertionError(f"Read-only comparison imported heavy runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location(
        "synthetic_read_only_comparison", PROJECT / "src/asgcn_unet/ablation_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "Primary comparison" in module.render_a_e(_pair())


def test_finite_inputs_with_overflowing_delta_are_unavailable():
    rows = [_row(micro_psnr=-1e308), _row("transformer", micro_psnr=1e308)]
    metric = _measurements(compare_a_e(rows)[0])["PSNR-micro (dB)"]
    assert metric["a"] == -1e308
    assert metric["e"] == 1e308
    assert metric["delta"] is None
    assert metric["ratio"] is None


def test_finite_inputs_with_overflowing_ratio_are_unavailable():
    rows = [_row(mean_ms=1e-308), _row("transformer", mean_ms=1e308)]
    metric = _measurements(compare_a_e(rows)[0])["Latency (ms)"]
    assert metric["a"] == 1e-308
    assert metric["e"] == 1e308
    assert metric["delta"] == 1e308
    assert metric["ratio"] is None
    assert "inf" not in render_a_e(rows).lower().split("| latency (ms)")[1].split("\n")[0]
