"""Explicit, non-destructive orchestration of the architecture ablation matrix.

Planning and summary never initialize torch/CUDA. Actual execution reuses the
existing environment, profiling, checkpoint, and evaluation safety contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAMILIES = {
    "unet": ("identity", "unet", "A"),
    "pointwise_unet": ("pointwise", "unet", "B"),
    "graph_unet": ("graph", "unet", "C/D"),
    "transformer": ("identity", "transformer", "E"),
}
LEGACY_FAMILIES = ("graph_transformer",)
STAGES = ("plan", "profile", "train", "calibrate", "eval", "all", "summary")
STEPS = (4, 8, 16, 32)
DYNAMICS = ("literal_eq15", "standard_if")
TRANSFORMER_CONFIG = {
    "depths": [1, 1, 2, 1, 1],
    "heads": [3, 6, 12, 6, 3],
    "window_size": 8,
    "mlp_ratio": 4.0,
}


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return result


def expected_config(project: Path, family: str, split: str) -> dict[str, Any]:
    """The existing full experiment, with ONLY the named architecture/path change."""
    encoder, decoder, _ = FAMILIES[family]
    source = {"train": "fast.json", "hdr": "hdr-fast.json", "aid": "aid-fast.json"}[split]
    config = copy.deepcopy(_load(project / "configs" / source))
    config["model"].update(encoder_kind=encoder, decoder_kind=decoder)
    if encoder != "graph":
        config["model"]["spline_backend"] = "torch"
    if decoder == "transformer":
        config["model"]["transformer_config"] = copy.deepcopy(TRANSFORMER_CONFIG)
    if split == "train":
        config["output"]["run_dir"] = f"runs/ablations/{family}"
    else:
        config["eval"]["output_dir"] = f"runs/ablations/{family}/eval/{split}"
    return config


def validate_suite(project: Path, families: tuple[str, ...]) -> None:
    if not families or len(set(families)) != len(families):
        raise ValueError("Select one or more distinct experiment families")
    for family in families:
        if family in LEGACY_FAMILIES:
            raise ValueError(
                f"{family} is a preserved legacy experiment, not active group E. "
                "Use transformer for the graph-free, non-spiking Transformer baseline."
            )
        if family not in FAMILIES:
            raise ValueError(f"Unknown family: {family}")
        for split in ("train", "hdr", "aid"):
            path = config_path(project, family, split)
            if _load(path) != expected_config(project, family, split):
                raise ValueError(
                    f"Ablation contract differs from the full baseline: {path.name}. "
                    "Do not silently change dataset, training budget, or model scale."
                )


def config_path(project: Path, family: str, split: str) -> Path:
    return project / "configs" / "ablations" / f"{family}-{split}.json"


def modes(family: str) -> tuple[tuple[str, int | None, str | None], ...]:
    ann = (("ann", None, None),)
    return (
        ann
        if FAMILIES[family][0] == "identity"
        else ann + tuple(("snn", steps, dynamics) for dynamics in DYNAMICS for steps in STEPS)
    )


def mode_label(mode: str, steps: int | None, dynamics: str | None) -> str:
    return "ann" if mode == "ann" else f"snn_{dynamics}_T{steps}"


@dataclass(frozen=True)
class Command:
    family: str
    stage: str
    argv: tuple[str, ...]


def plan_commands(
    project: Path,
    *,
    stage: str = "all",
    families: tuple[str, ...] = tuple(FAMILIES),
    python: str = sys.executable,
    cpu_threads: int = 4,
) -> list[Command]:
    validate_suite(project, families)
    if stage not in STAGES or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Invalid stage or CPU thread count")
    if stage == "summary":
        return []
    stages = ("profile", "train", "calibrate", "eval") if stage in {"plan", "all"} else (stage,)
    commands = [
        Command(
            "suite",
            "environment",
            (
                python,
                str(project / "scripts/check_env.py"),
                "--require-cuda",
                "--require-full-data",
                "--lock",
                str(project / "constraints/py312.txt"),
                "--runtime-profile",
                str(project / "constraints/server.json"),
            ),
        )
    ]
    cli = (python, "-B", "-m", "asgcn_unet.cli")
    if "profile" in stages:
        # Data transforms are identical across families: decode once, not four times.
        for split in ("train", "aid"):
            commands.append(
                Command(
                    "suite",
                    "inspect",
                    cli
                    + (
                        "inspect",
                        "--config",
                        str(config_path(project, families[0], split)),
                        "--validate-all",
                        "--samples",
                        "3",
                    ),
                )
            )
    for family in families:
        train = str(config_path(project, family, "train"))
        run = project / "runs/ablations" / family
        profile = str(project / "runs/ablations" / f"{family}-profile.json")
        if "profile" in stages:
            commands.append(
                Command(
                    family,
                    "profile",
                    cli
                    + (
                        "profile",
                        "--config",
                        train,
                        "--output",
                        profile,
                        "--samples",
                        "3",
                        "--top-density",
                        "10",
                        "--cpu-threads",
                        str(cpu_threads),
                    ),
                )
            )
        if "train" in stages:
            commands.append(
                Command(
                    family,
                    "train",
                    cli
                    + (
                        "train",
                        "--config",
                        train,
                        "--preflight-report",
                        profile,
                    ),
                )
            )
        if "calibrate" in stages and FAMILIES[family][0] != "identity":
            commands.append(
                Command(
                    family,
                    "calibrate",
                    cli
                    + (
                        "calibrate",
                        "--config",
                        train,
                        "--checkpoint",
                        str(run / "best.pt"),
                        "--output",
                        str(run / "best_snn.pt"),
                        "--samples",
                        "all",
                    ),
                )
            )
        if "eval" in stages:
            for split in ("hdr", "aid"):
                config = str(config_path(project, family, split))
                for mode, steps, dynamics in modes(family):
                    checkpoint = str(run / ("best.pt" if mode == "ann" else "best_snn.pt"))
                    common = (
                        "--config",
                        config,
                        "--checkpoint",
                        checkpoint,
                        "--output-dir",
                        str(run / "eval" / split),
                        "--inference-mode",
                        mode,
                    )
                    if mode == "snn":
                        common += (
                            "--simulation-steps",
                            str(steps),
                            "--snn-dynamics",
                            str(dynamics),
                        )
                    commands.append(Command(family, "evaluate", cli + ("evaluate",) + common))
                    commands.append(
                        Command(
                            family,
                            "benchmark",
                            cli
                            + ("benchmark",)
                            + common
                            + (
                                "--warmup",
                                "10",
                                "--steps",
                                "100",
                            ),
                        )
                    )
    # All requested architectures must pass measured preflight before any long
    # training starts. Stable ordering preserves each evaluation/benchmark pair.
    order = {
        "environment": 0,
        "inspect": 1,
        "profile": 2,
        "train": 3,
        "calibrate": 4,
        "evaluate": 5,
        "benchmark": 5,
    }
    return sorted(commands, key=lambda command: order[command.stage])


def _option(argv: tuple[str, ...], option: str) -> str:
    return argv[argv.index(option) + 1]


def _verify_calibration(command: Command) -> None:
    import torch

    from .engine import (
        _current_source_contract,
        _reporting_checkpoint_contract,
        load_model_checkpoint,
    )

    checkpoint = Path(_option(command.argv, "--output"))
    config = _load(Path(_option(command.argv, "--config")))
    _, saved = load_model_checkpoint(checkpoint, torch.device("cpu"), config["model"])
    _reporting_checkpoint_contract(
        saved, checkpoint, "snn", allow_unsealed_checkpoint_for_non_reporting=False
    )
    source = Path(_option(command.argv, "--checkpoint"))
    with source.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if saved["calibration_protocol"]["source_ann_checkpoint_sha256"] != digest:
        raise ValueError("Existing calibration belongs to a different ANN checkpoint")
    if saved["calibration_protocol"]["calibration_source"] != _current_source_contract():
        raise ValueError("Existing calibration belongs to different source code")


def execute_commands(
    commands: list[Command],
    project: Path,
    *,
    execute: bool = False,
    resume: bool = False,
    cpu_threads: int = 4,
    runner: Callable[..., Any] = subprocess.run,
    emit: Callable[[str], None] = print,
) -> None:
    """No shell, GPU selection, signals, automatic archive, or destructive recovery."""
    inspected: dict[tuple[str, str], tuple[bool, bool]] = {}
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("CPU thread count must be positive")
    environment.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    environment.setdefault("MKL_NUM_THREADS", environment["OMP_NUM_THREADS"])
    for command in commands:
        argv = command.argv
        if execute and resume and command.stage == "profile":
            output = Path(_option(argv, "--output"))
            if output.exists():
                if _load(output).get("passed") is True:
                    argv = argv[:4] + (
                        "verify-profile",
                        "--config",
                        _option(argv, "--config"),
                        "--report",
                        str(output),
                    )
                else:
                    argv += ("--resume-scan",)
        if execute and resume and command.stage == "train":
            config = _load(Path(_option(argv, "--config")))
            last = project / config["output"]["run_dir"] / "last.pt"
            if last.exists():
                argv += ("--resume", str(last))
        if (
            execute
            and resume
            and command.stage == "calibrate"
            and Path(_option(argv, "--output")).exists()
        ):
            _verify_calibration(command)
            emit(f"[{command.family}] verified existing full calibration; retained")
            continue
        if execute and resume and command.stage in {"evaluate", "benchmark"}:
            mode = _option(argv, "--inference-mode")
            steps = _option(argv, "--simulation-steps") if mode == "snn" else "16"
            dynamics = _option(argv, "--snn-dynamics") if mode == "snn" else None
            output = _option(argv, "--output-dir")
            label = mode_label(mode, int(steps), dynamics)
            key = (output, label)
            if key not in inspected:
                mode_dir = Path(output) / label
                if mode_dir.exists():
                    inspect = [
                        argv[0],
                        str(project / "scripts/eval_resume.py"),
                        "--config",
                        _option(argv, "--config"),
                        "--checkpoint",
                        _option(argv, "--checkpoint"),
                        "--output-dir",
                        output,
                        "--inference-mode",
                        mode,
                        "--simulation-steps",
                        steps,
                        "--benchmark-warmup",
                        "10",
                        "--benchmark-steps",
                        "100",
                        "--require-cuda",
                    ]
                    if dynamics:
                        inspect.extend(("--snn-dynamics", dynamics))
                    result = runner(
                        inspect,
                        cwd=project,
                        env=environment,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    lines = result.stdout.strip().splitlines()
                    fields = lines[-1].split() if lines else []
                    if fields not in (["0", "0"], ["0", "1"], ["1", "0"], ["1", "1"]):
                        raise ValueError("Invalid evaluation resume inspector response")
                    completed = tuple(value == "1" for value in fields)
                    if not completed[0] and any(mode_dir.iterdir()):
                        raise ValueError(
                            f"Incomplete quality output retained at {mode_dir}; no automatic archive/overwrite"
                        )
                    if completed[0] and not completed[1] and (mode_dir / "benchmark.json").exists():
                        raise ValueError(
                            f"Incomplete benchmark retained at {mode_dir}; no automatic overwrite"
                        )
                    inspected[key] = completed
                else:
                    inspected[key] = (False, False)
            if inspected[key][0 if command.stage == "evaluate" else 1]:
                emit(f"[{command.family}] verified completed {label} {command.stage}; retained")
                continue
        emit(f"[{command.family}:{command.stage}] {shlex.join(argv)}")
        if execute:
            # Deliberately no shell=True and no exception handler that kills children/sessions.
            runner(list(argv), cwd=project, env=environment, check=True)


def _finite(value: Any) -> int | float | None:
    return value if type(value) in {int, float} and math.isfinite(value) else None


def _checked_contract(value: Any) -> dict | None:
    if not isinstance(value, dict) or not isinstance(value.get("contract"), dict):
        return None
    encoded = json.dumps(
        value["contract"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return value["contract"] if value.get("sha256") == hashlib.sha256(encoded).hexdigest() else None


def _training_settings(config: dict) -> dict:
    result = copy.deepcopy(config)
    for key in ("output", "preflight_gate"):
        result.pop(key, None)
    for key in ("root", "val_root", "split_manifest", "file_manifest"):
        result.get("dataset", {}).pop(key, None)
    return result


def collect_summary(project: Path, families: tuple[str, ...] = tuple(FAMILIES)) -> list[dict]:
    """Read exact suite paths; never re-label runs/fast or archive results as ablations."""
    from .offline_viewer import ExportLimits, _read_selected

    validate_suite(project, families)
    selection = {
        "quality": {"frames": True, "micro": True, "macro": True},
        "report_eligible": True,
        "inference_mode": True,
        "simulation_steps": True,
        "snn_dynamics": True,
        "checkpoint_model_sha256": True,
        "execution": {"model": {"total_parameters": True, "trainable_parameters": True}},
        "evaluation_protocol": {
            "model_config": True,
            "execution": True,
            "evaluation_dataset": {"sha256": True},
            "runtime": True,
            "source": True,
            "precision": True,
            "checkpoint": {"training_config": True, "source_ann": {"training_config": True}},
        },
    }
    bench_selection = {
        key: True
        for key in (
            "mean_ms",
            "fps",
            "peak_gpu_memory_mb",
            "report_eligible",
            "inference_mode",
            "simulation_steps",
            "snn_dynamics",
            "checkpoint_model_sha256",
            "io_excluded",
        )
    }
    bench_selection["benchmark_protocol"] = {
        "runtime": True,
        "source": True,
        "precision": True,
        "evaluation_dataset": True,
        "model_config": True,
    }
    rows = []
    for family in families:
        for split in ("hdr", "aid"):
            model = _load(config_path(project, family, split))["model"]
            for mode, steps, dynamics in modes(family):
                label = mode_label(mode, steps, dynamics)
                directory = project / "runs/ablations" / family / "eval" / split / label
                reports = {}
                for name, selected in (("metrics", selection), ("benchmark", bench_selection)):
                    path = directory / f"{name}.json"
                    reports[name] = (
                        _read_selected(path, selected, ExportLimits()) if path.is_file() else {}
                    )
                q, b = reports["metrics"], reports["benchmark"]
                issues = []
                if not q:
                    issues.append("missing quality")
                if not b:
                    issues.append("missing benchmark")
                protocol = q.get("evaluation_protocol", {})
                stored_model = _checked_contract(protocol.get("model_config"))
                if q and stored_model != model:
                    issues.append("model contract mismatch/unavailable")
                checkpoint = protocol.get("checkpoint", {})
                training_identity = (
                    checkpoint.get("training_config")
                    if mode == "ann"
                    else checkpoint.get("source_ann", {}).get("training_config")
                )
                stored_training = _checked_contract(training_identity)
                expected_training = _load(config_path(project, family, "train"))
                training_match = stored_training is not None and (
                    _training_settings(stored_training) == _training_settings(expected_training)
                )
                if q and not training_match:
                    issues.append("training settings mismatch/unavailable")
                mode_validity = {}
                for name, report in (("quality", q), ("benchmark", b)):
                    valid = (
                        (
                            report.get("inference_mode") == mode
                            and report.get("simulation_steps") == steps
                            and report.get("snn_dynamics") == dynamics
                        )
                        if report
                        else None
                    )
                    mode_validity[name] = valid
                    if valid is False:
                        issues.append("mode contract mismatch")
                benchmark_checkpoint_match = (
                    (
                        isinstance(q.get("checkpoint_model_sha256"), str)
                        and bool(q.get("checkpoint_model_sha256"))
                        and q.get("checkpoint_model_sha256") == b.get("checkpoint_model_sha256")
                    )
                    if q and b
                    else None
                )
                if benchmark_checkpoint_match is False:
                    issues.append("benchmark checkpoint mismatch")
                benchmark_protocol = b.get("benchmark_protocol", {})
                benchmark_model_valid = (
                    _checked_contract(benchmark_protocol.get("model_config")) == model
                )
                if b and not benchmark_model_valid:
                    issues.append("benchmark model contract mismatch/unavailable")
                quality_contracts = {
                    field: _checked_contract(protocol.get(field))
                    for field in ("runtime", "source", "precision")
                }
                benchmark_contracts = {
                    field: _checked_contract(benchmark_protocol.get(field))
                    for field in ("runtime", "source", "precision", "evaluation_dataset")
                }
                for field, contract in quality_contracts.items():
                    if q and contract is None:
                        issues.append(f"quality {field} hash invalid/unavailable")
                for field, contract in benchmark_contracts.items():
                    if b and contract is None:
                        issues.append(f"benchmark {field} hash invalid/unavailable")
                for field in ("runtime", "source", "precision"):
                    if (
                        q
                        and b
                        and (
                            quality_contracts[field] is None
                            or benchmark_contracts[field] is None
                            or quality_contracts[field] != benchmark_contracts[field]
                        )
                    ):
                        issues.append(f"benchmark {field} mismatch/unavailable")
                quality = q.get("quality", {})
                group = (
                    "A"
                    if family == "unet"
                    else "C"
                    if family == "graph_unet" and mode == "ann"
                    else "D"
                    if family == "graph_unet"
                    else "B"
                    if family == "pointwise_unet"
                    else "E"
                )
                rows.append(
                    {
                        "group": group
                        + ("-ANN-control" if family == "pointwise_unet" and mode == "ann" else ""),
                        "family": family,
                        "dataset": split,
                        "mode": label,
                        "run": str(directory),
                        "frames": _finite(quality.get("frames")),
                        **{
                            f"{aggregation}_{metric}": _finite(
                                quality.get(aggregation, {}).get(metric)
                            )
                            for aggregation in ("micro", "macro")
                            for metric in ("psnr", "ssim")
                        },
                        "mean_ms": _finite(b.get("mean_ms")),
                        "fps": _finite(b.get("fps")),
                        "vram_mib": _finite(b.get("peak_gpu_memory_mb")),
                        "parameters": _finite(
                            q.get("execution", {}).get("model", {}).get("total_parameters")
                        ),
                        "trainable_parameters": _finite(
                            q.get("execution", {}).get("model", {}).get("trainable_parameters")
                        ),
                        "training_settings_match": training_match if q else None,
                        "quality_model_contract_valid": stored_model == model if q else None,
                        "quality_mode_valid": mode_validity["quality"],
                        "benchmark_mode_valid": mode_validity["benchmark"],
                        "benchmark_checkpoint_match": benchmark_checkpoint_match,
                        "benchmark_model_contract_valid": benchmark_model_valid if b else None,
                        "quality_dataset_claimed_sha256": protocol.get(
                            "evaluation_dataset", {}
                        ).get("sha256"),
                        "quality_dataset_hash_verified": False if q else None,
                        **{
                            f"quality_{field}_sha256": protocol[field]["sha256"]
                            if contract is not None
                            else None
                            for field, contract in quality_contracts.items()
                        },
                        **{
                            f"benchmark_{field}_sha256": benchmark_protocol[field]["sha256"]
                            if contract is not None
                            else None
                            for field, contract in benchmark_contracts.items()
                        },
                        "benchmark_contracts_verified": (
                            benchmark_model_valid
                            and all(
                                contract is not None for contract in benchmark_contracts.values()
                            )
                        )
                        if b
                        else None,
                        "quality_eligible": q.get("report_eligible"),
                        "benchmark_eligible": b.get("report_eligible"),
                        "io_excluded": b.get("io_excluded"),
                        "status": "; ".join(dict.fromkeys(issues))
                        or "stored contracts match; runtime/data comparability not reverified",
                    }
                )
    for split in ("hdr", "aid"):
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
            available = [
                row
                for row in rows
                if row["dataset"] == split
                and row["mean_ms" if field.startswith("benchmark_") else "frames"] is not None
            ]
            identities = {row[field] for row in available if isinstance(row[field], str)}
            matched = (
                len(identities) == 1 and len(available) > 1 and all(row[field] for row in available)
            )
            for row in available:
                row[field.removesuffix("_sha256") + "_matched"] = bool(matched)
                if not matched:
                    row["status"] += (
                        f"; {field.removesuffix('_sha256')} comparison unavailable/mismatched"
                    )
    return rows


def render_summary(rows: list[dict]) -> str:
    fields = (
        "group",
        "dataset",
        "mode",
        "frames",
        "parameters",
        "micro_psnr",
        "micro_ssim",
        "macro_psnr",
        "macro_ssim",
        "mean_ms",
        "fps",
        "vram_mib",
        "quality_eligible",
        "benchmark_eligible",
        "io_excluded",
        "training_settings_match",
        "quality_dataset_claimed_matched",
        "quality_dataset_hash_verified",
        "quality_runtime_matched",
        "quality_source_matched",
        "benchmark_contracts_verified",
        "benchmark_evaluation_dataset_matched",
        "benchmark_runtime_matched",
        "benchmark_source_matched",
        "benchmark_precision_matched",
        "status",
    )

    def cell(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.5f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(field)) for field in fields) + " |" for row in rows)
    return (
        "\n".join(lines)
        + "\n\nN/A is missing, not zero. Timing is compute-only when io_excluded=true. "
        "quality_* columns do not validate FPS comparability; use benchmark_* columns. "
        "Quality dataset hashes are stored claims only (full-frame identity arrays are not retained/rehashed here). "
        "Stored eligibility is not proof of matched training, hardware, or dataset provenance. "
        "Prior runs/fast results are separate historical references, never relabelled."
    )
