"""Read-only A/E decoder-only comparison; never relabel the old graph Transformer.

Quality dataset identifiers are compared as stored claims, not rehashed against
the original datasets here. Delta columns remain unavailable when their own
quality/benchmark provenance checks fail. Neither this module nor its imports
load checkpoints, initialize CUDA, or run model inference.
"""

from __future__ import annotations

import math
import re
from typing import Any


def _number(value: Any) -> int | float | None:
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _matching_digest(left: dict, right: dict, field: str) -> bool:
    value = left.get(field)
    return (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and value == right.get(field)
    )


def _pair_issues(left: dict, right: dict, kind: str) -> list[str]:
    if not left or not right or any(
        row.get(f"{kind}_model_contract_valid") is None for row in (left, right)
    ):
        return ["A or E result missing"]
    issues = []
    for label, row in (("A", left), ("E", right)):
        if row.get("training_settings_match") is not True:
            issues.append(f"{label} training contract unavailable/mismatched")
        for flag in (f"{kind}_eligible", f"{kind}_model_contract_valid", f"{kind}_mode_valid"):
            if row.get(flag) is not True:
                issues.append(f"{label} {flag} is not true")
        if kind == "benchmark":
            for flag in ("benchmark_contracts_verified", "benchmark_checkpoint_match", "io_excluded"):
                if row.get(flag) is not True:
                    issues.append(f"{label} {flag} is not true")
    if kind == "quality":
        count = _number(left.get("frames"))
        if count is None or count <= 0 or count != _number(right.get("frames")):
            issues.append("quality frame coverage unavailable/mismatched")
        fields = (
            "quality_dataset_claimed_sha256", "quality_runtime_sha256",
            "quality_source_sha256", "quality_precision_sha256",
        )
    else:
        fields = (
            "benchmark_evaluation_dataset_sha256", "benchmark_runtime_sha256",
            "benchmark_source_sha256", "benchmark_precision_sha256",
        )
    issues.extend(f"{field} unavailable/mismatched" for field in fields if not _matching_digest(left, right, field))
    return issues


def compare_a_e(rows: list[dict]) -> list[dict]:
    """One pair per dataset, A=unet/ann and E=transformer/ann only."""
    output = []
    datasets = sorted({row["dataset"] for row in rows if isinstance(row.get("dataset"), str)})
    for dataset in datasets:
        pair = {}
        for family, group in (("unet", "A"), ("transformer", "E")):
            selected = [
                row for row in rows if row.get("dataset") == dataset
                and row.get("family") == family and row.get("group") == group
                and row.get("mode") == "ann"
            ]
            if len(selected) > 1:
                raise ValueError(f"Ambiguous {group} baseline result for {dataset}")
            pair[group] = selected[0] if selected else {}
        left, right = pair["A"], pair["E"]
        quality_issues = _pair_issues(left, right, "quality")
        benchmark_issues = _pair_issues(left, right, "benchmark")
        measurements = []
        for field, label, kind in (
            ("frames", "Frames", "quality"),
            ("parameters", "Parameters", "model"),
            ("micro_psnr", "PSNR-micro (dB)", "quality"),
            ("micro_ssim", "SSIM-micro", "quality"),
            ("macro_psnr", "PSNR-macro (dB)", "quality"),
            ("macro_ssim", "SSIM-macro", "quality"),
            ("mean_ms", "Latency (ms)", "benchmark"),
            ("fps", "FPS", "benchmark"),
            ("vram_mib", "VRAM (MiB)", "benchmark"),
        ):
            a, e = _number(left.get(field)), _number(right.get(field))
            matched = (
                not quality_issues if kind == "quality" else not benchmark_issues
                if kind == "benchmark" else all(
                    row.get("quality_model_contract_valid") is True for row in (left, right)
                )
            )
            valid = matched and a is not None and e is not None
            measurements.append({
                "metric": label, "a": a, "e": e,
                "delta": _number(e - a) if valid else None,
                "ratio": _number(e / a) if valid and a > 0 and e >= 0 and kind != "quality" else None,
            })
        output.append({
            "dataset": dataset, "measurements": measurements,
            "quality_issues": quality_issues, "benchmark_issues": benchmark_issues,
            "a_run": left.get("run"), "e_run": right.get("run"),
        })
    return output


def render_a_e(rows: list[dict]) -> str:
    """Compact primary table; existing detailed suite records remain separate."""
    def cell(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.5f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "## Primary comparison: A (U-Net only) vs E (Transformer only)", "",
        "Same normalized-event mean raster input; neither model has a GNN/SNN encoder.",
        "Old graph_transformer results are not E and are never reused in this comparison.", "",
    ]
    pairs = compare_a_e(rows)
    if not pairs:
        lines.extend(["No A/E results available. No training or inference was started.", ""])
    for pair in pairs:
        lines.extend([
            f"### {pair['dataset']}", "",
            "| Metric | A: U-Net | E: Transformer | E - A | E / A |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        lines.extend(
            "| " + " | ".join(cell(row[key]) for key in ("metric", "a", "e", "delta", "ratio")) + " |"
            for row in pair["measurements"]
        )
        lines.append("")
        for label, key in (("Quality", "quality_issues"), ("Benchmark", "benchmark_issues")):
            issues = pair[key]
            lines.append(f"{label} pair: " + (
                "; ".join(issues) if issues else "stored comparison conditions match"
            ))
            lines.append("")
    lines.extend([
        "PSNR/SSIM/FPS higher is better; latency/VRAM lower is better. Ratios use E/A, not A/E.",
        "Raw values are stored results. Deltas/ratios are N/A if their comparison conditions fail.",
        ("Quality dataset hashes are stored identity claims, not original-data revalidation. "
         "Benchmark speed/memory are compute-only sample measurements, not full-run maxima."),
        "Training duration is not derived from inference latency and is not measured by this summary.",
        "Use --details for supplementary B/C/D, ANN controls, all T/dynamics, and provenance records.",
    ])
    return "\n".join(lines)
