"""Print existing evaluation results without importing torch or changing artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SCORES = ("psnr", "ssim", "rmse", "temporal_l1")
_COLUMNS = (
    "dataset",
    "run",
    "mode",
    "frames",
    *(f"{aggregation}_{metric}" for aggregation in ("micro", "macro") for metric in _SCORES),
    "mean_ms",
    "fps",
    "peak_allocated_mib",
    "quality_eligible",
    "benchmark_eligible",
    "benchmark_io",
    "status",
)
_COMPACT = (
    "mode",
    "frames",
    "micro_psnr",
    "micro_ssim",
    "macro_psnr",
    "macro_ssim",
    "mean_ms",
    "fps",
    "peak_allocated_mib",
    "quality_eligible",
    "benchmark_eligible",
    "benchmark_io",
    "status",
)
_LABELS = {
    "mode": "Mode",
    "frames": "Frames",
    "micro_psnr": "PSNR-u",
    "micro_ssim": "SSIM-u",
    "macro_psnr": "PSNR-m",
    "macro_ssim": "SSIM-m",
    "mean_ms": "ms",
    "fps": "FPS",
    "peak_allocated_mib": "VRAM-MiB",
    "quality_eligible": "Quality",
    "benchmark_eligible": "Bench",
    "benchmark_io": "I/O",
    "status": "Status",
}


def _archived(path: Path) -> bool:
    return any(
        ".failed-" in part or part.startswith(".failed") or ".incomplete-" in part
        for part in path.parts
    )


def _read_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"invalid {path.name}: {type(error).__name__}"
    if not isinstance(result, dict):
        return None, f"invalid {path.name}: expected an object"
    return result, None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _eligibility(value: Any) -> str:
    return "yes" if value is True else "no" if value is False else "unavailable"


def _dataset(quality: dict[str, Any], benchmark: dict[str, Any]) -> str:
    dataset = quality.get("dataset")
    if not isinstance(dataset, str):
        protocol = benchmark.get("benchmark_protocol", {})
        evaluation_dataset = (
            protocol.get("evaluation_dataset", {}) if isinstance(protocol, dict) else {}
        )
        transform = (
            evaluation_dataset.get("transform", {}) if isinstance(evaluation_dataset, dict) else {}
        )
        dataset = transform.get("type") if isinstance(transform, dict) else None
    if not isinstance(dataset, str):
        dataset = None
    return {"eventhdr": "EventHDR", "eventaid_r_zip": "EventAid-R"}.get(
        dataset, dataset or "unavailable"
    )


def collect_rows(roots: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep distinct runs distinct; skip recovery archives and expose incomplete runs."""
    directories: dict[Path, str] = {}
    display_root = Path.cwd().resolve()
    warnings = []
    for root in roots:
        if not root.exists():
            warnings.append(f"Result path does not exist: {root}")
            continue
        if root.is_file():
            if root.name not in ("metrics.json", "benchmark.json"):
                warnings.append(f"Expected metrics.json or benchmark.json: {root}")
                continue
            paths = [root]
        else:
            paths = []
            for current, children, filenames in os.walk(root):
                children[:] = [name for name in children if not _archived(Path(name))]
                paths.extend(
                    Path(current) / name
                    for name in ("metrics.json", "benchmark.json")
                    if name in filenames
                )
        for path in paths:
            if _archived(path):
                continue
            directory = path.parent.resolve()
            try:
                run = directory.parent.relative_to(display_root).as_posix()
            except ValueError:
                run = directory.parent.as_posix()
            # Multiple command-line roots can refer to the same artifact.
            # Distinct roots with the same basename must not share a run label.
            directories.setdefault(directory, run)
    rows = []
    for directory, run in sorted(directories.items(), key=lambda item: str(item[0])):
        quality, quality_issue = _read_report(directory / "metrics.json")
        benchmark, benchmark_issue = _read_report(directory / "benchmark.json")
        issues = [issue for issue in (quality_issue, benchmark_issue) if issue is not None]
        q = quality or {}
        b = benchmark or {}
        quality_scores = q.get("quality", {})
        if not isinstance(quality_scores, dict):
            quality_scores = {}
            issues.append("invalid quality object")
        identity_fields = (
            "checkpoint_model_sha256",
            "inference_mode",
            "simulation_steps",
            "snn_dynamics",
        )
        mismatches = [key for key in identity_fields if key in q and key in b and q[key] != b[key]]
        if mismatches:
            issues.append("benchmark identity mismatch: " + ",".join(mismatches))
        row = {key: None for key in _COLUMNS}
        row.update(
            {
                "dataset": _dataset(q, b),
                "run": run,
                "mode": directory.name,
                "frames": _number(quality_scores.get("frames")),
                "quality_eligible": _eligibility(q.get("report_eligible")),
                "benchmark_eligible": "mismatch"
                if mismatches
                else _eligibility(b.get("report_eligible")),
                "benchmark_io": "excluded"
                if b.get("io_excluded") is True
                else "included"
                if b.get("io_excluded") is False
                else "unavailable",
            }
        )
        for aggregation in ("micro", "macro"):
            scores = quality_scores.get(aggregation)
            if not isinstance(scores, dict):
                if quality is not None:
                    issues.append(f"missing quality.{aggregation}")
                continue
            for metric in _SCORES:
                row[f"{aggregation}_{metric}"] = _number(scores.get(metric))
                if row[f"{aggregation}_{metric}"] is None:
                    issues.append(f"unavailable {aggregation}.{metric}")
        if not mismatches:
            for field in ("mean_ms", "fps"):
                row[field] = _number(b.get(field))
            row["peak_allocated_mib"] = _number(b.get("peak_gpu_memory_mb"))
        if quality is not None and row["frames"] is None:
            issues.append("unavailable frames")
        if benchmark is not None and not mismatches:
            for field in ("mean_ms", "fps"):
                if row[field] is None:
                    issues.append(f"unavailable benchmark.{field}")
        if q.get("report_eligible") is False:
            issues.append(
                "quality ineligible: "
                + str(q.get("report_ineligible_reasons", "reason unavailable"))
            )
        if b.get("report_eligible") is False:
            issues.append(
                "benchmark ineligible: "
                + str(b.get("report_ineligible_reasons", "reason unavailable"))
            )
        row["status"] = "ok" if not issues else "incomplete/check"
        for issue in issues:
            warnings.append(f"{row['dataset']} {run}/{directory.name}: {issue}")
        rows.append(row)

    def ordering(row: dict[str, Any]) -> tuple[str, str, str, int]:
        name = row["mode"]
        base, marker, suffix = name.rpartition("_T")
        return (
            row["dataset"],
            row["run"],
            base if marker and suffix.isdigit() else name,
            int(suffix) if marker and suffix.isdigit() else -1,
        )

    return sorted(rows, key=ordering), warnings


def _formatted(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def render_rows(rows: Sequence[dict[str, Any]], output_format: str = "text") -> str:
    if output_format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _formatted(row[key]) for key in _COLUMNS} for row in rows)
        return stream.getvalue().rstrip("\n")
    if output_format not in ("text", "markdown"):
        raise ValueError(f"Unsupported format: {output_format}")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["run"]), []).append(row)
    output = []
    for (dataset, run), group in groups.items():
        output.extend([f"{dataset} | {run}", ""])
        cells = [[_LABELS.get(key, key) for key in _COMPACT]] + [
            [_formatted(row[key]) for key in _COMPACT] for row in group
        ]
        if output_format == "markdown":
            output.append("| " + " | ".join(cells[0]) + " |")
            output.append("| " + " | ".join("---" for _ in _COMPACT) + " |")
            output.extend(
                "| "
                + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in row)
                + " |"
                for row in cells[1:]
            )
        else:
            widths = [max(len(row[index]) for row in cells) for index in range(len(_COMPACT))]
            output.extend(
                "  ".join(
                    value.ljust(width) for value, width in zip(row, widths, strict=True)
                ).rstrip()
                for row in cells
            )
        output.append("")
    output.append("u=micro (frame mean); m=macro (group mean). PSNR/SSIM higher is better.")
    output.append(
        "ms/FPS/VRAM are benchmark measurements, not full-dataset end-to-end speed or memory maxima."
    )
    output.append(
        "Quality/Bench show stored report_eligible flags; N/A/unavailable means not recorded. Archived .failed-* and .incomplete-* runs are excluded."
    )
    return "\n".join(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("runs")],
        help="result directories or metrics/benchmark JSON files (default: runs)",
    )
    parser.add_argument("--format", choices=("text", "markdown", "csv"), default="text")
    args = parser.parse_args(argv)
    rows, warnings = collect_rows(args.paths)
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if not rows:
        print(
            "No evaluation artifacts found. Supply the completed evaluation directory.",
            file=sys.stderr,
        )
        return 1
    print(render_rows(rows, args.format))
    # Preserve usable rows while making incomplete/invalid inputs visible to automation.
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
