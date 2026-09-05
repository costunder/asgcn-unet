from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

from scripts import summarize_eval


def _reports(root: Path, mode: str = "ann", dataset: str = "eventaid_r_zip") -> Path:
    directory = root / mode
    directory.mkdir(parents=True)
    scores = {"psnr": 11.72167, "ssim": 0.75323, "rmse": 0.26289, "temporal_l1": 0.01379}
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "report_eligible": True,
                "checkpoint_model_sha256": "same-model",
                "quality": {"frames": 51512, "micro": scores, "macro": {**scores, "ssim": 0.77978}},
            }
        )
    )
    (directory / "benchmark.json").write_text(
        json.dumps(
            {
                "report_eligible": True,
                "checkpoint_model_sha256": "same-model",
                "mean_ms": 71.858,
                "fps": 13.9163,
                "peak_gpu_memory_mb": 198.07,
                "io_excluded": True,
            }
        )
    )
    return directory


def test_summary_preserves_both_datasets_and_numeric_t_order_and_excludes_archives(
    tmp_path: Path,
) -> None:
    _reports(tmp_path / "aid", "ann")
    _reports(tmp_path / "aid", "snn_literal_eq15_T16")
    _reports(tmp_path / "aid", "snn_literal_eq15_T4")
    _reports(tmp_path / "aid", "ann.failed-20260904T020452Z")
    _reports(tmp_path / "aid" / "ann.incomplete-20260905T010203Z-123-random", "ann")
    _reports(tmp_path / "hdr", "ann", "eventhdr")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}

    rows, warnings = summarize_eval.collect_rows([tmp_path])

    assert warnings == []
    assert len(rows) == 4
    assert [row["mode"] for row in rows if row["dataset"] == "EventAid-R"] == [
        "ann",
        "snn_literal_eq15_T4",
        "snn_literal_eq15_T16",
    ]
    assert {row["dataset"] for row in rows} == {"EventHDR", "EventAid-R"}
    assert all(row["macro_ssim"] == 0.77978 for row in rows)
    assert all(row["benchmark_io"] == "excluded" for row in rows)
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}


def test_missing_benchmark_remains_explicit_and_not_zero(tmp_path: Path) -> None:
    directory = tmp_path / "ann"
    directory.mkdir()
    (directory / "metrics.json").write_text('{"quality":{"frames":10},"report_eligible":true}')
    rows, warnings = summarize_eval.collect_rows([tmp_path])
    assert rows[0]["fps"] is None
    assert rows[0]["benchmark_eligible"] == "unavailable"
    assert rows[0]["micro_psnr"] is None
    assert any("missing benchmark.json" in warning for warning in warnings)
    assert "N/A" in summarize_eval.render_rows(rows)


def test_identity_mismatch_prevents_combining_benchmark_with_quality(tmp_path: Path) -> None:
    directory = _reports(tmp_path)
    path = directory / "benchmark.json"
    benchmark = json.loads(path.read_text())
    benchmark["checkpoint_model_sha256"] = "different-model"
    path.write_text(json.dumps(benchmark))
    rows, warnings = summarize_eval.collect_rows([tmp_path])
    assert rows[0]["micro_psnr"] == 11.72167
    assert rows[0]["fps"] is None
    assert rows[0]["benchmark_eligible"] == "mismatch"
    assert any("identity mismatch" in warning for warning in warnings)


def test_malformed_json_does_not_hide_other_results(tmp_path: Path) -> None:
    directory = _reports(tmp_path, "ann")
    _reports(tmp_path, "snn_literal_eq15_T4")
    (directory / "metrics.json").write_text("{")
    rows, warnings = summarize_eval.collect_rows([tmp_path])
    assert len(rows) == 2
    assert any("invalid metrics.json" in warning for warning in warnings)


def test_csv_contains_full_scores_and_markdown_has_compact_comparison(tmp_path: Path) -> None:
    _reports(tmp_path)
    rows, warnings = summarize_eval.collect_rows([tmp_path])
    assert not warnings
    csv_rows = list(csv.DictReader(io.StringIO(summarize_eval.render_rows(rows, "csv"))))
    assert csv_rows[0]["micro_rmse"] == "0.26289"
    assert csv_rows[0]["macro_ssim"] == "0.77978"
    markdown = summarize_eval.render_rows(rows, "markdown")
    assert "| Mode | Frames |" in markdown
    assert "excluded" in markdown


def test_summary_cli_works_without_site_packages_or_torch_and_is_read_only(tmp_path: Path) -> None:
    _reports(tmp_path)
    script = Path(summarize_eval.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, "-S", str(script), str(tmp_path), "--format", "csv"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "EventAid-R" in completed.stdout
    assert "0.77978" in completed.stdout
