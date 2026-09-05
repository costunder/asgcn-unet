"""Result summaries must not conflate distinct successful experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import summarize_eval
from tests.test_summarize_eval import _reports


def test_equal_basename_roots_keep_distinct_paths_and_independent_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = _reports(tmp_path / "first" / "aid", "ann")
    right = _reports(tmp_path / "second" / "aid", "ann")
    left_score = json.loads((left / "metrics.json").read_text(encoding="utf-8"))["quality"][
        "micro"
    ]["psnr"]
    report = json.loads((right / "metrics.json").read_text(encoding="utf-8"))
    report["quality"]["micro"]["psnr"] = 23.25
    (right / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.chdir(tmp_path)
    rows, warnings = summarize_eval.collect_rows([left.parent, right.parent])
    assert warnings == []
    assert len(rows) == 2
    assert {row["run"] for row in rows} == {"first/aid", "second/aid"}
    assert {row["mode"] for row in rows} == {"ann"}
    assert {row["micro_psnr"] for row in rows} == {left_score, 23.25}
    rendered = summarize_eval.render_rows(rows, "markdown")
    assert "first/aid" in rendered and "second/aid" in rendered
    assert {path: path.read_bytes() for path in before} == before
