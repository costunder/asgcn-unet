"""Tiny synthetic metadata-budget tests; no actual model/data/server execution."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from asgcn_unet import offline_viewer as viewer
from tests.test_offline_viewer import fixture, snapshot


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_metadata_budget_requires_explicit_positive_integer(tmp_path, value):
    with pytest.raises(viewer.OfflineViewerError, match="retained_budget_bytes"):
        viewer.build_payload(tmp_path, retained_budget_bytes=value)


def test_budgeted_payload_preserves_every_existing_frame_and_mode(tmp_path):
    root = fixture(tmp_path / "synthetic-only")
    before = snapshot(root)
    original = viewer.build_payload(root)
    bounded = viewer.build_payload(root, retained_budget_bytes=32 * 1024**2)
    for key in ("datasets", "images", "warnings"):
        assert bounded[key] == original[key]
    assert bounded["export"]["retained_metadata_estimate_bytes"] > 0
    assert bounded["export"]["retained_metadata_estimate_bytes"] < 32 * 1024**2
    assert bounded["export"]["metadata_collection_peak_estimate_bytes"] <= 32 * 1024**2
    assert bounded["export"]["retained_metadata_budget_bytes"] == 32 * 1024**2
    assert snapshot(root) == before


def test_base_payload_failure_precedes_directory_or_input_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No directory/input read should occur after the initial budget refusal")
    monkeypatch.setattr(viewer, "_iter_artifact_directory", forbidden)
    with pytest.raises(viewer.OfflineViewerError, match="before base payload"):
        viewer.build_payload(tmp_path, retained_budget_bytes=1)


def test_cumulative_candidate_charge_stops_stream_before_next_png(tmp_path, monkeypatch):
    root = fixture(tmp_path / "synthetic-only")
    original_charge = viewer._RetainedMetadataBudget.charge
    original_png = viewer._png
    seen = {"candidates": 0, "pngs": 0}

    def amplified_charge(self, amount, stage):
        if stage == "registering a prediction candidate":
            seen["candidates"] += 1
            amount += 8 * 1024**2  # Amplified mocked planning cost, no actual allocation.
        return original_charge(self, amount, stage)

    def count_png(*args, **kwargs):
        seen["pngs"] += 1
        return original_png(*args, **kwargs)

    monkeypatch.setattr(viewer._RetainedMetadataBudget, "charge", amplified_charge)
    monkeypatch.setattr(viewer, "_png", count_png)
    with pytest.raises(viewer.OfflineViewerError, match="metadata budget"):
        viewer.build_payload(root, retained_budget_bytes=24 * 1024**2)
    assert 1 < seen["candidates"] < 8
    assert seen["pngs"] < 16
    assert not list(root.glob("*.html"))


def test_does_not_materialize_path_glob_or_iterdir_lists(tmp_path, monkeypatch):
    root = fixture(tmp_path / "synthetic-only")

    def forbidden(*args, **kwargs):
        pytest.fail("Artifact enumeration must stream through os.scandir")
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    result = viewer.build_payload(root, retained_budget_bytes=32 * 1024**2)
    assert sum(len(dataset["frames"]) for dataset in result["datasets"]) == 4


def test_unsorted_directory_stream_still_returns_sorted_modes(tmp_path, monkeypatch):
    root = fixture(tmp_path / "synthetic-only")
    original = viewer._iter_artifact_directory

    def reversed_tiny_fixture(path, *, predictions=False):
        yield from reversed(list(original(path, predictions=predictions)))
    monkeypatch.setattr(viewer, "_iter_artifact_directory", reversed_tiny_fixture)
    result = viewer.build_payload(root, retained_budget_bytes=32 * 1024**2)
    for dataset in result["datasets"]:
        assert [mode["id"] for mode in dataset["modes"]] == ["ann", "snn_literal_eq15_T4"]
        for frame in dataset["frames"]:
            assert [item["mode"] for item in frame["images"]] == ["ann", "snn_literal_eq15_T4"]


class _TrackedText(io.StringIO):
    def __init__(self, text):
        super().__init__(text)
        self.requests = []

    def readline(self, size=-1):
        assert size > 0, "CSV must never perform an unbounded readline"
        self.requests.append(size)
        return super().readline(size)


@pytest.mark.parametrize("record", ["x" * 80 + ",scene\n", '"' + "x\n" * 40 + '",scene\n',
                                   "," * 80 + "\n"])
def test_csv_bounds_entire_record_before_full_parse(record):
    handle = _TrackedText("sample_id,scene\n" + record)
    budget = viewer._RetainedMetadataBudget(2 * 1024**2)
    lines = viewer._BudgetCSVLines(handle, budget, viewer.ExportLimits(max_metadata_bytes=64))
    lines.begin_record()
    reader = csv.DictReader(lines)
    assert reader.fieldnames == ["sample_id", "scene"]
    lines.begin_record()
    with pytest.raises(viewer.OfflineViewerError, match="logical record"):
        next(reader)
    assert max(handle.requests) <= 65


def test_csv_record_budget_resets_between_small_rows():
    handle = _TrackedText("sample_id,scene\na,b\nc,d\n")
    lines = viewer._BudgetCSVLines(handle, viewer._RetainedMetadataBudget(2 * 1024**2),
                                  viewer.ExportLimits(max_metadata_bytes=16))
    lines.begin_record()
    reader = csv.DictReader(lines)
    assert reader.fieldnames == ["sample_id", "scene"]
    for expected in ({"sample_id": "a", "scene": "b"}, {"sample_id": "c", "scene": "d"}):
        lines.begin_record()
        assert next(reader) == expected


def test_json_parser_reservation_precedes_read(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Parser must not run without transient headroom")
    monkeypatch.setattr(viewer, "_read_selected", forbidden)
    budget = viewer._RetainedMetadataBudget(1000)
    with pytest.raises(viewer.OfflineViewerError, match="JSON parser buffers"):
        budget.read_selected(tmp_path / "not-read.json", {}, viewer.ExportLimits())


def test_explicit_parser_budget_is_derived_from_remaining_not_original_limit(tmp_path, monkeypatch):
    observed = []

    def tiny_parser(path, selection, limits):
        observed.append(limits.max_metadata_bytes)
        return {"ok": True}
    monkeypatch.setattr(viewer, "_read_selected", tiny_parser)
    budget = viewer._RetainedMetadataBudget(16 * 1024**2)
    budget.charge(4 * 1024**2, "mock retained metadata")
    assert budget.read_selected(tmp_path / "unused", {}, viewer.ExportLimits()) == {"ok": True}
    assert observed[0] < viewer.ExportLimits().max_metadata_bytes
    assert budget.peak <= budget.limit


def test_budget_none_keeps_existing_parser_limit(tmp_path, monkeypatch):
    observed = []
    limits = viewer.ExportLimits()

    def tiny_parser(path, selection, actual):
        observed.append(actual)
        return {"ok": True}
    monkeypatch.setattr(viewer, "_read_selected", tiny_parser)
    budget = viewer._RetainedMetadataBudget(None)
    budget.read_selected(tmp_path / "unused", {}, limits)
    assert observed == [limits]
    assert budget.retained > 0
