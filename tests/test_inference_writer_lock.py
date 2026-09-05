"""CPU-only control-flow tests: an existing writer must prevent inference work."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from asgcn_unet import engine
from asgcn_unet.artifact_lock import ArtifactWriterBusyError, exclusive_artifact_writer


@pytest.mark.parametrize("name", ["_evaluate_dataset", "_benchmark_dataset"])
@pytest.mark.parametrize(
    ("mode", "steps", "dynamics", "label"),
    [
        ("ann", 32, None, "ann"),
        ("snn", 4, "literal_eq15", "snn_literal_eq15_T4"),
        ("snn", 8, "standard_if", "snn_standard_if_T8"),
    ],
)
def test_existing_writer_blocks_direct_entry_before_model_or_output_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    mode: str,
    steps: int | None,
    dynamics: str | None,
    label: str,
) -> None:
    function = getattr(engine, name)
    assert hasattr(function, "__wrapped__"), "The direct entry point must acquire its own lock"
    config = {
        "device": "cpu",
        "model": {"snn_dynamics": "literal_eq15"},
        "eval": {"output_dir": str(tmp_path / "evaluation")},
    }
    # Required non-routing arguments are deliberately unusable: the underlying
    # function must never be entered while another writer owns the mode.
    arguments = {
        parameter.name: None
        for parameter in inspect.signature(function).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    arguments.update(
        config=config,
        inference_mode=mode,
        simulation_steps=steps,
        snn_dynamics=dynamics,
    )
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("inference_work")
        raise AssertionError("A locked mode must not resolve a GPU or load its model")

    monkeypatch.setattr(engine, "resolve_device", forbidden)
    monkeypatch.setattr(engine, "load_model_checkpoint", forbidden)
    output = Path(config["eval"]["output_dir"]) / label
    with exclusive_artifact_writer(output):
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        with pytest.raises(ArtifactWriterBusyError):
            function(**arguments)
        after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        assert after == before
    assert calls == []
    assert not (output / "metrics.json").exists()
    assert not (output / "benchmark.json").exists()
