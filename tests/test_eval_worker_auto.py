"""CPU routing regression, not a GPU throughput or quality measurement."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from asgcn_unet import engine
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
from tests.fixtures import make_eventhdr
from tests.test_p0_engine import _eval_config


def test_eval_worker_auto_profiles_workers_without_changing_explicit_physical_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "debug-hdr"
    fixture = make_eventhdr(root)
    shutil.copy2(fixture, root / "second.h5")
    config = _eval_config(root, tmp_path / "debug-eval")
    config["eval"].update(
        batch_size=2,
        num_workers="auto",
        batch_candidates=[1, 2, 4],
        worker_candidates=[0, 1],
        max_samples=None,
        profile_debug_cpu=True,
    )
    model_state = ASGCNUNet(**config["model"]).state_dict()
    checkpoint = tmp_path / "untrained-cpu-unit-fixture.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 0,
            "model": model_state,
            "model_state_sha256": engine._model_state_sha256(model_state),
            "model_config": config["model"],
            "paper_core_version": 2,
        },
        checkpoint,
    )
    selections = []

    def routing_only_profile(dataset, model, device, *, section, **kwargs):
        assert device.type == "cpu"
        assert section["batch_candidates"] == [2]
        assert section["num_workers"] == "auto"
        assert callable(kwargs["run_batch"])
        assert callable(kwargs["loader_factory"])
        selections.append(len(dataset))
        return {
            "batch_size": 2,
            "num_workers": 0,
            "report": {
                "debug_cpu": True,
                "cuda_measured": False,
                "report_eligible": False,
                "report_ineligible_reasons": [
                    "routing-only CPU unit stub; no throughput was measured"
                ],
            },
        }

    monkeypatch.setattr(engine, "profile_inference_batches", routing_only_profile)
    result = engine.evaluate(
        config,
        checkpoint,
        inference_mode="ann",
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert selections == [8]
    assert result["quality"]["frames"] == 8
    assert result["execution"]["batching"]["physical_batch_size"] == 2
    assert result["execution"]["loader"]["num_workers"] == 0
    assert result["performance"]["physical_batch_histogram"] == {"2": 4}
    assert result["batch_profile"]["precision"]["effective"] == "fp32"
    assert result["report_eligible"] is False
    assert config["eval"]["batch_size"] == 2
    assert config["eval"]["num_workers"] == "auto"
