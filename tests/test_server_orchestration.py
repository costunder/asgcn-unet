from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_full_script_covers_the_complete_ordered_experiment_matrix() -> None:
    script = _text("scripts/full.sh")

    preflight = script.index("[1/5]")
    inspection = script.index("[2/5]")
    training = script.index("[3/5]")
    calibration = script.index("[4/5]")
    evaluation = script.index("[5/5]")
    assert preflight < inspection < training < calibration < evaluation

    for config in (
        "configs/hdr_train.json",
        "configs/hdr_ann.json",
        "configs/hdr_snn.json",
        "configs/aid_ann.json",
        "configs/aid_snn.json",
    ):
        assert config in script
    assert 'SIMULATION_STEPS_LIST="${SIMULATION_STEPS_LIST:-4 8 16 32}"' in script
    assert "for dynamics in literal_eq15 standard_if" in script
    assert "RUN_BENCHMARK=1" in script
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'for config_path in "${TRAIN_CONFIG}" "${AID_ANN_CONFIG}"' in script


def test_calibration_wrapper_defaults_to_all_samples_and_protects_output() -> None:
    script = _text("scripts/calibrate.sh")
    assert 'CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-all}"' in script
    assert 'OVERWRITE_CALIBRATION="${OVERWRITE_CALIBRATION:-0}"' in script
    assert '--samples "${CALIBRATION_SAMPLES}"' in script
    assert "calibrated output already exists" in script
    assert "CALIBRATE_ARGS+=(--overwrite)" in script
    assert "rm -f" not in script


def test_all_wrappers_support_optional_validate_all_preflight() -> None:
    for relative in ("scripts/train.sh", "scripts/eval.sh", "scripts/calibrate.sh"):
        script = _text(relative)
        assert "INSPECT_VALIDATE_ALL" in script
        assert "INSPECT_ARGS+=(--validate-all)" in script


def test_calibration_has_slurm_and_pbs_entrypoints_with_dependency_examples() -> None:
    for relative in ("server/calibrate.sbatch", "server/calibrate.pbs"):
        script = _text(relative)
        assert script.startswith("#!/usr/bin/env bash\n")
        assert "scripts/calibrate.sh" in script
        assert "CALIBRATION_SAMPLES" in script
        assert "depend" in script


def test_eventaid_downloader_defaults_to_the_complete_release() -> None:
    script = _text("scripts/get_aid.sh")
    assert "the complete 14-scene release is downloaded" in script
    assert "SCENES=(R-bear)" not in script
    assert 'if ((DOWNLOAD_ALL == 0)) && ((${#SCENES[@]} == 0)); then\n  DOWNLOAD_ALL=1' in script
