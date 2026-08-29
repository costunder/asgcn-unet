from __future__ import annotations

import copy
import sys

import pytest
import torch

from asgcn_recon.engine import (
    _centralize_gradients,
    _clip_and_validate_gradients,
    _ensure_finite_loss,
    _model_state_sha256,
    _prediction_artifact_stem,
    _training_protocol,
    _validate_snn_request,
    _validate_training_protocol,
    load_model_checkpoint,
)
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from scripts import check_env


def _config() -> dict:
    return {
        "seed": 23,
        "train": {
            "epochs": 10,
            "batch_size": 1,
            "num_workers": 0,
            "amp": True,
            "learning_rate": 1e-3,
            "weight_decay": 5e-3,
            "grad_clip": 2.0,
            "max_train_samples": 20,
            "validate_every": 2,
            "log_every": 10,
            "loss_weights": {
                "charbonnier": 1.0,
                "ssim": 0.25,
                "gradient": 0.05,
                "temporal": 0.1,
            },
        },
    }


def test_training_protocol_captures_trajectory_but_allows_run_control_changes() -> None:
    config = _config()
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["optimizer"]["name"] == "AdamW"
    assert protocol["optimizer"]["learning_rate"] == pytest.approx(1e-3)
    assert protocol["mixed_precision"] == {
        "requested": True,
        "effective": False,
        "autocast_dtype": None,
        "gradient_scaler": False,
    }

    allowed = copy.deepcopy(config)
    allowed["train"].update({"epochs": 99, "log_every": 1, "resume": "/another/last.pt"})
    assert _training_protocol(allowed, torch.device("cpu")) == protocol

    changed = copy.deepcopy(config)
    changed["train"]["learning_rate"] = 2e-3
    with pytest.raises(ValueError, match=r"training protocol differs.*optimizer"):
        _validate_training_protocol(
            {"training_protocol": protocol},
            _training_protocol(changed, torch.device("cpu")),
        )


def test_paper_optimizer_mode_records_gc_and_milestone_schedule() -> None:
    config = _config()
    config["train"].update({"optimizer": "adam_gc", "lr_milestones": [8, 4], "lr_gamma": 0.2})
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["optimizer"]["name"] == "Adam"
    assert protocol["optimizer"]["gradient_centralization"] is True
    assert protocol["scheduler"] == {
        "name": "MultiStepLR",
        "milestones": [4, 8],
        "gamma": 0.2,
        "step_unit": "epoch",
        "step_timing": "after_epoch",
    }

    model = torch.nn.Linear(3, 2)
    model.weight.grad = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    model.bias.grad = torch.tensor([1.0, 2.0])
    original_bias = model.bias.grad.clone()
    _centralize_gradients(model)
    assert torch.allclose(model.weight.grad.mean(dim=1), torch.zeros(2))
    assert torch.equal(model.bias.grad, original_bias)


def test_exact_resume_rejects_checkpoint_without_training_protocol() -> None:
    with pytest.raises(ValueError, match="missing training_protocol"):
        _validate_training_protocol({}, _training_protocol(_config(), torch.device("cpu")))


def test_nonfinite_loss_components_and_gradients_fail_fast() -> None:
    with pytest.raises(FloatingPointError, match="total loss"):
        _ensure_finite_loss(
            torch.tensor(float("nan")),
            {"charbonnier": 1.0, "total": float("nan")},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )
    with pytest.raises(FloatingPointError, match="charbonnier"):
        _ensure_finite_loss(
            torch.tensor(1.0),
            {"charbonnier": float("inf"), "total": 1.0},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )

    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="gradients after clipping"):
        _clip_and_validate_gradients(model, 1.0, epoch=1, step=2, sample_id="sample-a")


def test_snn_requires_paper_core_parameter_normalization() -> None:
    checkpoint = {
        "checkpoint_type": "snn_inference",
        "batch_norm_folded": True,
        "snn_calibration_samples": 1,
        "paper_core_version": 2,
    }
    with pytest.raises(ValueError, match="parameter_normalized"):
        _validate_snn_request("snn", 4, checkpoint)
    checkpoint["parameter_normalized"] = True
    checkpoint["snn_calibration_valid_samples"] = 1
    _validate_snn_request("snn", 4, checkpoint)


def test_checkpoint_loader_rejects_unversioned_legacy_model(tmp_path) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": {}, "model_config": {}}, checkpoint)
    with pytest.raises(ValueError, match="architecture_version"):
        load_model_checkpoint(checkpoint, torch.device("cpu"), {})

    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model = ASGCNReconstructor(**model_config)
    mismatch = tmp_path / "mismatch.pt"
    torch.save(
        {"model": model.state_dict(), "model_config": model_config},
        mismatch,
    )
    changed = dict(model_config, recurrent=True)
    with pytest.raises(ValueError, match="model_config differs"):
        load_model_checkpoint(mismatch, torch.device("cpu"), changed)


def test_checkpoint_loader_cross_checks_conversion_metadata_and_layer_state(tmp_path) -> None:
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model = ASGCNReconstructor(**model_config)
    model_state = model.state_dict()
    path = tmp_path / "tampered.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "snn_inference",
            "model_config": model_config,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "batch_norm_folded": True,
            "parameter_normalized": True,
        },
        path,
    )
    with pytest.raises(ValueError, match="batch_norm_folded metadata disagrees"):
        load_model_checkpoint(path, torch.device("cpu"), model_config)


def test_checkpoint_loader_rejects_finite_model_tensor_tampering(tmp_path) -> None:
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model_state = ASGCNReconstructor(**model_config).state_dict()
    checkpoint = {
        "checkpoint_type": "ann_inference",
        "model_config": model_config,
        "model": model_state,
        "model_state_sha256": _model_state_sha256(model_state),
    }
    valid_path = tmp_path / "valid.pt"
    atomic_torch_save(checkpoint, valid_path)
    load_model_checkpoint(valid_path, torch.device("cpu"), model_config)

    tensor_name = next(name for name, value in model_state.items() if value.is_floating_point())
    model_state[tensor_name] = model_state[tensor_name].clone()
    model_state[tensor_name].view(-1)[0] += 0.25
    tampered_path = tmp_path / "finite-tampered.pt"
    atomic_torch_save(checkpoint, tampered_path)
    with pytest.raises(ValueError, match="does not match tensor bytes"):
        load_model_checkpoint(tampered_path, torch.device("cpu"), model_config)


def test_prediction_artifact_stems_are_cross_platform_safe_and_collision_resistant() -> None:
    first = _prediction_artifact_stem("a/b_c:CON?*", 3)
    second = _prediction_artifact_stem(r"a_b/c:CON?*\\tail", 3)
    repeated_at_other_index = _prediction_artifact_stem("a/b_c:CON?*", 4)

    assert first != second
    assert first != repeated_at_other_index
    assert first.startswith("00000003_")
    assert len(first) <= 86
    assert all(
        character.isascii() and (character.isalnum() or character in "._-") for character in first
    )


@pytest.mark.parametrize(
    ("flag", "required_key", "absent_keys"),
    [
        (
            "--require-eventhdr-smoke",
            "smoke manifest files",
            ("at least 51", "eventhdr_eval_h5", "eventaid_r_zip"),
        ),
        (
            "--require-eventhdr-train",
            "eventhdr_train_h5",
            ("eventhdr_eval_h5", "eventaid_r_zip"),
        ),
        (
            "--require-eventhdr-eval",
            "eventhdr_eval_h5",
            ("eventhdr_train_h5", "eventaid_r_zip"),
        ),
        (
            "--require-eventaid-all",
            "eventaid_r_zip",
            ("eventhdr_train_h5", "eventhdr_eval_h5"),
        ),
    ],
)
def test_check_env_dataset_requirements_are_independent(
    tmp_path, monkeypatch, flag: str, required_key: str, absent_keys: tuple[str, ...]
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(runs_root),
            flag,
        ],
    )
    with pytest.raises(SystemExit) as error:
        check_env.main()
    message = str(error.value)
    assert required_key in message
    assert all(key not in message for key in absent_keys)


def test_check_env_full_data_preserves_all_requirements(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-full-data",
        ],
    )
    with pytest.raises(SystemExit) as error:
        check_env.main()
    message = str(error.value)
    assert "eventhdr_train_h5" in message
    assert "eventhdr_eval_h5" in message
    assert "eventaid_r_zip" in message


def test_check_env_smoke_accepts_only_the_four_manifest_h5_files(tmp_path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    train_root = data_root / "EventHDR" / "train"
    train_root.mkdir(parents=True)
    for name in ("1.h5", "2.h5", "48.h5", "49.h5"):
        (train_root / name).touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(data_root),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-eventhdr-smoke",
        ],
    )
    check_env.main()
