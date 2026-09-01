from __future__ import annotations

import copy
import sys
import weakref

import pytest
import torch

import asgcn_unet.engine as engine_module
from asgcn_unet.engine import (
    _centralize_gradients,
    _clip_and_validate_gradients,
    _ensure_finite_loss,
    _inference_precision,
    _model_state_sha256,
    _prediction_artifact_stem,
    _require_finite_structure,
    _reset_benchmark_measurement_window,
    _training_protocol,
    _validate_snn_request,
    _validate_terminal_validation_resume,
    _validate_training_protocol,
    benchmark,
    evaluate,
    load_model_checkpoint,
)
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save, move_inference_sample, move_sample
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


class _TransferProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.device, bool]] = []

    def to(self, device: torch.device, *, non_blocking: bool = False):
        self.calls.append((device, non_blocking))
        return object()


def test_compute_transfer_keeps_target_on_host_while_quality_transfer_moves_it() -> None:
    device = torch.device("cuda")
    compute_events = _TransferProbe()
    compute_target = _TransferProbe()
    compute_sample = {"events": compute_events, "target": compute_target}

    moved_compute = move_inference_sample(compute_sample, device)

    assert moved_compute["events"] is not compute_events
    assert moved_compute["target"] is compute_target
    assert compute_events.calls == [(device, True)]
    assert compute_target.calls == []

    quality_events = _TransferProbe()
    quality_target = _TransferProbe()
    moved_quality = move_sample(
        {"events": quality_events, "target": quality_target},
        device,
    )

    assert moved_quality["events"] is not quality_events
    assert moved_quality["target"] is not quality_target
    assert quality_events.calls == [(device, True)]
    assert quality_target.calls == [(device, True)]


def test_benchmark_releases_warmup_and_cuda_cache_before_peak_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class WarmupReference:
        pass

    retained: list[WarmupReference] = [WarmupReference()]
    reference = weakref.ref(retained[0])

    def release() -> None:
        events.append("release")
        retained.clear()

    monkeypatch.setattr(
        engine_module.torch.cuda,
        "synchronize",
        lambda device: events.append("synchronize"),
    )
    monkeypatch.setattr(
        engine_module.gc,
        "collect",
        lambda: events.append("collect"),
    )

    def empty_cache() -> None:
        assert reference() is None
        events.append("empty_cache")

    monkeypatch.setattr(engine_module.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(
        engine_module.torch.cuda,
        "reset_peak_memory_stats",
        lambda device: events.append("reset_peak"),
    )

    _reset_benchmark_measurement_window(torch.device("cuda"), release)

    assert events == [
        "synchronize",
        "release",
        "collect",
        "empty_cache",
        "reset_peak",
    ]


@pytest.mark.parametrize("entrypoint", [evaluate, benchmark])
def test_evaluation_entrypoints_close_dataset_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
) -> None:
    class TrackingDataset:
        def __init__(self) -> None:
            self.closed = False

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            raise AssertionError(f"unexpected dataset read: {index}")

        def close(self) -> None:
            self.closed = True

    dataset = TrackingDataset()
    monkeypatch.setattr(engine_module, "build_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        engine_module,
        "load_model_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )
    config = {"device": "cpu", "dataset": {}, "model": {}, "eval": {}}

    with pytest.raises(RuntimeError, match="setup failed"):
        if entrypoint is benchmark:
            entrypoint(config, "checkpoint.pt", warmup=0, steps=1)
        else:
            entrypoint(config, "checkpoint.pt")

    assert dataset.closed is True


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
        "overflow_policy": None,
    }
    assert len(protocol["source"]["source_tree_sha256"]) == 64
    assert protocol["runtime"]["gpu_name"] is None
    assert protocol["runtime"]["compute_capability"] is None
    assert protocol["version"] == 7

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


def test_training_protocol_can_reserve_validation_for_the_final_epoch() -> None:
    config = _config()
    config["train"]["validate_every"] = None
    protocol = _training_protocol(config, torch.device("cpu"))
    assert protocol["validate_every"] is None
    assert protocol["checkpoint_selection"] == "single_final_epoch"
    assert protocol["terminal_validation"] == {
        "mode": "single_final_epoch",
        "planned_epoch": 10,
    }

    extended = copy.deepcopy(config)
    extended["train"]["epochs"] = 11
    extended_protocol = _training_protocol(extended, torch.device("cpu"))
    with pytest.raises(ValueError, match="already completed"):
        _validate_terminal_validation_resume(
            {
                "training_protocol": protocol,
                "terminal_validation_state": {
                    "planned_epoch": 10,
                    "completed": True,
                    "completed_epoch": 10,
                },
            },
            extended_protocol,
        )


def test_inference_precision_and_finite_artifact_guards_are_explicit() -> None:
    model = torch.nn.Linear(2, 1)
    precision, dtype = _inference_precision(
        {"precision": "fp32", "tf32": False}, torch.device("cpu"), model
    )
    assert dtype is None
    assert precision == {
        "requested": "fp32",
        "effective": "fp32",
        "autocast_dtype": None,
        "model_parameter_dtype": "float32",
        "device": "cpu",
        "tf32": False,
        "tf32_requested": False,
    }
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _inference_precision(
            {"precision": "amp_fp16"}, torch.device("cpu"), model
        )
    with pytest.raises(FloatingPointError, match=r"metrics\.nested"):
        _require_finite_structure(
            {"nested": [1.0, float("nan")]}, "metrics", "sample-a"
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
            {"charbonnier": torch.tensor(1.0)},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )
    with pytest.raises(FloatingPointError, match="charbonnier"):
        _ensure_finite_loss(
            torch.tensor(1.0),
            {"charbonnier": torch.tensor(float("inf"))},
            epoch=1,
            step=2,
            sample_id="sample-a",
        )

    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="gradients before clipping"):
        _clip_and_validate_gradients(model, 1.0, epoch=1, step=2, sample_id="sample-a")


def test_reconstruction_loss_keeps_components_on_device_until_packed_validation() -> None:
    prediction = torch.rand((1, 1, 8, 8), requires_grad=True)
    target = torch.rand_like(prediction)
    total, parts = ReconstructionLoss()(prediction, target)

    assert all(isinstance(value, torch.Tensor) for value in parts.values())
    assert all(value.device == prediction.device for value in parts.values())
    assert all(not value.requires_grad for value in parts.values())
    values = _ensure_finite_loss(
        total,
        parts,
        epoch=1,
        step=1,
        sample_id="packed",
    )
    assert set(values) == {"total", "charbonnier", "ssim", "gradient"}
    assert all(isinstance(value, float) for value in values.values())


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
    model = ASGCNUNet(**model_config)
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
    model = ASGCNUNet(**model_config)
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
    model_state = ASGCNUNet(**model_config).state_dict()
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


@pytest.mark.parametrize(
    ("flag", "subdirectory", "expected_count"),
    [
        ("--require-eventhdr-train", "train", 51),
        ("--require-eventhdr-eval", "eval", 19),
    ],
)
def test_check_env_requires_exact_official_eventhdr_names(
    tmp_path, monkeypatch, flag: str, subdirectory: str, expected_count: int
) -> None:
    root = tmp_path / "data" / "EventHDR" / subdirectory
    root.mkdir(parents=True)
    for index in range(1, expected_count):
        (root / f"{index}.h5").touch()
    (root / f"{expected_count + 1}.h5").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            flag,
        ],
    )

    with pytest.raises(SystemExit) as error:
        check_env.main()

    message = str(error.value)
    assert f"missing={expected_count}.h5" in message
    assert f"extra={expected_count + 1}.h5" in message


def test_check_env_accepts_exact_official_eventhdr_names(tmp_path, monkeypatch) -> None:
    train_root = tmp_path / "data" / "EventHDR" / "train"
    eval_root = tmp_path / "data" / "EventHDR" / "eval"
    train_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    for index in range(1, 52):
        (train_root / f"{index}.h5").touch()
    for index in range(1, 20):
        (eval_root / f"{index}.h5").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_env.py",
            "--data-root",
            str(tmp_path / "data"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--require-eventhdr-train",
            "--require-eventhdr-eval",
        ],
    )

    check_env.main()
