from __future__ import annotations

import copy
import json
import socket

import pytest
import torch

import asgcn_unet.engine as engine_module
from asgcn_unet.data import EventAidRZipDataset, EventHDRDataset
from asgcn_unet.engine import (
    _canonical_sha256,
    _dataset_content_fingerprint,
    _file_sha256,
    _model_state_sha256,
    _sampling_summary,
    _training_protocol,
    _validation_protocol,
    benchmark,
    evaluate,
)
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
from scripts import eval_resume
from tests.fixtures import make_eventaid, make_eventhdr


def _verified_preflight_gate() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema": "asgcn_preflight_verification_v1",
        "status": "verified",
        "report_eligible": True,
        "report": "runs/profile.json",
        "report_sha256": digest,
        "measurement_scope": {
            "name": "selected_top_density_training_steps",
            "topology_scope": "complete_eventhdr_training_split",
            "absolute_vram_guarantee": False,
            "statement": "Empirical GPU preflight for the recorded runtime.",
        },
        "config_sha256": digest,
        "data_sha256": digest,
        "source_tree_sha256": digest,
        "gpu": {"name": "synthetic-test-gpu"},
        "measured_steps": 1,
    }


def test_train_eval_benchmark_contract(tmp_path):
    """Keep the full pipeline check test-only and confined to pytest's temp tree."""
    hdr_root = tmp_path / "hdr"
    aid_root = tmp_path / "aid"
    make_eventhdr(hdr_root)
    make_eventaid(aid_root)

    hdr = EventHDRDataset(hdr_root, max_events=32)
    aid = EventAidRZipDataset(aid_root, max_events=32)
    assert len(hdr) == 4
    assert len(aid) == 3

    model_config = {
        "architecture_version": 2,
        "graph_operator": "spline",
        "spline_backend": "torch",
        "spline_pseudo": "distance_over_radius",
        "spline_is_open": True,
        "hidden_dim": 8,
        "graph_layers": 2,
        "event_sampling_factor": 1,
        "graph_radius": 2.0,
        "graph_position_dims": 3,
        "graph_chunk_size": 16,
        "spline_kernel_size": 3,
        "spline_degree": 1,
        "spline_root_weight": True,
        "raster_downsample": 4,
        "decoder_channels": 4,
        "output_channels": 1,
        "recurrent": True,
    }
    model = ASGCNUNet(**model_config)
    sample = hdr[0]
    prediction, diagnostics = model.forward_sample(sample)
    loss, _ = ReconstructionLoss()(prediction, sample["target"].unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["nodes"] == 32

    checkpoint = tmp_path / "model.pt"
    model_state = model.state_dict()
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "epoch": 0,
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
        },
        checkpoint,
    )
    output_dir = tmp_path / "eval"
    config = {
        "seed": 7,
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": str(aid_root),
            "target_channels": 1,
            "max_events": 32,
            "crop_size": None,
            "target_offset": 1,
            "tone_map": "none",
        },
        "model": model_config,
        "eval": {
            "num_workers": 0,
            "max_samples": 2,
            "save_predictions": 1,
            "output_dir": str(output_dir),
            "precision": "fp32",
            "tf32": False,
        },
    }
    with pytest.raises(ValueError, match="Checkpoint reporting protocol is not sealed"):
        evaluate(config, checkpoint)
    with pytest.raises(ValueError, match="Checkpoint reporting protocol is not sealed"):
        benchmark(config, checkpoint, warmup=1, steps=2)
    result = evaluate(
        config,
        checkpoint,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    timing = benchmark(
        config,
        checkpoint,
        warmup=1,
        steps=2,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )

    assert result["quality"]["frames"] == 2
    assert timing["frames"] == 2
    assert timing["mean_raw_events"] == 80
    assert timing["mean_retained_events"] == 32
    assert timing["retention_ratio"] == 32 / 80
    assert timing["raw_events_per_second"] > timing["retained_events_per_second"]
    assert timing["graph_nodes_per_second"] == timing["retained_events_per_second"]
    assert timing["events_per_second"] == timing["retained_events_per_second"]
    assert timing["recurrent_context_frames"] == 1
    assert timing["state_resets"] == 0
    assert timing["state_reset_ratio"] == 0.0
    assert result["precision"] == timing["precision"]
    assert result["precision"]["requested"] == "fp32"
    assert result["precision"]["model_parameter_dtype"] == "float32"
    assert result["report_eligible"] is False
    assert timing["report_eligible"] is False
    assert result["report_ineligible_reasons"][0] == (
        "explicit non-reporting checkpoint override requested"
    )
    assert result["checkpoint"] == "$EXTERNAL/model.pt"
    assert result["output_dir"] == "$EXTERNAL/ann"
    assert timing["output_path"] == "$EXTERNAL/benchmark.json"
    ann_output = output_dir / "ann"
    assert (ann_output / "metrics.json").is_file()
    assert (ann_output / "frames.csv").is_file()
    assert (ann_output / "benchmark.json").is_file()
    assert len(list((ann_output / "predictions").glob("*_pred.png"))) == 1

    hdr.close()
    aid.close()


def test_sealed_ann_reporting_protocol_is_complete_redacted_and_reuses_data_hash_cache(
    tmp_path, monkeypatch
) -> None:
    aid_root = tmp_path / "private-mount" / "aid"
    make_eventaid(aid_root)
    aid_manifest_path = tmp_path / "aid-files.json"
    aid_manifest_path.write_text(
        json.dumps({"files": [{"scene": "R-test"}]}),
        encoding="utf-8",
    )
    train_root = tmp_path / "hdr-train"
    val_root = tmp_path / "hdr-val"
    make_eventhdr(train_root)
    make_eventhdr(val_root)
    split_manifest_path = tmp_path / "eventhdr-split.json"
    split_manifest_path.write_text(
        json.dumps(
            {
                "status": "final",
                "split_schema": "official_separate_roots_v1",
                "group_semantics": "h5_sequence_file_not_physical_scene",
                "train_files": ["test.h5"],
                "val_files": ["test.h5"],
            }
        ),
        encoding="utf-8",
    )
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    output_dir = tmp_path / "report-eval"
    config = {
        "seed": 11,
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": str(aid_root),
            "file_manifest": str(aid_manifest_path),
            "expected_file_count": 1,
            "target_offset": 1,
        },
        "model": model_config,
        "eval": {
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": None,
            "output_dir": str(output_dir),
            "precision": "fp32",
            "tf32": False,
        },
    }
    training_config_input = {
        "seed": 11,
        "device": "cpu",
        "dataset": {
            "type": "eventhdr",
            "root": str(train_root),
            "val_root": str(val_root),
            "split_manifest": str(split_manifest_path),
        },
        "model": model_config,
        "train": {
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "amp": False,
            "validate_every": None,
            "max_train_samples": None,
            "max_val_samples": None,
            "validation_context_frames": 0,
            "loss_weights": None,
        },
    }
    train_dataset = engine_module.build_dataset(
        training_config_input["dataset"], split="train"
    )
    val_dataset = engine_module.build_dataset(
        training_config_input["dataset"], split="val"
    )
    try:
        training_content = _dataset_content_fingerprint(train_dataset)
        sample_count = len(val_dataset)
        sampling = _sampling_summary(
            val_dataset, list(range(len(val_dataset)))
        )
        sampling.update(
            {
                "context_policy": "none_non_recurrent",
                "max_context_frames_per_group": 0,
                "context_samples": 0,
                "forward_samples": sample_count,
            }
        )
        validation_protocol = _validation_protocol(
            training_config_input,
            sampling,
            train_dataset,
            val_dataset,
            {},
        )
    finally:
        train_dataset.close()
        val_dataset.close()
    evaluation_dataset = EventAidRZipDataset(aid_root)
    try:
        content = _dataset_content_fingerprint(evaluation_dataset)
    finally:
        evaluation_dataset.close()
    model_state = ASGCNUNet(**model_config).state_dict()
    training_protocol = _training_protocol(
        training_config_input, torch.device("cpu")
    )
    training_config = engine_module._public_config(training_config_input)
    preflight_gate = _verified_preflight_gate()
    preflight_gate["config_sha256"] = _canonical_sha256(training_config)
    preflight_gate["data_sha256"] = training_content["sha256"]
    preflight_gate["source_tree_sha256"] = training_protocol["source"][
        "source_tree_sha256"
    ]
    checkpoint = tmp_path / "sealed-ann.pt"
    checkpoint_payload = {
        "checkpoint_type": "ann_inference",
        "epoch": 1,
        "model": model_state,
        "model_state_sha256": _model_state_sha256(model_state),
        "model_config": model_config,
        "paper_core_version": 2,
        "val": {"frames": sample_count, "macro": {"ssim": 0.75}},
        "best_ssim": 0.75,
        "best_metric": "macro_ssim",
        "checkpoint_selection": "single_final_epoch",
        "preflight_gate": preflight_gate,
        "training_config": training_config,
        "validation_protocol": validation_protocol,
        "training_protocol": training_protocol,
        "terminal_validation_state": {
            "planned_epoch": 1,
            "completed": True,
            "completed_epoch": 1,
        },
    }
    atomic_torch_save(checkpoint_payload, checkpoint)
    for field, message in (
        ("data_sha256", "training data"),
        ("source_tree_sha256", "training source"),
        ("config_sha256", "training config"),
    ):
        mismatched_payload = copy.deepcopy(checkpoint_payload)
        mismatched_payload["preflight_gate"][field] = "f" * 64
        mismatched_checkpoint = tmp_path / f"mismatched-{field}.pt"
        atomic_torch_save(mismatched_payload, mismatched_checkpoint)
        with pytest.raises(ValueError, match=message):
            evaluate(config, mismatched_checkpoint)
    bypassed_checkpoint = tmp_path / "bypassed-ann.pt"
    atomic_torch_save(
        {
            **checkpoint_payload,
            "preflight_gate": {
                "schema": "asgcn_preflight_verification_v1",
                "status": "bypassed_non_reporting",
                "report_eligible": False,
                "report": "profile.json",
                "warning": "synthetic bypass",
            },
        },
        bypassed_checkpoint,
    )
    with pytest.raises(ValueError, match="preflight gate"):
        evaluate(config, bypassed_checkpoint)

    missing_count_config = copy.deepcopy(config)
    missing_count_config["dataset"].pop("expected_file_count")
    missing_count_config["eval"]["output_dir"] = str(tmp_path / "missing-count")
    with pytest.raises(ValueError, match="expected-file-count commitment"):
        evaluate(missing_count_config, checkpoint)

    missing_manifest_config = copy.deepcopy(config)
    missing_manifest_config["dataset"].pop("file_manifest")
    missing_manifest_config["eval"]["output_dir"] = str(tmp_path / "missing-manifest")
    with pytest.raises(ValueError, match="fixed file manifest"):
        evaluate(missing_manifest_config, checkpoint)

    original_content_fingerprint = engine_module._dataset_content_fingerprint
    cache_entry_counts: list[int] = []

    def tracked_content_fingerprint(dataset, cache=None):
        if cache is not None:
            cache_entry_counts.append(len(cache))
        return original_content_fingerprint(dataset, cache)

    monkeypatch.setattr(
        engine_module,
        "_dataset_content_fingerprint",
        tracked_content_fingerprint,
    )
    result = evaluate(config, checkpoint)
    timing = benchmark(config, checkpoint, warmup=0, steps=1)

    resume_config = tmp_path / "resume-eval-config.json"
    resume_config.write_text(json.dumps(config), encoding="utf-8")
    resume_args = eval_resume._parser().parse_args(
        [
            "--config", str(resume_config),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--inference-mode", "ann",
            "--simulation-steps", "16",
            "--benchmark-warmup", "0",
            "--benchmark-steps", "1",
        ]
    )
    assert eval_resume.inspect_mode(resume_args) == (1, 1)

    assert result["report_eligible"] is True
    assert timing["report_eligible"] is True
    assert result["report_ineligible_reasons"] == []
    assert cache_entry_counts[0] == 0
    assert cache_entry_counts[1] >= 1
    assert result["evaluation_protocol"]["evaluation_dataset"]["contract"]["content"] == content
    assert timing["benchmark_protocol"]["evaluation_dataset"]["contract"]["content"] == content

    for protocol_name, protocol in (
        ("evaluation_protocol", result["evaluation_protocol"]),
        ("benchmark_protocol", timing["benchmark_protocol"]),
    ):
        assert protocol["schema"] == "asgcn_reporting_protocol_v1"
        assert len(protocol["public_config"]["sha256"]) == 64
        assert len(protocol["model_config"]["sha256"]) == 64
        assert len(protocol["source"]["sha256"]) == 64
        assert len(protocol["runtime"]["sha256"]) == 64
        assert len(protocol["precision"]["sha256"]) == 64
        assert len(protocol["execution"]["sha256"]) == 64
        assert protocol["execution"]["contract"]["inference_mode"] == "ann"
        assert protocol["execution"]["contract"]["simulation_steps"] is None
        assert protocol["execution"]["contract"]["snn_dynamics"] is None
        assert protocol["checkpoint"]["checkpoint_file_sha256"] == _file_sha256(checkpoint)
        assert protocol["checkpoint"]["model_state_sha256"] == _model_state_sha256(
            model_state
        )
        assert len(protocol["checkpoint"]["training_protocol"]["sha256"]) == 64
        assert len(protocol["checkpoint"]["validation_protocol"]["sha256"]) == 64
        unsigned = dict(protocol)
        digest = unsigned.pop("protocol_sha256")
        assert digest == _canonical_sha256(unsigned), protocol_name
        public_text = json.dumps(protocol, ensure_ascii=False, allow_nan=False)
        assert str(tmp_path) not in public_text
        machine_name = socket.gethostname()
        if machine_name:
            assert machine_name not in public_text

    cache_text = (output_dir / ".data_hash_cache.json").read_text(encoding="utf-8")
    assert str(aid_root) not in cache_text
    cache_payload = json.loads(cache_text, parse_constant=lambda value: pytest.fail(value))
    assert cache_payload["version"] == 1
    for artifact in (
        output_dir / "ann" / "metrics.json",
        output_dir / "ann" / "benchmark.json",
    ):
        json.loads(
            artifact.read_text(encoding="utf-8"),
            parse_constant=lambda value: pytest.fail(f"non-finite JSON token: {value}"),
        )


def test_standalone_evaluate_and_benchmark_reject_nonfinite_predictions(
    tmp_path, monkeypatch
) -> None:
    aid_root = tmp_path / "aid"
    make_eventaid(aid_root)
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": False,
    }
    model_state = ASGCNUNet(**model_config).state_dict()
    checkpoint = tmp_path / "model.pt"
    atomic_torch_save(
        {
            "checkpoint_type": "ann_inference",
            "model": model_state,
            "model_state_sha256": _model_state_sha256(model_state),
            "model_config": model_config,
        },
        checkpoint,
    )
    config = {
        "device": "cpu",
        "dataset": {
            "type": "eventaid_r_zip",
            "root": str(aid_root),
            "target_offset": 1,
        },
        "model": model_config,
        "eval": {
            "num_workers": 0,
            "max_samples": 1,
            "output_dir": str(tmp_path / "eval"),
            "precision": "fp32",
            "tf32": False,
        },
    }
    original_forward = ASGCNUNet.forward_sample

    def poisoned_forward(self, *args, **kwargs):
        prediction, diagnostics = original_forward(self, *args, **kwargs)
        return torch.full_like(prediction, float("nan")), diagnostics

    monkeypatch.setattr(ASGCNUNet, "forward_sample", poisoned_forward)
    with pytest.raises(FloatingPointError, match="Non-finite prediction"):
        evaluate(
            config,
            checkpoint,
            allow_unsealed_checkpoint_for_non_reporting=True,
        )
    with pytest.raises(FloatingPointError, match="Non-finite prediction"):
        benchmark(
            config,
            checkpoint,
            warmup=0,
            steps=1,
            allow_unsealed_checkpoint_for_non_reporting=True,
        )
