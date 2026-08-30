from __future__ import annotations

import copy
import csv

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import asgcn_unet.engine as engine_module
from asgcn_unet.data import EventAidRZipDataset, collate_samples
from asgcn_unet.engine import (
    _continues_sequence,
    _dataset_index_contract,
    _dataset_sample_identity,
    _model_state_sha256,
    _sample_sequence_info,
    _sampling_summary,
    benchmark,
    evaluate,
    validate,
)
from asgcn_unet.model import ASGCNUNet
from asgcn_unet.utils import atomic_torch_save
from tests.fixtures import make_eventaid


class _PartDataset(Dataset):
    """Isolate engine part semantics from ZIP parsing using a temporary test fixture."""

    def __init__(self, root):
        make_eventaid(root, frames=5)
        self.base = EventAidRZipDataset(root, max_events=8)
        self.root = self.base.root
        self.zip_paths = self.base.zip_paths
        self.samples = []
        for index, record in enumerate(self.base.samples):
            part = index // 2
            self.samples.append(
                {
                    **record,
                    "scene": "R-traffic",
                    "sequence_id": f"R-traffic/part-{part:03d}",
                    "part_index": part,
                    # Global adjacent indices deliberately cannot reveal the part boundary.
                    "sequence_index": index,
                    "t0_us": 1_000_000 + index * 10_000,
                    "t1_us": 1_000_000 + (index + 1) * 10_000,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.base[index]
        record = self.samples[index]
        sample["sample_id"] = f"R-traffic/{index:06d}"
        sample["metadata"] = {
            **sample["metadata"],
            **{
                key: record[key]
                for key in ("scene", "sequence_id", "part_index", "sequence_index", "t0_us", "t1_us")
            },
        }
        return sample

    def close(self):
        self.base.close()


@pytest.fixture
def part_dataset(tmp_path):
    dataset = _PartDataset(tmp_path / "aid")
    yield dataset
    dataset.close()


def test_sequence_identity_prefers_declared_part_and_falls_back_to_scene():
    first = {
        "sensor_size": [32, 48],
        "metadata": {"scene": "R-traffic", "sequence_index": 7},
    }
    assert _sample_sequence_info(first) == ("R-traffic", 7, (32, 48))
    first["metadata"]["sequence_id"] = "R-traffic/part-000"
    second = copy.deepcopy(first)
    second["metadata"].update(sequence_id="R-traffic/part-001", sequence_index=8)
    assert not _continues_sequence(*_sample_sequence_info(second), *_sample_sequence_info(first))
    second["metadata"]["sequence_id"] = "R-traffic/part-000"
    assert _continues_sequence(*_sample_sequence_info(second), *_sample_sequence_info(first))


@pytest.mark.parametrize("key", ["sequence_id", "part_index", "t0_us", "t1_us"])
def test_sample_identity_binds_part_and_timestamp_mapping_without_changing_group(
    part_dataset, key
):
    record = part_dataset.samples[0]
    record["part_index"] = np.int64(record["part_index"])
    record["t0_us"] = np.int64(record["t0_us"])
    record["t1_us"] = np.float64(record["t1_us"])
    identity = _dataset_sample_identity(part_dataset, 0)
    assert identity["group"] == "R-traffic"
    assert identity["sequence_id"] == "R-traffic/part-000"
    assert type(identity["part_index"]) is int
    assert type(identity["t0_us"]) is int
    assert type(identity["t1_us"]) is float
    original = _dataset_index_contract(part_dataset)
    # Index identity does not depend on the working directory or the dataset root path.
    original_root = part_dataset.root
    part_dataset.root = original_root.parent
    assert _dataset_sample_identity(part_dataset, 0) == identity
    part_dataset.root = original_root
    assert _dataset_index_contract(part_dataset) == original
    record[key] = "R-traffic/part-099" if key == "sequence_id" else record[key] + 1
    changed = _dataset_index_contract(part_dataset)
    assert changed["sample_identities_sha256"] != original["sample_identities_sha256"]
    assert original["per_group"] == changed["per_group"] == {"R-traffic": 4}


def test_validation_evaluation_and_benchmark_reset_parts_but_keep_scene_metrics(
    part_dataset, tmp_path, monkeypatch
):
    model_config = {
        "architecture_version": 2,
        "hidden_dim": 2,
        "graph_layers": 1,
        "spline_kernel_size": 2,
        "decoder_channels": 4,
        "recurrent": True,
    }
    model = ASGCNUNet(**model_config)
    state_inputs = []
    original_forward = ASGCNUNet.forward_sample

    def record_state(self, sample, *args, **kwargs):
        state_inputs.append(kwargs.get("recurrent_state") is not None)
        return original_forward(self, sample, *args, **kwargs)

    monkeypatch.setattr(ASGCNUNet, "forward_sample", record_state)
    loader = DataLoader(part_dataset, batch_size=1, collate_fn=collate_samples)
    validation = validate(model, loader, torch.device("cpu"))
    assert state_inputs == [False, True, False, True]
    assert list(validation["per_scene"]) == ["R-traffic"]
    assert validation["per_scene"]["R-traffic"]["frames"] == 4
    assert validation["macro"] == validation["micro"]
    sampling = _sampling_summary(part_dataset, list(range(len(part_dataset))))
    assert sampling["per_group"] == {"R-traffic": 4}
    assert sampling["selected_groups"] == 1

    model_state = model.state_dict()
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
            "root": str(part_dataset.root),
            "target_offset": 1,
        },
        "model": model_config,
        "eval": {
            "num_workers": 0,
            "max_samples": None,
            "output_dir": str(tmp_path / "eval"),
            "precision": "fp32",
            "tf32": False,
        },
    }
    monkeypatch.setattr(engine_module, "build_dataset", lambda *_args, **_kwargs: part_dataset)
    state_inputs.clear()
    result = evaluate(config, checkpoint, allow_unsealed_checkpoint_for_non_reporting=True)
    assert state_inputs == [False, True, False, True]
    quality = result["quality"]
    assert list(quality["per_scene"]) == ["R-traffic"]
    assert quality["per_scene"]["R-traffic"]["frames"] == 4
    assert quality["per_scene"]["R-traffic"]["temporal_l1_frames"] == 2
    assert quality["macro"] == quality["micro"]
    with (tmp_path / "eval" / "ann" / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scene"] for row in rows] == ["R-traffic"] * 4
    assert [bool(row["temporal_l1"]) for row in rows] == [False, True, False, True]

    state_inputs.clear()
    timing = benchmark(
        config,
        checkpoint,
        warmup=0,
        steps=4,
        allow_unsealed_checkpoint_for_non_reporting=True,
    )
    assert state_inputs == [False, True, False, True]
    assert timing["state_resets"] == 2
    assert timing["state_reset_ratio"] == 0.5
