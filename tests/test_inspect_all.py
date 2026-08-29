from __future__ import annotations

import json

import pytest

from asgcn_recon.cli import inspect_dataset
from tests.fixtures import make_eventhdr


def test_inspect_validate_all_reads_every_sample_but_limits_preview(tmp_path) -> None:
    root = tmp_path / "hdr"
    make_eventhdr(root, frames=4)
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(root),
            "target_channels": 1,
            "max_events": 8,
        }
    }

    result = inspect_dataset(config, samples=1, validate_all=True)

    assert result["samples"] == 4
    assert result["validated_samples"] == 4
    assert result["validation_complete"] is True
    assert len(result["preview"]) == 1


def test_inspect_rejects_negative_preview_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        inspect_dataset({"dataset": {}}, samples=-1)


def test_inspect_uses_separate_validation_root(tmp_path) -> None:
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    train_path = make_eventhdr(train_root)
    val_path = make_eventhdr(val_root)
    train_path.rename(train_root / "train.h5")
    val_path.rename(val_root / "val.h5")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "final",
                "train_files": ["train.h5"],
                "val_files": ["val.h5"],
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "type": "eventhdr",
            "root": str(train_root),
            "val_root": str(val_root),
            "split_manifest": str(manifest),
            "max_events": 8,
        }
    }

    result = inspect_dataset(config, samples=1)

    assert result["splits"]["train"]["preview"][0]["metadata"]["source"].endswith(
        "train.h5"
    )
    assert result["splits"]["val"]["preview"][0]["metadata"]["source"].endswith(
        "val.h5"
    )
