from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from asgcn_unet.data import EventAidRZipDataset, EventHDRDataset
from asgcn_unet.data.common import image_array_to_tensor, validate_target_normalization
from asgcn_unet.losses import ReconstructionLoss
from asgcn_unet.utils import validate_experiment_config
from tests.fixtures import make_eventaid, make_eventhdr


def _append_zip_member(path: Path, name: str, content: bytes = b"duplicate\n") -> None:
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)


def _replace_zip_member(path: Path, member: str, content: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    with zipfile.ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as destination:
        for info, original in entries:
            destination.writestr(info.filename, content if info.filename == member else original)
    temporary.replace(path)


def test_integer_target_normalization_preserves_dtype_max_contract() -> None:
    image = np.array([[0, 32768, 65535]], dtype=np.uint16)
    target = image_array_to_tensor(
        image,
        target_normalization={"mode": "integer_dtype_max"},
    )
    torch.testing.assert_close(
        target,
        torch.tensor([[[0.0, 32768 / 65535, 1.0]]]),
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_float_target_rejects_nonfinite_values(invalid: float) -> None:
    image = np.array([[0.0, invalid]], dtype=np.float32)
    with pytest.raises(ValueError, match="NaN or Inf"):
        image_array_to_tensor(
            image,
            target_normalization={"mode": "already_normalized"},
            source="fixture",
        )


def test_eventhdr_loader_does_not_hide_nonfinite_float_target(tmp_path: Path) -> None:
    path = make_eventhdr(tmp_path / "eventhdr")
    with h5py.File(path, "a") as archive:
        source = archive["images/image000000000"]
        attributes = dict(source.attrs)
        image = np.asarray(source, dtype=np.float32) / 65535.0
        image[0, 0] = np.nan
        del archive["images/image000000000"]
        target = archive["images"].create_dataset("image000000000", data=image)
        for name, value in attributes.items():
            target.attrs[name] = value
    dataset = EventHDRDataset(
        path.parent,
        target_normalization={"mode": "already_normalized"},
    )
    try:
        with pytest.raises(ValueError, match="NaN or Inf"):
            dataset[0]
    finally:
        dataset.close()


def test_target_normalization_rejects_ambiguous_dtype_and_range() -> None:
    with pytest.raises(ValueError, match="requires an integer dtype"):
        image_array_to_tensor(
            np.array([[0.0, 1.0]], dtype=np.float32),
            target_normalization={"mode": "integer_dtype_max"},
        )
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        image_array_to_tensor(
            np.array([[0.0, 1.01]], dtype=np.float32),
            target_normalization={"mode": "already_normalized"},
        )
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        image_array_to_tensor(
            np.array([[0.0, 101.0]], dtype=np.float32),
            target_normalization={"mode": "known_scale", "scale": 100.0},
        )


def test_known_scale_is_explicit_and_percentile_is_debug_only() -> None:
    target = image_array_to_tensor(
        np.array([[0.0, 50.0, 100.0]], dtype=np.float32),
        target_normalization={"mode": "known_scale", "scale": 100.0},
    )
    torch.testing.assert_close(target, torch.tensor([[[0.0, 0.5, 1.0]]]))

    with pytest.raises(ValueError, match="debug_only=true"):
        validate_target_normalization({"mode": "percentile_debug_only"})
    debug = image_array_to_tensor(
        np.array([[0.0, 50.0, 100.0]], dtype=np.float32),
        target_normalization={
            "mode": "percentile_debug_only",
            "debug_only": True,
            "lower_percentile": 0.0,
            "upper_percentile": 100.0,
        },
    )
    torch.testing.assert_close(debug, torch.tensor([[[0.0, 0.5, 1.0]]]))


@pytest.mark.parametrize("tone_map_mu", [0.0, -1.0, float("nan"), float("inf")])
def test_log_tone_map_requires_a_finite_positive_mu(tone_map_mu: float) -> None:
    with pytest.raises(ValueError, match="tone_map_mu"):
        image_array_to_tensor(
            np.array([[0, 255]], dtype=np.uint8),
            tone_map="log",
            tone_map_mu=tone_map_mu,
            target_normalization={"mode": "integer_dtype_max"},
        )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("alternate/event/1.txt", "duplicate numeric event ID"),
        ("alternate/gt/2_img.png", "duplicate numeric GT ID"),
        ("nested/timestamps.txt", "duplicate timestamps.txt"),
        ("nested/shape.txt", "duplicate shape.txt"),
        ("EVENT/000001.TXT", "case-insensitive duplicate ZIP member"),
    ],
)
def test_eventaid_rejects_logically_duplicate_zip_members(
    tmp_path: Path, name: str, message: str
) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _append_zip_member(path, name)

    with pytest.raises(ValueError, match=message):
        EventAidRZipDataset(path.parent)


@pytest.mark.parametrize(
    "content",
    [
        b"48\n",
        b"48 32 extra\n",
        b"48.0 32\n",
        b"48 zero\n",
        b"0 32\n",
        b"48 -1\n",
    ],
)
def test_eventaid_rejects_malformed_shape(tmp_path: Path, content: bytes) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "shape.txt", content)

    with pytest.raises(ValueError, match="Invalid EventAid-R shape"):
        EventAidRZipDataset(path.parent)


def test_eventaid_rejects_duplicate_timestamp_values(tmp_path: Path) -> None:
    path = make_eventaid(tmp_path / "eventaid")
    _replace_zip_member(path, "timestamps.txt", b"100\n200\n200\n300\n")

    with pytest.raises(ValueError, match="strictly increasing"):
        EventAidRZipDataset(path.parent)


@pytest.mark.parametrize("target_offset", [True, False, 1.5, "1"])
def test_eventaid_target_offset_requires_an_exact_integer(
    tmp_path: Path, target_offset: object
) -> None:
    path = make_eventaid(tmp_path / "eventaid")

    with pytest.raises(TypeError, match="target_offset must be an integer"):
        EventAidRZipDataset(path.parent, target_offset=target_offset)  # type: ignore[arg-type]


def test_eventhdr_orders_image_keys_by_numeric_suffix(tmp_path: Path) -> None:
    path = make_eventhdr(tmp_path / "eventhdr")
    with h5py.File(path, "a") as archive:
        images = archive["images"]
        images.move("image000000003", "image20")
        images.move("image000000002", "image10")
        images.move("image000000001", "image2")

    dataset = EventHDRDataset(path.parent)
    try:
        keys = [sample["image_key"] for sample in dataset.samples]
        assert keys == ["image000000000", "image2", "image10", "image20"]
    finally:
        dataset.close()


def test_eventhdr_rejects_duplicate_numeric_image_suffix(tmp_path: Path) -> None:
    path = make_eventhdr(tmp_path / "eventhdr")
    with h5py.File(path, "a") as archive:
        archive["images"]["image1"] = archive["images/image000000001"]

    with pytest.raises(ValueError, match="duplicates numeric image index"):
        EventHDRDataset(path.parent)


@pytest.mark.parametrize(
    "weights",
    [
        {"charbonier": 1.0},
        {"ssim": -0.1},
        {"gradient": float("nan")},
        {"gradient": float("inf")},
        {"charbonnier": True},
    ],
)
def test_reconstruction_loss_rejects_invalid_weights(weights: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReconstructionLoss(weights)


def test_reconstruction_loss_accepts_separately_consumed_temporal_weight() -> None:
    loss = ReconstructionLoss({"temporal": 0.2})
    assert "temporal" not in loss.weights


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"train": {"batch_size": 2}}, "train.batch_size"),
        ({"train": {"max_train_samples": 0}}, "train.max_train_samples"),
        ({"train": {"max_val_samples": -1}}, "train.max_val_samples"),
        ({"eval": {"batch_size": 2}}, "eval.batch_size"),
        ({"eval": {"max_samples": 0}}, "eval.max_samples"),
        ({"eval": {"precision": "fp16"}}, "eval.precision"),
        ({"eval": {"tf32": 1}}, "eval.tf32"),
        ({"dataset": {"target_offset": True}}, "dataset.target_offset"),
        ({"dataset": {"target_offset": 1.5}}, "dataset.target_offset"),
    ],
)
def test_experiment_config_rejects_invalid_sample_contracts(config: dict, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_experiment_config(config)


def test_checked_in_configs_declare_target_and_inference_precision() -> None:
    import json

    repository = Path(__file__).resolve().parents[1]
    for name in ("train", "hdr", "aid"):
        config = json.loads((repository / "configs" / f"{name}.json").read_text("utf-8"))
        assert config["dataset"]["target_normalization"] == {"mode": "integer_dtype_max"}
        validate_experiment_config(config)
        if name != "train":
            assert config["eval"]["precision"] == "fp32"
            assert config["eval"]["tf32"] is False
    train = json.loads((repository / "configs/train.json").read_text("utf-8"))
    assert train["train"]["rehash_data"] is True
