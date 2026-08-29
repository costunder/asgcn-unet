from __future__ import annotations

import torch

from asgcn_recon.data import EventAidRZipDataset, EventHDRDataset
from asgcn_recon.engine import _model_state_sha256, benchmark, evaluate
from asgcn_recon.losses import ReconstructionLoss
from asgcn_recon.model import ASGCNReconstructor
from asgcn_recon.utils import atomic_torch_save
from tests.fixtures import make_eventaid, make_eventhdr


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
    model = ASGCNReconstructor(**model_config)
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
        },
    }
    result = evaluate(config, checkpoint)
    timing = benchmark(config, checkpoint, warmup=1, steps=2)

    assert result["quality"]["frames"] == 2
    assert timing["frames"] == 2
    assert timing["mean_raw_events"] == 80
    assert timing["mean_retained_events"] == 27
    assert timing["retention_ratio"] == 27 / 80
    assert timing["raw_events_per_second"] > timing["retained_events_per_second"]
    assert timing["graph_nodes_per_second"] == timing["retained_events_per_second"]
    assert timing["events_per_second"] == timing["retained_events_per_second"]
    assert timing["recurrent_context_frames"] == 1
    assert timing["state_resets"] == 0
    assert timing["state_reset_ratio"] == 0.0
    ann_output = output_dir / "ann"
    assert (ann_output / "metrics.json").is_file()
    assert (ann_output / "frames.csv").is_file()
    assert (ann_output / "benchmark.json").is_file()
    assert len(list((ann_output / "predictions").glob("*_pred.png"))) == 1

    hdr.close()
    aid.close()
