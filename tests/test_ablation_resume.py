"""Synthetic protocol checks for explicitly graph-free comparison artifacts."""

import json

import pytest

from asgcn_unet.engine import _canonical_sha256, _hashed_contract
from tests import test_eval_resume as original


@pytest.mark.parametrize("encoder", ["identity", "pointwise"])
def test_completed_no_graph_protocol_is_verified_not_overwritten(tmp_path, monkeypatch, encoder):
    config = original.load_json(original.CONFIG_PATH)
    config["model"].update(encoder_kind=encoder, decoder_kind="unet", spline_backend="torch")
    config_path = tmp_path / "aid-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(original, "CONFIG_PATH", config_path)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"mock sealed checkpoint: protocol-only diagnostic")
    output = tmp_path / "aid"
    directory = original._write_complete_mode(output, checkpoint)
    original._stub_request_validation(monkeypatch, checkpoint)
    for name, key in [("metrics.json", "evaluation_protocol"), ("benchmark.json", "benchmark_protocol")]:
        path = directory / name
        payload = json.loads(path.read_text())
        protocol = payload[key]
        contract = protocol["execution"]["contract"]
        contract["graph_edge_guard"].update(edge_guard_applicable=False, topology_kind="no_graph")
        protocol["execution"] = _hashed_contract(contract)
        protocol.pop("protocol_sha256")
        protocol["protocol_sha256"] = _canonical_sha256(protocol)
        path.write_text(json.dumps(payload), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert original.resume.inspect_mode(original._args(output, checkpoint)) == (1, 1)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
