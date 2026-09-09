"""Legacy diagnostic boundaries, without external data or device initialization."""

import pytest

from asgcn_unet.diagnostic_graph import build_diagnostic_graph
from asgcn_unet.graph_preview import _encoder_topology_kind, build_graph_preview
from asgcn_unet.preflight import _sample_topology
from asgcn_unet.sample_probe import probe_evaluation_sample
from asgcn_unet.stream_input import reject_streaming_frame_diagnostic
from asgcn_unet.topology_scan import _sample_record, scan_evaluation_topology
from tests.test_stream_preflight import _config


@pytest.mark.parametrize("function", [
    lambda sample, model: build_diagnostic_graph(sample, model, memory_budget_bytes=1024),
    lambda sample, model: build_graph_preview(sample, model, max_graph_edges=100),
    lambda sample, model: _sample_topology(sample, model, 0),
    lambda sample, model: _sample_record(sample, model, 0),
])
@pytest.mark.parametrize("model_contract", [True, False])
def test_static_graph_tools_reject_model_or_sample_contract_before_graph_work(function, model_contract):
    model = _config()["model"] if model_contract else {}
    sample = {} if model_contract else {"metadata": {"stream_time": {"schema": "physical_seconds_v1"}}}
    with pytest.raises(ValueError, match="causal predecessor stream state"):
        function(sample, model)


def test_legacy_topology_kind_unchanged_and_stream_kind_explicitly_refused():
    for encoder in ("graph", "pointwise", "identity"):
        assert _encoder_topology_kind({"encoder_kind": encoder}) == (
            "radius_graph" if encoder == "graph" else "no_graph")
    with pytest.raises(ValueError, match="v2 static frame"):
        _encoder_topology_kind(_config()["model"])
    with pytest.raises(ValueError, match="v2 static frame"):
        reject_streaming_frame_diagnostic({}, {"event_time_contract": "physical_seconds_v1"})


@pytest.mark.parametrize("operation", ["scan", "probe"])
def test_single_frame_cli_implementation_refuses_before_data_device_or_output(tmp_path, monkeypatch, operation):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported stream diagnostic must not allocate a device or read data")
    for module in ("sample_probe", "topology_scan"):
        monkeypatch.setattr(f"asgcn_unet.{module}.build_dataset", forbidden)
        monkeypatch.setattr(f"asgcn_unet.{module}.resolve_device", forbidden)
    with pytest.raises(ValueError, match="causal predecessor stream state"):
        if operation == "scan":
            scan_evaluation_topology(_config(), tmp_path / "no-output.json")
        else:
            probe_evaluation_sample(_config(), tmp_path / "missing.pt", sample_index=0, max_graph_edges=100)
    assert list(tmp_path.iterdir()) == []
