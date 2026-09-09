"""Read-only, loopback-only viewer for completed evaluation artifacts.

Predictions are never regenerated. Graphs are reconstructed explicitly on the
CPU from the current dataset, and only their display edges may be subsampled.
No evaluation, checkpoint loading, data hash cache writes, or CUDA selection
occurs in this module.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

_MODE = re.compile(r"ann|snn_(?:literal_eq15|standard_if)_T[1-9][0-9]*")
_STEM = re.compile(r"(?P<index>[0-9]{8,})_.+_[0-9a-f]{12}_pred\.png")
_LABELS = {"aid": "EventAid-R", "hdr": "EventHDR"}
_TYPES = {"aid": "eventaid_r_zip", "hdr": "eventhdr"}


class ViewerError(ValueError):
    """An explicit artifact, identity, or viewer request error."""


def _object(path: Path) -> dict[str, Any]:
    def invalid_constant(value: str):
        raise ViewerError(f"Non-finite JSON value {value} in {path.name}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ViewerError(f"Expected a JSON object: {path}")
    return value


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ViewerError("Artifact symlink/path escapes the selected evaluation root")
    if not resolved.is_file():
        raise ViewerError(f"Expected a regular file: {path.name}")
    return resolved


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ViewerError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ViewerError("Boolean value in a numeric metric")
    result = float(value)
    if not math.isfinite(result):
        raise ViewerError("Non-finite value in a saved metric")
    return result


def _url(endpoint: str, **values: Any) -> str:
    return endpoint + "?" + urlencode(values)


@dataclass
class SavedFrame:
    index: int
    sample_id: str
    row: dict[str, str]
    prediction: Path
    target: Path


@dataclass
class SavedRun:
    mode: str
    metrics_path: Path
    total_frames: int
    eligible: bool | None
    frames: dict[int, SavedFrame] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    signatures: dict[Path, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class DatasetView:
    key: str
    runs: dict[str, SavedRun]
    config_path: Path | None
    dataset: Any = None
    config: dict[str, Any] | None = None


def _read_run(directory: Path, root: Path, dataset_key: str) -> SavedRun:
    from .engine import _prediction_artifact_stem
    from .viewer_protocol import prepare_viewer_protocol

    metrics_path = _inside(directory / "metrics.json", root)
    report = _object(metrics_path)
    if report.get("dataset") != _TYPES[dataset_key]:
        raise ViewerError(f"Dataset mismatch in {dataset_key}/{directory.name}")
    quality = report.get("quality")
    if not isinstance(quality, dict):
        raise ViewerError("Saved quality must be a JSON object")
    total = _integer(quality.get("frames"), "quality.frames", 1)
    mode = directory.name
    expected_mode = "ann" if mode == "ann" else "snn"
    if report.get("inference_mode") != expected_mode:
        raise ViewerError(f"Inference mode mismatch in {dataset_key}/{mode}")
    if expected_mode == "snn" and (
        mode != f"snn_{report.get('snn_dynamics')}_T{report.get('simulation_steps')}"
    ):
        raise ViewerError(f"SNN identity mismatch in {dataset_key}/{mode}")
    run = SavedRun(mode, metrics_path, total, report.get("report_eligible"))
    candidates: dict[int, Path] = {}
    predictions = directory / "predictions"
    if predictions.exists() and not predictions.resolve().is_relative_to(root):
        raise ViewerError("Predictions directory escapes the selected evaluation root")
    for path in sorted(predictions.glob("*_pred.png")):
        match = _STEM.fullmatch(path.name)
        if match is None:
            raise ViewerError(f"Unrecognized prediction filename: {path.name}")
        index = int(match.group("index"))
        if index in candidates or index >= total:
            raise ViewerError(f"Duplicate/out-of-range saved dataset index: {index}")
        candidates[index] = _inside(path, root)
    csv_path = _inside(directory / "frames.csv", root)
    count = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"sample_id", "scene"}.issubset(reader.fieldnames):
            raise ViewerError(f"Missing frame identity columns: {csv_path}")
        for index, row in enumerate(reader):
            count += 1
            if index not in candidates:
                continue
            sample_id = row.get("sample_id")
            if not sample_id or None in row:
                raise ViewerError(f"Invalid saved frame row {index}")
            stem = _prediction_artifact_stem(sample_id, index)
            if candidates[index].name != stem + "_pred.png":
                raise ViewerError(f"Prediction/CSV identity mismatch at dataset index {index}")
            target = _inside(predictions / (stem + "_gt.png"), root)
            run.frames[index] = SavedFrame(index, sample_id, row, candidates[index], target)
    if count != total:
        raise ViewerError(f"Incomplete frames.csv: {count} rows, metrics declare {total}")
    if len(run.frames) != len(candidates):
        raise ViewerError("Some saved predictions have no matching frame row")
    # Validate the large report once, retaining only the few saved-frame identities.
    # Never hold the full selected-sample list for every mode in viewer RAM.
    run.protocol = prepare_viewer_protocol(report, dataset_indices=sorted(run.frames))
    run.signatures = {path: _signature(path) for path in (metrics_path, csv_path)}
    return run


def _signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _png_array(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        if image.format != "PNG" or image.mode not in {"L", "RGB"}:
            raise ViewerError(f"Expected an evaluation L/RGB PNG: {path.name}")
        image.load()
        return np.array(image)


def _target_array(tensor):
    import numpy as np

    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim != 3 or array.shape[0] not in {1, 3}:
        raise ViewerError("Expected a CHW target with 1 or 3 channels")
    array = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
    return (array * 255.0 + 0.5).astype(np.uint8)


def _png_bytes(array) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class ResultViewer:
    def __init__(
        self,
        eval_root: str | Path,
        *,
        configs: dict[str, str | Path] | None = None,
        display_edges: int = 5000,
    ) -> None:
        self.root = Path(eval_root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ViewerError("--eval-root must be an evaluation directory")
        self.display_edges = _integer(display_edges, "display_edges")
        self.datasets: dict[str, DatasetView] = {}
        self.warnings: list[str] = []
        self._lock = threading.RLock()
        # One current graph across both datasets: no unbounded frame/edge cache.
        self._graph_key: tuple[str, int] | None = None
        self._graph: Any = None
        self._raw_target: bytes | None = None
        for key in _LABELS:
            base = self.root / key
            if not base.is_dir():
                continue
            if not base.resolve().is_relative_to(self.root):
                raise ViewerError("Dataset directory escapes the selected evaluation root")
            runs: dict[str, SavedRun] = {}
            for directory in sorted(base.iterdir()):
                if not directory.is_dir() or not _MODE.fullmatch(directory.name):
                    continue
                try:
                    run = _read_run(directory, self.root, key)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    self.warnings.append(f"{key}/{directory.name}: {error}")
                    continue
                runs[run.mode] = run
                if not run.frames:
                    self.warnings.append(f"{key}/{run.mode}: no saved comparison PNGs")
            if not runs:
                continue
            config_value = (configs or {}).get(key)
            path = Path(config_value).expanduser().resolve() if config_value else None
            if path is not None and not path.is_file():
                self.warnings.append(f"{key}: graph config not found; image viewing still works")
                path = None
            self.datasets[key] = DatasetView(key, runs, path)
        if not self.datasets:
            detail = "; ".join(self.warnings)
            raise ViewerError("No completed aid/hdr evaluation artifacts found. " + detail)

    def _view(self, dataset: str) -> DatasetView:
        if dataset not in self.datasets:
            raise ViewerError("Unknown dataset")
        return self.datasets[dataset]

    def _frames(self, dataset: str, index: int) -> list[tuple[SavedRun, SavedFrame]]:
        _integer(index, "dataset index")
        view = self._view(dataset)
        result = [(run, run.frames[index]) for run in view.runs.values() if index in run.frames]
        if not result:
            raise ViewerError("No saved PNG for this frame; the viewer never invents predictions")
        if len({frame.sample_id for _, frame in result}) != 1:
            raise ViewerError("Sample identities differ between saved modes")
        for run, _ in result:
            if any(_signature(_inside(path, self.root)) != signature
                   for path, signature in run.signatures.items()):
                raise ViewerError("Evaluation metadata changed after startup; restart the viewer")
        return result

    def catalog(self) -> dict[str, Any]:
        datasets = []
        for key, view in self.datasets.items():
            indices = sorted({index for run in view.runs.values() for index in run.frames})
            frame_list = []
            for index in indices:
                entries = self._frames(key, index)
                frame = entries[0][1]
                frame_list.append({
                    "index": index, "sample_id": frame.sample_id,
                    "group": frame.row["scene"], "modes": [run.mode for run, _ in entries],
                })
            totals = {run.total_frames for run in view.runs.values()}
            if len(totals) != 1:
                raise ViewerError(f"{key}: modes declare different evaluation frame counts")
            datasets.append({
                "id": key, "label": _LABELS[key], "total_frames": totals.pop(),
                "modes": [{"id": run.mode, "label": run.mode, "saved_frames": len(run.frames)}
                          for run in view.runs.values()],
                "frames": frame_list,
            })
        return {"datasets": datasets, "readonly": True, "warnings": self.warnings}

    def frame(self, dataset: str, index: int) -> dict[str, Any]:
        import numpy as np

        entries = self._frames(dataset, index)
        reference = _png_array(_inside(entries[0][1].target, self.root))
        images = []
        for run, frame in entries:
            if not np.array_equal(reference, _png_array(_inside(frame.target, self.root))):
                raise ViewerError("Saved ground truths differ between modes; comparison refused")
            if _png_array(_inside(frame.prediction, self.root)).shape != reference.shape:
                raise ViewerError("Prediction and ground-truth image shapes differ")
            params = {"dataset": dataset, "index": index, "mode": run.mode}
            images.append({
                "mode": run.mode,
                "url": _url("api/image", **params, kind="prediction"),
                "gt_url": _url("api/image", **params, kind="target"),
                "metrics": {key: _number(frame.row.get(key))
                            for key in ("psnr", "ssim", "rmse", "temporal_l1")},
                "report_eligible": run.eligible,
            })
        return {
            "dataset": dataset, "index": index, "sample_id": entries[0][1].sample_id,
            "images": images, "graph_available": self._view(dataset).config_path is not None,
            "evaluation_domain": "Saved evaluation GT: normalized / configured channels / "
                                 "configured tone mapping; 8-bit PNG, not raw HDR radiance.",
            "note": "Predictions retain the original full-evaluation recurrent context. "
                    "Metrics are the saved float CSV values, not recomputed from PNGs.",
        }

    def image(self, dataset: str, index: int, mode: str, kind: str) -> bytes:
        entries = self._frames(dataset, index)
        for run, frame in entries:
            if run.mode == mode:
                if kind not in {"prediction", "target"}:
                    raise ViewerError("Unknown image kind")
                path = frame.prediction if kind == "prediction" else frame.target
                return _inside(path, self.root).read_bytes()
        raise ViewerError("Unknown mode or missing saved image")

    def _graph_inputs(self, dataset: str, index: int):
        from .data import build_dataset
        from .engine import _dataset_sample_identity
        from .utils import load_json, resolve_experiment_paths
        from .viewer_protocol import validate_viewer_protocol

        view = self._view(dataset)
        if view.config_path is None:
            raise ViewerError("A matching evaluation config is required for graph/source viewing")
        entries = self._frames(dataset, index)
        if view.config is None:
            view.config = resolve_experiment_paths(load_json(view.config_path), view.config_path)
        config = view.config
        from .stream_input import reject_streaming_frame_diagnostic

        reject_streaming_frame_diagnostic(config["model"], config["dataset"])
        # All modes must describe the same graph/data transform before showing a
        # common graph. Only small contracts survive these report reads.
        guard = None
        for run, _ in entries:
            binding = validate_viewer_protocol(run.protocol, config=config, dataset_index=index)
            current = binding["effective_max_graph_edges"]
            if guard is not None and current != guard:
                raise ViewerError("Saved modes have different graph guards")
            guard = current
        if view.dataset is None:
            view.dataset = build_dataset(config["dataset"], split="eval")
        if index >= len(view.dataset):
            raise ViewerError("Saved index is outside the current source dataset")
        current_identity = _dataset_sample_identity(view.dataset, index)
        for run, _ in entries:
            validate_viewer_protocol(run.protocol, config=config, dataset_index=index,
                                     current_identity=current_identity)
        sample = view.dataset[index]
        if sample["sample_id"] != entries[0][1].sample_id:
            raise ViewerError("Current source sample_id does not match saved prediction")
        import numpy as np

        expected_target = _target_array(sample["target"])
        for _, frame in entries:
            if not np.array_equal(expected_target, _png_array(_inside(frame.target, self.root))):
                raise ViewerError("Current source target does not match saved evaluation GT")
        return view, sample, entries, guard

    def graph(self, dataset: str, index: int) -> dict[str, Any]:
        from .graph_preview import build_graph_preview

        with self._lock:
            key = (dataset, index)
            self._frames(dataset, index)
            if self._graph_key != key:
                # Release the old dense edge tensor before building another.
                self._graph_key, self._graph, self._raw_target = None, None, None
                view, sample, entries, guard = self._graph_inputs(dataset, index)
                preview = build_graph_preview(
                    sample, view.config["model"], max_graph_edges=guard,
                    display_edges=self.display_edges,
                )
                stats = preview.payload["statistics"]
                for _, frame in entries:
                    for column, field_name in (("nodes", "nodes"),
                                               ("edges", "actual_directed_edges")):
                        raw = frame.row.get(column)
                        if raw is None or not raw.isdecimal() or int(raw) != stats[field_name]:
                            raise ViewerError(f"Reconstructed {column} differs from saved CSV")
                # Decode the same indexed source with tone mapping disabled.
                # This changes only a private reader attribute under the lock;
                # config files, evaluation targets, and source files are untouched.
                original_tone = view.dataset.tone_map
                try:
                    view.dataset.tone_map = "none"
                    linear_sample = view.dataset[index]
                finally:
                    view.dataset.tone_map = original_tone
                if linear_sample["sample_id"] != sample["sample_id"]:
                    raise ViewerError("Source preview identity mismatch")
                raw_target = _png_bytes(_target_array(linear_sample["target"]))
                preview.payload["provenance_note"] = (
                    "CPU reconstruction from current source files, not a stored graph or new "
                    "inference. Config/manifest, sample ID, saved GT pixels and node/edge counts "
                    "were checked. Full source-file SHA-256 was NOT rehashed; this is a "
                    "visual diagnostic, not proof of historical tensor identity. Display-edge "
                    "sampling never changes model topology or predictions."
                )
                preview.payload["raw_target_url"] = _url(
                    "api/raw-target", dataset=dataset, index=index,
                )
                self._graph_key, self._graph, self._raw_target = key, preview, raw_target
            return self._graph.payload

    def neighbors(self, dataset: str, index: int, node: int) -> dict[str, Any]:
        with self._lock:
            self.graph(dataset, index)
            return self._graph.neighbors(node)

    def raw_target(self, dataset: str, index: int) -> bytes:
        with self._lock:
            self.graph(dataset, index)
            return self._raw_target

    def close(self) -> None:
        with self._lock:
            self._graph_key, self._graph, self._raw_target = None, None, None
            for view in self.datasets.values():
                if view.dataset is not None:
                    view.dataset.close()
                    view.dataset = None


class ViewerHTTPServer(ThreadingHTTPServer):
    """No directory listing, arbitrary file endpoints, CORS, or public bind."""

    daemon_threads = True

    def __init__(self, viewer: ResultViewer, port: int = 8765):
        self.viewer = viewer
        self.token = secrets.token_urlsafe(24)
        super().__init__(("127.0.0.1", port), ViewerHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/{self.token}/"


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        # The access token must not be copied into per-request HTTP logs.
        return

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; "
                         "style-src 'self'; img-src 'self' data:; connect-src 'self'; "
                         "base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._reply(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                    "application/json; charset=utf-8")

    def do_GET(self) -> None:
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin is not None and origin != f"http://{host}"):
            self._json(403, {"error": "Loopback same-origin requests only"})
            return
        try:
            split = urlsplit(self.path)
        except ValueError as error:
            self._json(400, {"error": f"Invalid request URL: {error}"})
            return
        prefix = f"/{self.server.token}/"
        if not split.path.startswith(prefix):
            self._json(403, {"error": "Open the complete private URL printed by the viewer"})
            return
        endpoint = split.path[len(prefix):]
        try:
            params = parse_qs(split.query, strict_parsing=True, max_num_fields=8)

            def one(name: str) -> str:
                value = params.get(name)
                if not value or len(value) != 1:
                    raise ViewerError(f"Exactly one {name} is required")
                return value[0]

            def number(name: str) -> int:
                value = one(name)
                if not value.isascii() or not value.isdecimal() or len(value) > 12:
                    raise ViewerError(f"Invalid {name}")
                return int(value)

            app = self.server.viewer
            if endpoint in {"", "index.html", "viewer.css", "viewer.js"}:
                filename = endpoint or "index.html"
                mime = {"index.html": "text/html", "viewer.css": "text/css",
                        "viewer.js": "text/javascript"}[filename]
                body = files("asgcn_unet").joinpath("viewer_assets", filename).read_bytes()
                self._reply(200, body, mime + "; charset=utf-8")
            elif endpoint == "api/catalog":
                self._json(200, app.catalog())
            elif endpoint == "api/frame":
                self._json(200, app.frame(one("dataset"), number("index")))
            elif endpoint == "api/image":
                body = app.image(one("dataset"), number("index"), one("mode"), one("kind"))
                self._reply(200, body, "image/png")
            elif endpoint == "api/graph":
                self._json(200, app.graph(one("dataset"), number("index")))
            elif endpoint == "api/neighbors":
                self._json(200, app.neighbors(one("dataset"), number("index"), number("node")))
            elif endpoint == "api/raw-target":
                self._reply(200, app.raw_target(one("dataset"), number("index")), "image/png")
            else:
                self._json(404, {"error": "Unknown viewer endpoint"})
        except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self._json(400, {"error": f"{type(error).__name__}: {error}"})
