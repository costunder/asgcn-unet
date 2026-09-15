"""Offline display of recorded raw-window coordinates and audited query edges.

Standard library only: no dataset, model, torch, network or CUDA access. This
module does not rebuild a graph or treat stored oracle claims as revalidation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from .diagnostic_resources import preflight

COLUMNS = ["node_index", "raw_row_id", "x", "y", "timestamp_seconds", "polarity",
           "position_x", "position_y", "position_t"]
ASSETS = Path(__file__).with_name("viewer_assets")


def _integer(value, label, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError(f"{label} must be a safely representable integer >= {minimum}")
    return value


def _number(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def display_payload(report):
    """Validate identities/coverage without inventing absent nodes or edges."""
    if not isinstance(report, dict) or report.get("schema") != "asgcn_raw_graph_audit_v1":
        raise ValueError("Expected an asgcn_raw_graph_audit_v1 report")
    window, geometry = report["window"], report["geometry"]
    if not isinstance(window, dict) or not isinstance(geometry, dict):
        raise TypeError("Window and geometry must be objects")
    count = _integer(window["nodes"], "window.nodes")
    height, width = window["sensor_size"]
    _integer(height, "sensor height", 2)
    _integer(width, "sensor width", 2)
    cutoff = _number(window["cutoff_seconds"], "cutoff_seconds")
    readout = _number(window["readout_seconds"], "readout_seconds")
    duration = _number(window["window_seconds"], "window_seconds", True)
    if cutoff > readout or not math.isclose(readout - duration, cutoff, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Window clocks are inconsistent")
    scale = _number(geometry["time_scale_seconds"], "time_scale_seconds", True)
    origin = _number(geometry["sequence_origin_seconds"], "sequence_origin_seconds")
    _number(geometry["radius"], "radius", True)
    if geometry.get("position_dims") != 3 or geometry.get("polarity_in_topology") is not False:
        raise ValueError("Only the recorded 3D spatial/physical-time graph is supported")
    cloud = report.get("point_cloud")
    if cloud is not None:
        if not isinstance(cloud, dict):
            raise TypeError("Point cloud must be an object")
        _integer(cloud.get("nodes"), "point cloud nodes")
        if (cloud.get("schema") != "asgcn_graph_point_cloud_v1" or cloud.get("columns") != COLUMNS
                or cloud.get("coverage") != "all_window_nodes" or cloud.get("nodes") != count):
            raise ValueError("Point cloud schema or coverage is inconsistent")
        nodes = cloud["rows"]
        coverage = "all_window_nodes"
        if len(nodes) != count:
            raise ValueError("Point cloud must contain every window node; no subset is substituted")
    else:
        coverage = "query_neighborhoods_only"
        nodes = []
        if not isinstance(report["node_details"], dict):
            raise TypeError("node_details must be an object")
        for key, detail in report["node_details"].items():
            if str(detail["node_index"]) != key:
                raise ValueError("node_details key differs from its node identity")
            raw, prep = detail["original_source"], detail["preprocessed"]
            polarity = int(raw["polarity"]) if isinstance(raw["polarity"], bool) else _number(
                raw["polarity"], "source polarity")
            if polarity not in {-1, 0, 1}:
                raise ValueError("Unsupported raw polarity")
            nodes.append([detail["node_index"], detail["raw_row_id"], prep["x"], prep["y"],
                          prep["timestamp_seconds"], 1 if polarity > 0 else -1,
                          *detail["normalized_topology_coordinates"]])
    if not isinstance(nodes, list):
        raise TypeError("Node rows must be a list")
    by_id, source_ids = {}, set()
    for row in nodes:
        if not isinstance(row, list) or len(row) != len(COLUMNS):
            raise ValueError("Each node must contain exactly the declared nine columns")
        node = _integer(row[0], "node index")
        raw_id = _integer(row[1], "raw row id")
        if node >= count or node in by_id or raw_id in source_ids:
            raise ValueError("Node/source identity is duplicate or outside the window")
        for value in row[2:]:
            _number(value, "node coordinate")
        if (not 0 <= row[2] <= width - 1 or not 0 <= row[3] <= height - 1
                or not cutoff <= row[4] <= readout or row[4] < origin or row[5] not in {-1, 1}):
            raise ValueError("Node is outside the declared sensor/window or has invalid polarity")
        expected = [row[2] / (width - 1), row[3] / (height - 1), (row[4] - origin) / scale]
        if any(not math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-14)
               for a, b in zip(row[6:], expected, strict=True)):
            raise ValueError("Recorded physical and topology coordinates disagree")
        by_id[node] = row
        source_ids.add(raw_id)
    first_id, last_id = window.get("first_raw_row_id"), window.get("last_raw_row_id")
    delivered_end = report["source"].get("delivery_end_idx_exclusive")
    if first_id is not None or last_id is not None:
        _integer(first_id, "first raw row")
        _integer(last_id, "last raw row")
        if last_id - first_id + 1 != count or any(row[1] != first_id + row[0] for row in nodes):
            raise ValueError("Node/source mapping disagrees with the retained delivered suffix")
    if delivered_end is not None:
        _integer(delivered_end, "delivered end")
        if any(row[1] >= delivered_end for row in nodes):
            raise ValueError("Raw source identity lies outside the delivered prefix")
    if cloud is not None:
        for key, detail in report.get("node_details", {}).items():
            node = detail["node_index"]
            if (str(node) != key or node not in by_id or by_id[node][1] != detail["raw_row_id"]
                    or by_id[node][2:5] != [detail["preprocessed"][name]
                                           for name in ("x", "y", "timestamp_seconds")]
                    or by_id[node][6:] != detail["normalized_topology_coordinates"]):
                raise ValueError("Point cloud disagrees with saved node_details")
    queries, seen_queries = [], set()
    for query in report["queries"]:
        target = _integer(query["node_index"], "query node")
        _integer(query["raw_row_id"], "query raw row")
        _integer(query.get("all_window_sources_checked"), "checked source count")
        if (target not in by_id or target in seen_queries
                or query["raw_row_id"] != by_id[target][1]
                or query.get("oracle_match") is not True
                or query.get("all_window_sources_checked") != count):
            raise ValueError("Query identity or stored all-source oracle claim is inconsistent")
        seen_queries.add(target)
        neighbors = []
        seen_neighbors = set()
        for edge in query["neighbors"]:
            source = _integer(edge["node_index"], "neighbor node")
            _integer(edge["raw_row_id"], "neighbor raw row")
            if (source not in by_id or source == target or source in seen_neighbors
                    or edge["raw_row_id"] != by_id[source][1]):
                raise ValueError("Neighbor is missing, duplicated, self-linked or misidentified")
            distance = _number(edge["distance_over_radius"], "distance_over_radius")
            if not 0 <= distance < 1:
                raise ValueError("Recorded neighbor lies outside the strict radius")
            absolute = _number(edge["distance"], "distance")
            if not math.isclose(absolute, distance * geometry["radius"], rel_tol=1e-14, abs_tol=1e-14):
                raise ValueError("Saved edge distances disagree with recorded radius")
            coordinate_distance = math.hypot(*[(a - b) / geometry["radius"]
                                                for a, b in zip(by_id[source][6:], by_id[target][6:],
                                                                strict=True)])
            if not math.isclose(coordinate_distance, distance, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("Saved edge distance disagrees with endpoint coordinates")
            neighbors.append(source)
            seen_neighbors.add(source)
        if _integer(query["in_degree"], "in_degree") != len(neighbors):
            raise ValueError("All recorded query neighbors must be present")
        queries.append({"node_index": target, "raw_row_id": by_id[target][1],
                        "in_degree": len(neighbors), "oracle_match": True, "neighbors": neighbors})
    total = report.get("total_directed_edges")
    if total is not None:
        _integer(total, "total_directed_edges")
        full = report.get("full_count")
        if (not isinstance(full, dict) or full.get("all_nodes_counted") is not True
                or full.get("directed_edges") != total or full.get("nodes") != count
                or total % 2 or total > count * max(0, count - 1)
                or total < sum(q["in_degree"] for q in queries)):
            raise ValueError("Total edge count has no consistent complete counting record")
    source = report["source"]
    if not isinstance(source["file_name"], str):
        raise TypeError("source.file_name must be text")
    _integer(source["frame_index"], "frame_index")
    return {"schema": "asgcn_graph_inspection_view_v1", "nodes": nodes, "queries": queries,
            "window": window, "geometry": geometry,
            "source": {key: source[key] for key in ("file_name", "frame_index", "identity")},
            "coverage": coverage, "total_directed_edges": total,
            "synthetic_fixture": report.get("synthetic_fixture") is True,
            "saved_report_sha256": report.get("saved_report_sha256"),
            "provenance_note": "Stored diagnostic only; original data and oracle not revalidated. "
                               "Selected incoming edges only, not the full edge set. Not paper-exact."}


def _working_estimate(report):
    cloud = report.get("point_cloud")
    nodes = cloud.get("rows", []) if isinstance(cloud, dict) else report.get("node_details", {})
    queries = report.get("queries", [])
    # Conservative allowance for Python objects, indexing, JSON, escaped Unicode,
    # template copies and UTF-8 encoding. A budget failure never samples the graph.
    return 64 * 1024**2 + len(nodes) * 8192 + sum(len(q["neighbors"]) for q in queries) * 2048


def save_html(report, output, *, workspace, memory_budget_bytes, reserve_memory_bytes, cpu_threads):
    """Write one self-contained new HTML; no overwrite, network or full E build."""
    root = Path(workspace).resolve(strict=True)
    output = Path(output)
    output = (output if output.is_absolute() else root / output).resolve(strict=False)
    if (not output.is_relative_to(root) or output.suffix.lower() != ".html"
            or not output.parent.is_dir()):
        raise ValueError("Output must be a new HTML in an existing workspace directory")
    if output.exists():
        raise FileExistsError(f"Offline graph output already exists: {output}")
    preflight(budget_bytes=memory_budget_bytes, reserve_bytes=reserve_memory_bytes,
              cpu_threads=cpu_threads)
    if _working_estimate(report) > memory_budget_bytes:
        raise MemoryError("Offline graph payload exceeds the explicit RAM planning budget; no nodes omitted")
    payload = display_payload(report)
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = (ASSETS / "inspection.html").read_text(encoding="utf-8")
    replacements = {"__INSPECTION_CSS__": (ASSETS / "inspection.css").read_text(encoding="utf-8"),
                    "__INSPECTION_JS__": (ASSETS / "inspection.js").read_text(encoding="utf-8"),
                    "__INSPECTION_DATA__": encoded}
    # Split before insertion: source strings resembling a placeholder are data,
    # and are never recursively interpreted as a template directive.
    html = re.sub("|".join(replacements), lambda match: replacements[match[0]], template)
    preflight(budget_bytes=memory_budget_bytes, reserve_bytes=reserve_memory_bytes,
              cpu_threads=cpu_threads)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=output.parent, prefix=".graph-view-", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(html)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication without replacing an existing experiment/artifact.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink()
    return output


def export_saved_report(source, output, *, workspace, memory_budget_bytes,
                        reserve_memory_bytes, cpu_threads):
    """Read a bounded saved report, then export its actual available coordinates."""
    preflight(budget_bytes=memory_budget_bytes, reserve_bytes=reserve_memory_bytes,
              cpu_threads=cpu_threads)
    source = Path(source).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise ValueError("Input must be an existing diagnostic JSON")
    before = source.stat()
    read_budget = max(0, (memory_budget_bytes - 64 * 1024**2) // 32)
    if before.st_size > read_budget:
        raise MemoryError("Saved report exceeds the explicit JSON decoding memory budget")
    with source.open("rb") as handle:
        raw = handle.read(read_budget + 1)
    after = source.stat()
    if (len(raw) > read_budget or len(raw) != before.st_size
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise RuntimeError("Saved graph report changed while reading")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    report = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    if not isinstance(report, dict):
        raise TypeError("Graph report must be an object")
    report["saved_report_sha256"] = hashlib.sha256(raw).hexdigest()
    del raw
    return save_html(report, output, workspace=workspace, memory_budget_bytes=memory_budget_bytes,
                     reserve_memory_bytes=reserve_memory_bytes, cpu_threads=cpu_threads)
