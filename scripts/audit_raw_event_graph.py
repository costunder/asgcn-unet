"""Selected-query audit of one real uncapped EventHDR window; no training/GPU.

This script is intentionally outside the production package/source commitment.
Heavy imports occur only after the existing fail-closed resource preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from asgcn_unet.diagnostic_resources import preflight


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _code_identity():
    """Bounded hashes of explicitly named diagnostic/geometry files, not a full tree."""
    names = ["scripts/audit_raw_event_graph.py", "src/asgcn_unet/implicit_radius.py",
             "src/asgcn_unet/radius_candidates.py", "src/asgcn_unet/stream_graph.py",
             "src/asgcn_unet/data/eventhdr.py", "src/asgcn_unet/data/common.py",
             "src/asgcn_unet/stream_input.py", "src/asgcn_unet/stream_geometry.py",
             "src/asgcn_unet/diagnostic_resources.py",
             "src/asgcn_unet/resources.py"]
    hashes = {}
    for name in names:
        path = PROJECT / name
        if path.stat().st_size > 1024**2:
            raise MemoryError("Diagnostic code identity exceeds its 1-MiB per-file read budget")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"kind": "partial_code_identity", "complete_source_tree": False,
            "script_sha256": hashes[names[0]], "file_sha256": hashes}


def select_window_rows(raw_timestamps, *, timestamp_scale_to_seconds,
                       readout_seconds, window_seconds):
    """Filter only the DELIVERED prefix; retain cutoff equality and duplicate rows."""
    import numpy as np

    from asgcn_unet.stream_input import to_physical_seconds

    _positive(timestamp_scale_to_seconds, "timestamp_scale_to_seconds")
    _positive(window_seconds, "window_seconds")
    if not math.isfinite(readout_seconds):
        raise ValueError("readout_seconds must be finite")
    original = np.asarray(raw_timestamps)
    if (original.ndim != 1 or original.dtype.kind not in "iuf"
            or not np.all(np.isfinite(original)) or np.any(original[1:] < original[:-1])):
        raise ValueError("Delivered prefix timestamps must be finite and chronological")
    raw = np.asarray(original, dtype=np.float64)
    if np.any((original[1:] != original[:-1]) & (raw[1:] == raw[:-1])):
        raise ValueError("Float64 timestamp conversion loses distinct source timestamp resolution")
    seconds = to_physical_seconds(raw, timestamp_scale_to_seconds, source="selected prefix")
    if np.any(seconds > readout_seconds):
        raise ValueError("Delivered prefix includes events after the declared readout")
    rows = np.flatnonzero(seconds >= readout_seconds - window_seconds)
    return rows.astype(np.int64, copy=False), seconds[rows]


def normalized_positions(events, sensor_size, *, origin_seconds, time_scale_seconds):
    """Use the model's production geometry; do not reimplement its transform."""
    import torch

    from asgcn_unet.stream_geometry import physical_node_positions

    return physical_node_positions(
        torch.as_tensor(events, dtype=torch.float64, device="cpu"), sensor_size,
        origin_seconds=origin_seconds, time_scale_seconds=time_scale_seconds,
    )


def query_neighbors(positions, raw_ids, query_indices, *, radius, candidate_pair_budget, index=None):
    """Compare selected incoming queries with ALL sources using independent brute force."""
    import torch

    from asgcn_unet.implicit_radius import ImplicitRadiusIndex

    _positive(radius, "radius")
    _integer(candidate_pair_budget, "candidate_pair_budget", 1)
    if positions.device.type != "cpu" or positions.dtype != torch.float64:
        raise ValueError("Query audit requires float64 CPU positions")
    ids = torch.as_tensor(raw_ids, dtype=torch.long, device="cpu")
    if ids.shape != (len(positions),) or ids.unique().numel() != ids.numel():
        raise ValueError("raw_ids must identify every node uniquely")
    queries = [_integer(value, "query index") for value in query_indices]
    if len(set(queries)) != len(queries) or any(value >= len(positions) for value in queries):
        raise ValueError("Query indices must be unique and inside the complete window")
    if index is None:
        index = ImplicitRadiusIndex(
            positions, torch.zeros(len(positions), dtype=torch.long), batch_size=1,
            radius=radius, position_dims=3, chunk_size=1,
            candidate_pair_budget=candidate_pair_budget,
        )
    elif (not isinstance(index, ImplicitRadiusIndex) or index.positions is not positions
          or index.radius != radius or index.position_dims != 3 or index.batch_size != 1):
        raise ValueError("Reused query index does not match this complete window geometry")
    results = []
    for query in queries:
        sources, distances, stats = [], [], {}
        for source, destination, distance in index.iter_directed_neighbors(
            torch.tensor([query]), stats=stats,
        ):
            if not bool((destination == query).all()):
                raise RuntimeError("Production query returned a different destination")
            sources.append(source)
            distances.append(distance.flatten())
        actual = torch.cat(sources) if sources else torch.empty(0, dtype=torch.long)
        actual_distance = torch.cat(distances) if distances else torch.empty(0, dtype=torch.float64)
        order = actual.argsort()
        actual, actual_distance = actual[order], actual_distance[order]
        # The oracle uses no production cell index, candidate filter, or degree.
        oracle_distance = torch.linalg.vector_norm(
            (positions[:, :3] - positions[query, :3]) / radius, dim=1,
        )
        mask = oracle_distance < 1
        mask[query] = False
        expected = torch.nonzero(mask, as_tuple=True)[0]
        if (not torch.equal(actual, expected)
                or not torch.equal(actual_distance, oracle_distance[expected])):
            raise RuntimeError(f"Production query differs from brute-force oracle: {query}")
        results.append({
            "node_index": query, "raw_row_id": int(ids[query]),
            "in_degree": len(actual), "all_window_sources_checked": len(positions),
            "oracle_match": True, "production_work": stats,
            "neighbors": [
                {"node_index": int(node), "raw_row_id": int(ids[node]),
                 "distance_over_radius": distance, "distance": distance * radius}
                for node, distance in zip(actual.tolist(), actual_distance.tolist(), strict=True)
            ],
        })
    return results


def count_all_degrees(index):
    """Count all production incoming queries with O(N) degrees, never retain E edges.

    This is explicitly O(E) query work in the dense case, not the bulk total-only
    topology counter. It is not an independent full-graph oracle.
    """
    import torch

    from asgcn_unet.implicit_radius import ImplicitRadiusIndex

    if (not isinstance(index, ImplicitRadiusIndex) or index.positions.device.type != "cpu"
            or index.positions.dtype != torch.float64 or index.batch_size != 1
            or index.position_dims != 3):
        raise ValueError("Full degree audit requires one complete float64 CPU window index")
    count = len(index.positions)
    degrees = torch.zeros(count, dtype=torch.long)
    stats = {}
    started = time.perf_counter()
    for source, destination, distance in index.iter_directed_neighbors(stats=stats):
        degrees.index_add_(0, destination, torch.ones_like(destination))
        # Only the current bounded chunk is retained, never a whole E-sized list.
        del source, destination, distance
    total = int(degrees.sum())
    if (total % 2 or total > count * max(0, count - 1)
            or bool((degrees < 0).any()) or bool((degrees >= max(count, 1)).any())):
        raise RuntimeError("Full radius degree count violates no-self/symmetric graph invariants")
    return degrees, {
        "nodes": count, "directed_edges": total, "undirected_edges": total // 2,
        "degree_min": int(degrees.min()) if count else None,
        "degree_mean": total / count if count else None,
        "degree_max": int(degrees.max()) if count else None,
        "isolated_nodes": int((degrees == 0).sum()),
        "count_time_s": time.perf_counter() - started,
        "production_work": stats, "all_nodes_counted": True,
        "edge_list_materialized": False, "full_graph_oracle_verified": False,
        "method": "production index iter_directed_neighbors + incoming degree index_add_",
        "scope": "All nodes of this one selected delivered window, not the whole dataset",
        "complexity_note": "Potentially O(E) time; O(N) degree storage plus bounded candidate scratch",
        "symmetry_check": "Even directed total under the symmetric radius rule; not a reciprocity proof",
    }


def audit_raw_event_graph(*, source_file, frame_index, window_seconds, time_scale_seconds,
                          radius, timestamp_scale_to_seconds, interval_timestamp_scale_to_seconds,
                          cpu_threads, memory_budget_bytes, reserve_memory_bytes, query_indices=None,
                          count_all_nodes=False, target_options=None):
    """Preserve every R=1 event in one explicit frame's delivered physical window."""
    audit_started = time.perf_counter()
    if type(count_all_nodes) is not bool:
        raise TypeError("count_all_nodes must be an explicit boolean")
    if target_options is None:
        target_options = {}
    if (not isinstance(target_options, dict) or set(target_options) - {
            "target_channels", "target_normalization", "tone_map", "tone_map_mu"}):
        raise ValueError("Only explicit target-reader settings can be forwarded")
    for value, name in ((window_seconds, "window_seconds"), (time_scale_seconds, "time_scale_seconds"),
                        (radius, "radius"), (timestamp_scale_to_seconds, "timestamp_scale_to_seconds"),
                        (interval_timestamp_scale_to_seconds, "interval_timestamp_scale_to_seconds")):
        _positive(value, name)
    _integer(frame_index, "frame_index")
    _integer(memory_budget_bytes, "memory_budget_bytes", 1)
    _integer(reserve_memory_bytes, "reserve_memory_bytes", 1)
    _integer(cpu_threads, "cpu_threads", 1)
    resource = preflight(budget_bytes=memory_budget_bytes, reserve_bytes=reserve_memory_bytes,
                         cpu_threads=cpu_threads)
    overhead = 256 * 1024**2
    if memory_budget_bytes < overhead:
        raise MemoryError("Budget is below the explicit 256-MiB library/index planning allowance")
    # Only this new CPU diagnostic process's thread pools are configured.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)
    import h5py
    import numpy as np
    import torch

    from asgcn_unet.data.common import normalize_polarity
    from asgcn_unet.data.eventhdr import EventHDRDataset, _validate_event_values
    from asgcn_unet.implicit_radius import ImplicitRadiusIndex
    from asgcn_unet.stream_input import PHYSICAL_EVENT_TIME_CONTRACT, to_physical_seconds

    torch.set_num_threads(cpu_threads)
    path = Path(source_file).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("source_file must be one explicit EventHDR HDF5 file")
    source_stat = path.stat()
    estimates = []

    def guard(estimate, stage):
        estimates.append({"stage": stage, "estimated_working_bytes": estimate})
        if estimate > memory_budget_bytes:
            raise MemoryError(f"{stage}: planned {estimate:,} bytes exceeds budget "
                              f"{memory_budget_bytes:,}; no graph reduction was applied")
        preflight(budget_bytes=memory_budget_bytes, reserve_bytes=reserve_memory_bytes,
                  cpu_threads=cpu_threads)

    guard(overhead, "HDF5 metadata open and library allowance")
    with h5py.File(path, "r", rdcc_nbytes=1024**2) as handle:
        for group in ("events", "images"):
            if (not isinstance(handle.get(group, getlink=True), h5py.HardLink)
                    or not isinstance(handle.get(group), h5py.Group)):
                raise TypeError("EventHDR requires local hard-linked events/images groups")
        arrays = []
        for name in ("xs", "ys", "ts", "ps"):
            if not isinstance(handle["events"].get(name, getlink=True), h5py.HardLink):
                raise TypeError("External/soft HDF5 event links are unsupported")
            array = handle["events"].get(name)
            if not isinstance(array, h5py.Dataset) or array.ndim != 1:
                raise ValueError("EventHDR arrays must be one-dimensional")
            arrays.append(array)
        chunks = sum(math.prod(array.chunks or (1,)) * array.dtype.itemsize for array in arrays)
        metadata_cost = overhead + len(handle["images"]) * 4096 + chunks
        # Existing missing-index recovery reads bounded timestamp chunks.
        guard(metadata_cost + min(len(arrays[2]), 1_048_576) * 128,
              "Single-file index and optional timestamp-index recovery")
        for key in handle["images"]:
            if not isinstance(handle["images"].get(key, getlink=True), h5py.HardLink):
                raise TypeError("External/soft HDF5 image links are unsupported")
    dataset = EventHDRDataset(
        path.parent, allowed_files=[path.name], max_events=None, crop_size=None, random_crop=False,
        event_time_contract=PHYSICAL_EVENT_TIME_CONTRACT,
        timestamp_scale_to_seconds=timestamp_scale_to_seconds,
        interval_timestamp_scale_to_seconds=interval_timestamp_scale_to_seconds,
        **target_options,
    )
    try:
        if frame_index >= len(dataset):
            raise ValueError(f"frame_index must be below selected file's {len(dataset)} frames")
        item = dataset.samples[frame_index]
        end = item["end_idx"]
        handle = dataset._get_handle(path)
        image = handle["images"][item["image_key"]]
        sensor_size = dataset._topology_image_size(image, source=item["image_key"])
        readout = float(to_physical_seconds(
            item["timestamp"], interval_timestamp_scale_to_seconds, source=path.name,
        ))
        guard(metadata_cost + end * 80, "Delivered-prefix timestamps before payload read")
        timestamps = np.asarray(handle["events/ts"][:end])
        rows, seconds = select_window_rows(
            timestamps, timestamp_scale_to_seconds=timestamp_scale_to_seconds,
            readout_seconds=readout, window_seconds=window_seconds,
        )
        raw_timestamps = timestamps[rows].copy()
        del timestamps
        count = len(rows)
        queries = sorted({0, count // 2, count - 1}) if count else []
        if query_indices is not None:
            queries = list(query_indices)
        for query in queries:
            _integer(query, "query index")
        if len(set(queries)) != len(queries) or any(query >= count for query in queries):
            raise ValueError("Query indices must be unique and inside the complete window")
        # Reserve all N-1 possible neighbors per query, including JSON serialization.
        degree_scratch = count * 64 if count_all_nodes else 0
        persistent = metadata_cost + count * (4096 + len(queries) * 1536) + degree_scratch
        guard(persistent, "Complete window, query output, index and oracle")
        start = int(rows[0]) if count else end
        original_xs = np.asarray(handle["events/xs"][start:end])
        original_ys = np.asarray(handle["events/ys"][start:end])
        xs = original_xs.astype(np.float32)
        ys = original_ys.astype(np.float32)
        ps = np.asarray(handle["events/ps"][start:end])
        _validate_event_values(original_xs, original_ys, seconds, ps, expected=count,
                               height=sensor_size[0], width=sensor_size[1], source=path.name)
        events = np.column_stack((xs, ys, seconds, normalize_polarity(ps)))
        positions = normalized_positions(events, sensor_size,
                                         origin_seconds=item["sequence_origin_seconds"],
                                         time_scale_seconds=time_scale_seconds)
        # Candidate chunking limits scratch only, not graph nodes or edges.
        candidate_budget = max(1, min(65_536, (memory_budget_bytes - persistent) // 256))
        guard(persistent + candidate_budget * 256, "Exact selected-query scratch")
        index_started = time.perf_counter()
        index = ImplicitRadiusIndex(
            positions, torch.zeros(count, dtype=torch.long), batch_size=1,
            radius=radius, position_dims=3, chunk_size=128 if count_all_nodes else 1,
            candidate_pair_budget=candidate_budget,
        )
        index_time = time.perf_counter() - index_started
        full_count = None
        degrees = None
        if count_all_nodes:
            guard(persistent + candidate_budget * 256, "All-node degree count scratch (no E storage)")
            degrees, full_count = count_all_degrees(index)
        query_started = time.perf_counter()
        queries_report = query_neighbors(positions, rows, queries, radius=radius,
                                         candidate_pair_budget=candidate_budget, index=index)
        query_time = time.perf_counter() - query_started
        if full_count is not None:
            for query in queries_report:
                if int(degrees[query["node_index"]]) != query["in_degree"]:
                    raise RuntimeError("Full degree count differs from selected independent oracle")
            full_count["selected_oracle_queries_checked"] = len(queries_report)
            full_count["selected_oracle_degree_sum"] = sum(q["in_degree"] for q in queries_report)
            full_count["selected_oracle_degrees_match"] = True
        guard(persistent + candidate_budget * 256, "Raw-to-graph trace JSON construction")
        detailed_nodes = set(queries)
        for query in queries_report:
            detailed_nodes.update(neighbor["node_index"] for neighbor in query["neighbors"])
        node_details = {
            str(node): {
                "node_index": node, "raw_row_id": int(rows[node]),
                "original_source": {"x": original_xs[node].item(),
                                    "y": original_ys[node].item(),
                                    "timestamp": raw_timestamps[node].item(),
                                    "polarity": ps[node].item()},
                "preprocessed": {"x": float(xs[node]), "y": float(ys[node]),
                                 "timestamp_seconds": float(seconds[node])},
                "normalized_topology_coordinates": positions[node, :3].tolist(),
            }
            for node in sorted(detailed_nodes)
        }
        final_stat = path.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (
            source_stat.st_size, source_stat.st_mtime_ns,
        ):
            raise RuntimeError("Selected HDF5 file metadata changed during the diagnostic")
        return {
            "schema": "asgcn_raw_graph_audit_v1", "report_eligible": False,
            "paper_exact": False,
            "total_directed_edges": full_count["directed_edges"] if full_count else None,
            "count_all_nodes": count_all_nodes, "full_count": full_count,
            "scope": ("Full-window production degree count plus selected independent query oracles"
                      if count_all_nodes else
                      "Selected incoming queries against all nodes of one complete delivered window"),
            "source": {"file_name": path.name, "frame_index": frame_index,
                       "size_bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns,
                       "identity": "metadata_only_not_content_verified",
                       "image_key": item["image_key"], "delivery_end_idx_exclusive": end,
                       "event_idx_source": item["event_idx_source"],
                       "validated_timestamp_prefix_rows": end},
            "window": {"nodes": count, "sensor_size": list(sensor_size),
                       "readout_seconds": readout, "cutoff_seconds": readout - window_seconds,
                       "window_seconds": window_seconds, "cutoff_inclusive": True,
                       "first_raw_row_id": int(rows[0]) if count else None,
                       "last_raw_row_id": int(rows[-1]) if count else None,
                       "sampling_factor": 1, "max_events": None, "crop": None},
            "geometry": {"radius": radius, "position_dims": 3,
                          "position_builder": "asgcn_unet.stream_geometry.physical_node_positions",
                         "polarity_in_topology": False, "feature_parity_audited": False,
                         "time_scale_seconds": time_scale_seconds,
                         "sequence_origin_seconds": item["sequence_origin_seconds"],
                         "timestamp_scale_to_seconds": timestamp_scale_to_seconds,
                         "interval_timestamp_scale_to_seconds": interval_timestamp_scale_to_seconds,
                         "clock_arithmetic": "IEEE float64; distinct consecutive timestamp collapse refused",
                         "coordinates": "x/(W-1), y/(H-1), (t-fixed_origin)/time_scale",
                         "predicate": "float64 norm((source-destination)/radius) < 1; no self edges"},
            "queries": queries_report,
            "code_identity": _code_identity(),
            "timings": {"index_build_time_s": index_time, "selected_queries_time_s": query_time,
                        "count_time_s": full_count["count_time_s"] if full_count else None,
                        "audit_elapsed_s": time.perf_counter() - audit_started,
                        "scope": "CPU wall times; excludes report JSON serialization and writing"},
            "node_details": node_details,
            "target_reader_options": target_options,
            "node_details_coverage": (
                "Every requested node and every reported incoming neighbor; unrelated nodes "
                "are omitted from JSON details but were included in the complete-source queries."
            ),
            "resources": {"preflight": resource, "planning_estimates": estimates,
                          "cpu_threads": cpu_threads, "cuda_queried": False,
                          "all_node_degree_scratch_bytes": degree_scratch},
            "limitations": [
                "Selected independent queries are not a full-graph oracle correctness proof.",
                "Total E is measured only with explicit count_all_nodes; that may require O(E) time.",
                "This checks executable geometry, not its identity to a paper's design.",
                "Per-delivered-frame polarity normalization and model feature parity are not audited.",
                "No model, training, quality evaluation, GPU fit or throughput was evaluated.",
                "RAM checks are snapshots and planning estimates, not hard isolation.",
                "Chronology covers the delivered prefix; no whole-dataset validation.",
            ],
        }
    finally:
        dataset.close()


def save_report(report, output, *, workspace):
    """Create a new workspace JSON; no overwrite or arbitrary parent creation."""
    root = Path(workspace).resolve(strict=True)
    path = Path(output)
    path = (root / path).resolve(strict=False) if not path.is_absolute() else path.resolve(strict=False)
    if not path.is_relative_to(root) or path.suffix.lower() != ".json":
        raise ValueError("Output must be a new JSON path inside the workspace")
    if not path.parent.is_dir():
        raise ValueError("Output parent must already exist inside the workspace")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
    return path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--frame-index", required=True, type=int,
                        help="Zero-based selected HDF5 file frame; frame_stride=1")
    for name in ("window-seconds", "time-scale-seconds", "radius",
                 "timestamp-scale-to-seconds", "interval-timestamp-scale-to-seconds"):
        parser.add_argument("--" + name, required=True, type=float)
    parser.add_argument("--cpu-threads", required=True, type=int)
    parser.add_argument("--memory-budget-mib", required=True, type=int)
    parser.add_argument("--reserve-memory-mib", required=True, type=int)
    parser.add_argument("--query-indices", nargs="+", type=int,
                        help="Window-local nodes; omitted selects unique first/middle/last")
    parser.add_argument("--count-all-nodes", action="store_true",
                        help="Opt in to O(E)-time full-window degrees; no full edge list is stored")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output = output.resolve(strict=False)
        if (not output.is_relative_to(PROJECT) or output.suffix.lower() != ".json"
                or output.exists() or not output.parent.is_dir()):
            raise ValueError("Choose a new project JSON in an existing directory")
        if args.count_all_nodes:
            print("Explicit all-node count: one complete selected window, potentially O(E) time; "
                  "bounded scratch, no full edge list, no training/GPU.", flush=True)
        report = audit_raw_event_graph(
            source_file=args.source_file, frame_index=args.frame_index,
            window_seconds=args.window_seconds, time_scale_seconds=args.time_scale_seconds,
            radius=args.radius, timestamp_scale_to_seconds=args.timestamp_scale_to_seconds,
            interval_timestamp_scale_to_seconds=args.interval_timestamp_scale_to_seconds,
            cpu_threads=args.cpu_threads, memory_budget_bytes=args.memory_budget_mib * 1024**2,
            reserve_memory_bytes=args.reserve_memory_mib * 1024**2, query_indices=args.query_indices,
            count_all_nodes=args.count_all_nodes,
        )
        saved = save_report(report, output, workspace=PROJECT)
        print(f"Complete window: {report['window']['nodes']:,} nodes; "
              f"{len(report['queries'])} selected queries matched the independent oracle.")
        edge_label = (f"{report['total_directed_edges']:,} directed"
                      if report["count_all_nodes"] else "not measured")
        print(f"Total graph edges: {edge_label}. Paper equivalence: not established. Report: {saved}")
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        print(f"Raw graph audit failed: {error}", file=sys.stderr)
        print("No training, GPU selection, graph reduction or source/result overwrite occurred.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
