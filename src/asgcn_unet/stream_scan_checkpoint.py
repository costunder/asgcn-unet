"""Exact count-only scan checkpoints, never model/training checkpoints.

Identity binds source/config/data/schedule and immutable report identity, not
changing progress JSON. Cursor means completed physical batches. Callers must
pass the last complete state, never a partially mutated batch. Old JSON without
raw lane state cannot be migrated. RAM checks are planning, not hard isolation.
Importing this module does not import torch or initialize an accelerator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .artifact_lock import exclusive_artifact_writer

SCHEMA = "asgcn_stream_scan_checkpoint_v1"
IDENTITY_FIELDS = {"source_sha256", "config_sha256", "data_sha256", "schedule_sha256", "report_sha256"}
STATE_FIELDS = {"positions", "timestamps", "origin_seconds", "watermark_seconds", "sequence_index",
                "sequence_identity", "contract", "directed_edges"}
OPTIONAL_STATE_FIELDS = {"sampling_offset", "last_event_id"}


def _encoded(value):
    return json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                            allow_nan=False).iterencode(value)


def _digest(value):
    digest = hashlib.sha256()
    for piece in _encoded(value):
        digest.update(piece.encode("utf-8"))
    return digest.hexdigest()


def _identity(value):
    if (not isinstance(value, dict) or set(value) != IDENTITY_FIELDS
            or any(not isinstance(item, str) or not re.fullmatch("[0-9a-f]{64}", item)
                   for item in value.values())):
        raise ValueError("Scan checkpoint requires exact source/config/data/schedule/report SHA256 identity")
    return dict(value)


def _bytes(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(f"{name} must be an explicit finite nonnegative/positive MiB budget")
    result = int(value * 1024**2)
    if not zero and result < 1:
        raise ValueError(f"{name} must allow at least one byte")
    return result


def _regular(path):
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Scan checkpoint requires a regular non-symlink file: {path.name}")
    return path


def _directory(path):
    result = Path(path).absolute()
    if result == result.parent or result.is_symlink() or result.resolve() != result:
        raise ValueError("Scan checkpoint requires an explicit non-symlink directory")
    return result


def _read_json(path, budget):
    _regular(path)
    if path.stat().st_size > budget:
        raise MemoryError("Scan checkpoint JSON exceeds the explicit memory budget")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        for piece in _encoded(value):
            handle.write(piece)
        handle.flush()
        os.fsync(handle.fileno())


def _file_record(path):
    digest = hashlib.sha256()
    with _regular(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _headroom(required, budget, reserve):
    from .diagnostic_resources import _snapshot

    if required > budget:
        raise MemoryError(f"Scan checkpoint needs {required:,} planned CPU bytes; budget={budget:,}. No state was reduced.")
    measured = _snapshot()
    if required + reserve > measured["headroom_bytes"]:
        raise MemoryError("Insufficient measured CPU RAM headroom for exact scan checkpoint plus reserve")
    return {"required_bytes": required, "budget_bytes": budget, "reserve_bytes": reserve,
            "headroom_bytes": measured["headroom_bytes"], "hard_memory_isolation": False,
            "scope": "incremental_checkpoint_tensors_and_conservative_JSON_workspace_not_full_process_peak"}


def _owner(directory, identity, budget, *, create=False):
    path = directory / "owner.json"
    expected = {"schema": SCHEMA + "_owner", "identity": identity, "identity_sha256": _digest(identity)}
    if create and not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        _write_json(path, expected)
    if not path.exists():
        raise ValueError("No exact raw-state scan checkpoint owner exists; partial JSON cannot be resumed or migrated")
    if _read_json(path, budget) != expected:
        raise ValueError("Scan checkpoint directory belongs to a different experiment identity")


def inspect_scan_checkpoint(checkpoint_dir, *, expected_identity=None, memory_budget_mib):
    """Verify manifest and immutable file hashes without importing/loading torch."""
    directory, budget = _directory(checkpoint_dir), _bytes(memory_budget_mib, "memory_budget_mib")
    if not (directory / "latest.json").exists():
        raise ValueError("No committed raw-state scan checkpoint; old partial JSON is not resumable")
    manifest = _read_json(directory / "latest.json", budget)
    fields = {"schema", "identity", "identity_sha256", "generation", "completed_batches",
              "metadata", "tensors", "commitment_sha256"}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest["schema"] != SCHEMA:
        raise ValueError("Invalid scan checkpoint manifest schema")
    identity = _identity(manifest["identity"])
    if expected_identity is not None and identity != _identity(expected_identity):
        raise ValueError("Scan checkpoint source/config/data/schedule/report identity mismatch")
    _owner(directory, identity, budget)
    commitment = {key: value for key, value in manifest.items() if key != "commitment_sha256"}
    if manifest["identity_sha256"] != _digest(identity) or manifest["commitment_sha256"] != _digest(commitment):
        raise ValueError("Scan checkpoint manifest commitment mismatch")
    if not isinstance(manifest["generation"], str) or not re.fullmatch("[0-9a-f]{32}", manifest["generation"]):
        raise ValueError("Invalid scan checkpoint generation")
    if type(manifest["completed_batches"]) is not int or manifest["completed_batches"] < 0:
        raise ValueError("Invalid completed scan batch cursor")
    for field, suffix in (("metadata", ".json"), ("tensors", ".pt")):
        record = manifest[field]
        if (not isinstance(record, dict) or set(record) != {"filename", "size_bytes", "sha256"}
                or record["filename"] != manifest["generation"] + suffix
                or type(record["size_bytes"]) is not int or record["size_bytes"] < 0
                or record["size_bytes"] > budget):
            raise ValueError("Invalid or over-budget scan checkpoint file record")
        if _file_record(directory / record["filename"]) != record:
            raise ValueError(f"Scan checkpoint {field} file hash/size mismatch")
    return manifest


def _finals(value):
    if not isinstance(value, dict):
        raise TypeError("sequence_final_indices must map sequence identities to final indices")
    result = []
    for key, index in value.items():
        if (not isinstance(key, tuple) or len(key) != 2 or not all(isinstance(part, str) for part in key)
                or not key[0] or type(index) is not int or index < 0):
            raise ValueError("Invalid sequence final-index contract")
        result.append([list(key), index])
    return sorted(result)


def _validate_prefix(identity, batches, completed, topology, final_indices):
    if not isinstance(batches, (list, tuple)) or any(not isinstance(batch, (list, tuple)) or not batch for batch in batches):
        raise ValueError("Scan checkpoint requires the complete physical-batch schedule")
    if _digest(batches) != identity["schedule_sha256"]:
        raise ValueError("Scan checkpoint schedule commitment mismatch")
    if type(completed) is not int or not 0 <= completed <= len(batches):
        raise ValueError("Scan checkpoint cursor is outside the complete batch schedule")
    flattened = [index for batch in batches for index in batch]
    if any(type(index) is not int for index in flattened) or sorted(flattened) != list(range(len(flattened))):
        raise ValueError("Scan checkpoint schedule must cover every dataset index exactly once")
    if not isinstance(topology, dict) or not isinstance(topology.get("samples"), list):
        raise TypeError("Scan checkpoint needs exact topology records, not a summary-only JSON")
    records = topology["samples"]
    seen = {index for batch in batches[:completed] for index in batch}
    if (len(records) != len(flattened) or topology.get("dataset_samples") != len(records)
            or topology.get("scanned_samples") != len(seen)
            or (topology.get("scan_complete") is True and completed != len(batches))):
        raise ValueError("Topology report and completed batch cursor disagree")
    latest = {}
    sampling = topology.get("sampling_contract")
    if sampling is not None and (
            not isinstance(sampling, dict) or set(sampling) != {"factor", "ordinal_origin", "counter", "reset"}
            or type(sampling["factor"]) is not int or not 1 <= sampling["factor"] < 2**63
            or sampling["ordinal_origin"] != 0 or sampling["counter"] != "all_raw_events"
            or sampling["reset"] != "sequence_start_only"):
        raise ValueError("Invalid exact scan sampling contract")
    for number, batch in enumerate(batches):
        batch_keys = set()
        for index in batch:
            record = records[index]
            if number >= completed:
                if record is not None:
                    raise ValueError("Topology contains records past its completed batch cursor")
                continue
            if not isinstance(record, dict) or record.get("dataset_index") != index:
                raise ValueError("Missing/mismatched completed topology record")
            key = record.get("sequence_identity")
            if (not isinstance(key, (list, tuple)) or len(key) != 2
                    or not all(isinstance(part, str) for part in key)):
                raise ValueError("Invalid topology sequence identity")
            key = tuple(key)
            seq_index = record.get("sequence_index")
            if (key not in final_indices or type(seq_index) is not int or not 0 <= seq_index <= final_indices[key]
                    or key in batch_keys or (key in latest and seq_index != latest[key]["sequence_index"] + 1)):
                raise ValueError("Completed topology does not follow independent chronological sequences")
            start, end = record.get("interval_start_seconds"), record.get("interval_end_seconds")
            if (any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
                    for item in (start, end)) or end < start
                    or (key in latest and latest[key]["interval_end_seconds"] > start)):
                raise ValueError("Completed topology has invalid/overlapping physical clocks")
            if sampling is not None:
                prior = latest[key]["sampling_offset_after"] if key in latest else 0
                incoming = record.get("incoming_events")
                selected = record.get("sampled_incoming_events")
                before, after = record.get("sampling_offset_before"), record.get("sampling_offset_after")
                if (any(type(value) is not int or value < 0 for value in (incoming, selected, before, after))
                        or before != prior or after != prior + incoming):
                    raise ValueError("Exact scan sampling counter is missing or discontinuous")
                first = (-before) % sampling["factor"]
                expected_selected = 0 if first >= incoming else 1 + (incoming - 1 - first) // sampling["factor"]
                if selected != expected_selected:
                    raise ValueError("Exact scan sampled count differs from sequence-global ordinal selection")
                last_id = _last_id(record.get("raw_last_event_id"))
                previous_id = _last_id(latest[key].get("raw_last_event_id")) if key in latest else None
                if ((incoming and (last_id is None or (previous_id is not None and last_id <= previous_id)))
                        or (not incoming and last_id != previous_id)):
                    raise ValueError("Exact scan raw event identity endpoint is missing or discontinuous")
            batch_keys.add(key)
            latest[key] = record
            for name in ("readout_nodes", "readout_directed_edges", "prefix_union_nodes_upper_bound",
                         "prefix_union_directed_edges_upper_bound"):
                if type(record.get(name)) is not int or record[name] < 0:
                    raise ValueError("Invalid exact topology node/edge count")
            nodes, edges = record["readout_nodes"], record["readout_directed_edges"]
            if (edges % 2 or edges > nodes * max(nodes - 1, 0)
                    or nodes > record["prefix_union_nodes_upper_bound"]
                    or edges > record["prefix_union_directed_edges_upper_bound"]
                    or record["prefix_union_directed_edges_upper_bound"] % 2
                    or record["prefix_union_directed_edges_upper_bound"] > record["prefix_union_nodes_upper_bound"]
                    * max(record["prefix_union_nodes_upper_bound"] - 1, 0)):
                raise ValueError("Impossible exact topology node/edge counts")
    return {key: row for key, row in latest.items() if row["sequence_index"] < final_indices[key]}


def _last_id(value):
    if value is not None and (not isinstance(value, (list, tuple)) or len(value) != 2
                              or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError("Invalid raw scan last_event_id")
    return None if value is None else tuple(value)


def _validate_state(key, fields, positions, timestamps, row):
    import torch

    count = row["readout_nodes"]
    base_fields = STATE_FIELDS - {"positions", "timestamps"}
    if (not isinstance(fields, dict) or not base_fields <= set(fields) <= base_fields | OPTIONAL_STATE_FIELDS
            or tuple(fields["sequence_identity"]) != key or fields["sequence_index"] != row["sequence_index"]
            or type(fields["sequence_index"]) is not int or type(fields["directed_edges"]) is not int
            or fields["directed_edges"] != row["readout_directed_edges"]
            or fields["watermark_seconds"] != row["interval_end_seconds"]
            or not isinstance(fields["contract"], str) or not re.fullmatch("[0-9a-f]{64}", fields["contract"])):
        raise ValueError("Raw scan lane state disagrees with its latest completed topology record")
    offset = fields.get("sampling_offset", 0)
    if (type(offset) is not int or offset < 0
            or ("sampling_offset_after" in row and ("sampling_offset" not in fields or offset != row["sampling_offset_after"]))):
        raise ValueError("Raw scan sampling_offset disagrees with its committed raw count")
    last_id = _last_id(fields.get("last_event_id"))
    if "raw_last_event_id" in row and ("last_event_id" not in fields or last_id != _last_id(row["raw_last_event_id"])):
        raise ValueError("Raw scan last_event_id disagrees with its committed endpoint")
    if (not isinstance(positions, torch.Tensor) or not isinstance(timestamps, torch.Tensor)
            or positions.dtype != torch.float64 or timestamps.dtype != torch.float64
            or positions.shape != (count, 4) or timestamps.shape != (count,)
            or positions.device.type != "cpu" or timestamps.device.type != "cpu"
            or positions.layout != torch.strided or timestamps.layout != torch.strided
            or positions.requires_grad or timestamps.requires_grad):
        raise ValueError("Raw scan state requires CPU float64 positions[N,4] and timestamps[N]")
    origin, watermark = fields["origin_seconds"], fields["watermark_seconds"]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
           for item in (origin, watermark)) or origin > watermark or origin > row["interval_start_seconds"]:
        raise ValueError("Invalid raw scan state physical clock")
    if not bool(torch.stack((torch.isfinite(positions).all(), torch.isfinite(timestamps).all(),
                            (timestamps >= origin).all(), (timestamps <= watermark).all(),
                            (timestamps[1:] >= timestamps[:-1]).all())).all()):
        raise ValueError("Raw scan state contains nonfinite/reversed/out-of-range physical timestamps")


def save_scan_checkpoint(checkpoint_dir, *, identity, batches, completed_batches, states, topology,
                         sequence_final_indices, memory_budget_mib, reserve_memory_mib=0):
    """Commit exactly one complete batch prefix; keep the previous commit on failure."""
    identity = _identity(identity)
    budget, reserve = _bytes(memory_budget_mib, "memory_budget_mib"), _bytes(reserve_memory_mib, "reserve_memory_mib", zero=True)
    finals = _finals(sequence_final_indices)
    live = _validate_prefix(identity, batches, completed_batches, topology, sequence_final_indices)
    if not isinstance(states, dict) or set(states) != set(live):
        raise ValueError("Raw scan checkpoint must contain every unfinished lane and no completed lanes")
    metadata = {"schema": SCHEMA + "_state", "identity": identity, "completed_batches": completed_batches,
                "sequence_final_indices": finals, "topology": topology, "states": []}
    tensor_bytes = 0
    for key in sorted(states):
        state = states[key]
        names = set(vars(state))
        if not STATE_FIELDS <= names <= STATE_FIELDS | OPTIONAL_STATE_FIELDS:
            raise ValueError("Only exact raw count-only scan state can be checkpointed")
        fields = {name: getattr(state, name) for name in sorted(names - {"positions", "timestamps"})}
        fields["sequence_identity"] = list(fields["sequence_identity"])
        metadata["states"].append({"key": list(key), "fields": fields, "nodes": live[key]["readout_nodes"]})
        tensor_bytes += live[key]["readout_nodes"] * 40
    metadata_bytes = sum(len(piece.encode("utf-8")) for piece in _encoded(metadata))
    _headroom(2 * tensor_bytes + 8 * metadata_bytes, budget, reserve)
    import torch

    tensors = {}
    for index, (key, state) in enumerate(sorted(states.items())):
        count = live[key]["readout_nodes"]
        if (not isinstance(state.positions, torch.Tensor) or not isinstance(state.timestamps, torch.Tensor)
                or state.positions.shape != (count, 4) or state.timestamps.shape != (count,)
                or state.positions.dtype != torch.float64 or state.timestamps.dtype != torch.float64
                or state.positions.layout != torch.strided or state.timestamps.layout != torch.strided
                or state.positions.requires_grad or state.timestamps.requires_grad):
            raise ValueError("Raw scan source tensor shape/dtype must match the budgeted topology before copying")
        positions = state.positions.detach().to(device="cpu", copy=True).contiguous()
        timestamps = state.timestamps.detach().to(device="cpu", copy=True).contiguous()
        _validate_state(key, metadata["states"][index]["fields"], positions, timestamps, live[key])
        tensors[f"positions_{index}"] = positions
        tensors[f"timestamps_{index}"] = timestamps
    directory = _directory(checkpoint_dir)
    with exclusive_artifact_writer(directory):
        _owner(directory, identity, budget, create=True)
        previous = inspect_scan_checkpoint(directory, expected_identity=identity, memory_budget_mib=memory_budget_mib) if (directory / "latest.json").exists() else None
        if previous is not None and completed_batches < previous["completed_batches"]:
            raise ValueError("Scan checkpoint cursor cannot move backwards")
        generation = uuid4().hex
        tensor_path, metadata_path = directory / f"{generation}.pt", directory / f"{generation}.json"
        # Unique orphan files on interruption are ignored, never cleaned by glob.
        with tensor_path.open("xb") as handle:
            torch.save(tensors, handle)
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(metadata_path, metadata)
        manifest = {"schema": SCHEMA, "identity": identity, "identity_sha256": _digest(identity),
                    "generation": generation, "completed_batches": completed_batches,
                    "metadata": _file_record(metadata_path), "tensors": _file_record(tensor_path)}
        manifest["commitment_sha256"] = _digest(manifest)
        temporary = directory / f"{generation}.manifest.tmp"
        _write_json(temporary, manifest)
        os.replace(temporary, directory / "latest.json")
        if previous is not None:
            for field in ("metadata", "tensors"):
                old_path = directory / previous[field]["filename"]
                if _file_record(old_path) != previous[field]:
                    raise ValueError("Previous checkpoint changed after publication; it was preserved")
                old_path.unlink()
        return manifest


def load_scan_checkpoint(checkpoint_dir, *, expected_identity, batches, sequence_final_indices,
                         memory_budget_mib, reserve_memory_mib=0):
    """Restore CPU-owned exact states; no source migration or missing-state fallback."""
    identity = _identity(expected_identity)
    budget, reserve = _bytes(memory_budget_mib, "memory_budget_mib"), _bytes(reserve_memory_mib, "reserve_memory_mib", zero=True)
    directory = _directory(checkpoint_dir)
    with exclusive_artifact_writer(directory):
        manifest = inspect_scan_checkpoint(directory, expected_identity=identity, memory_budget_mib=memory_budget_mib)
        metadata_path, tensor_path = (directory / manifest[name]["filename"] for name in ("metadata", "tensors"))
        _headroom(8 * manifest["metadata"]["size_bytes"] + 2 * manifest["tensors"]["size_bytes"], budget, reserve)
        metadata = _read_json(metadata_path, budget)
        expected = {"schema", "identity", "completed_batches", "sequence_final_indices", "topology", "states"}
        if (not isinstance(metadata, dict) or set(metadata) != expected or metadata["schema"] != SCHEMA + "_state"
                or metadata["identity"] != identity or metadata["completed_batches"] != manifest["completed_batches"]
                or metadata["sequence_final_indices"] != _finals(sequence_final_indices)):
            raise ValueError("Invalid/mismatched scan checkpoint state metadata")
        live = _validate_prefix(identity, batches, metadata["completed_batches"], metadata["topology"], sequence_final_indices)
        rows = metadata["states"]
        if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {"key", "fields", "nodes"}
                                            for row in rows)):
            raise ValueError("Invalid raw scan lane metadata")
        keys = [tuple(row["key"]) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(live):
            raise ValueError("Raw scan checkpoint is missing an unfinished lane or duplicates a lane")
        if any(type(row["nodes"]) is not int or row["nodes"] != live[key]["readout_nodes"]
               for row, key in zip(rows, keys, strict=True)):
            raise ValueError("Raw scan state node counts differ from completed topology")
        with zipfile.ZipFile(tensor_path) as archive:
            members = archive.infolist()
            if any(member.compress_type != zipfile.ZIP_STORED for member in members):
                raise ValueError("Compressed scan tensor archives are not accepted")
            if sum(member.file_size for member in members) > budget:
                raise MemoryError("Scan tensor archive exceeds the explicit CPU memory budget")
        import torch

        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(tensors, dict) or set(tensors) != {f"{name}_{index}" for index in range(len(rows))
                                                           for name in ("positions", "timestamps")}:
            raise ValueError("Scan checkpoint contains unexpected tensors or missing raw state")
        states = {}
        for index, (key, row) in enumerate(zip(keys, rows, strict=True)):
            positions, timestamps = tensors[f"positions_{index}"], tensors[f"timestamps_{index}"]
            _validate_state(key, row["fields"], positions, timestamps, live[key])
            fields = dict(row["fields"], sequence_identity=key)
            fields.setdefault("sampling_offset", 0)
            fields["last_event_id"] = _last_id(fields.get("last_event_id"))
            states[key] = SimpleNamespace(**fields, positions=positions.clone(), timestamps=timestamps.clone())
        return {"states": states, "topology": metadata["topology"],
                "completed_batches": metadata["completed_batches"], "manifest": manifest}
