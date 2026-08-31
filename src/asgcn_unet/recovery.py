"""Explicit, recoverable handling of a failed run without an epoch checkpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def archive_uncheckpointed_run(
    run_dir: str | Path, project_root: str | Path
) -> Path | None:
    """Preserve metadata-only output before an explicitly requested fresh start.

    The caller must stop the old job first. A metadata-only directory proves that
    no epoch checkpoint is present, not that no optimizer updates ever occurred.
    No data, report, checkpoint, unknown file, or directory is removed.
    """
    original = Path(run_dir).expanduser()
    if original.is_symlink() or getattr(original, "is_junction", lambda: False)():
        raise ValueError("Refusing to archive a linked training directory")
    target = original.resolve()
    project = Path(project_root).resolve()
    if target in {Path(target.anchor), Path.home().resolve(), project} or target in project.parents:
        raise ValueError("Refusing to archive a broad directory as training output")
    if not target.exists():
        return None
    if not target.is_dir():
        raise ValueError("Training output is not a directory")
    allowed = {"config.json", "preflight_gate.json", ".data_hash_cache.json"}
    entries = list(target.iterdir())
    if not entries:
        return None
    for entry in entries:
        if entry.name not in allowed or entry.is_symlink() or not entry.is_file():
            raise ValueError(
                "Training output contains checkpoints, history, or unknown entries; "
                "use checkpoint resume or inspect it manually, not a fresh restart"
            )
    config_path = target / "config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if not isinstance(saved, dict) or any(
            not isinstance(saved.get(section), dict) for section in ("model", "dataset", "train")
        ):
            raise ValueError("Training metadata does not contain a recognizable run configuration")
    # A unique sibling container prevents POSIX rename from replacing an existing
    # destination. The original directory and every byte inside it are retained.
    container = Path(tempfile.mkdtemp(prefix=f"{target.name}.failed-", dir=target.parent))
    destination = container / target.name
    target.rename(destination)
    return destination
