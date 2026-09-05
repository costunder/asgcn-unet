#!/usr/bin/env python3
"""Scan repository text without checking private markers into the repository.

The built-in rules catch user-specific home-directory paths and labelled identity
values. Project-specific strings belong in an external denylist or in the
``PRIVATE_MARKERS_B64`` environment variable, never in a tracked test fixture.
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatternSpec:
    name: str
    expression: re.Pattern[str]


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    rule: str
    representation: str
    object_id: str = ""


def _generic_patterns() -> tuple[PatternSpec, ...]:
    slash = chr(47)
    backslash = chr(92)
    unix_home = re.escape(slash + "home" + slash)
    mac_home = re.escape(slash + "Users" + slash)
    windows_users = re.escape(":" + backslash + "Users" + backslash)
    windows_users_slash = re.escape(":" + slash + "Users" + slash)
    identity_labels = "(?:user(?:name)?|account|host(?:name)?)"
    path_component = r"(?![<$({])[A-Za-z0-9._-]{1,128}"

    return (
        PatternSpec(
            "generic-unix-home",
            re.compile(r"(?i)(?<![A-Za-z0-9_])" + unix_home + path_component + slash),
        ),
        PatternSpec(
            "generic-macos-home",
            re.compile(r"(?i)(?<![A-Za-z0-9_])" + mac_home + path_component + slash),
        ),
        PatternSpec(
            "generic-windows-home",
            re.compile(
                r"(?i)\b[A-Z](?:"
                + windows_users
                + "|"
                + windows_users_slash
                + ")"
                + path_component
                + r"[\\/]"
            ),
        ),
        PatternSpec(
            "generic-labelled-identity",
            re.compile(
                r"(?i)\b"
                + identity_labels
                + r"\s*[:=]\s*[\"']?(?![<$({])[A-Za-z0-9._-]{2,128}"
            ),
        ),
    )


def decode_text_bytes(data: bytes) -> str | None:
    """Return normalized UTF-8 text, or ``None`` for binary/non-UTF-8 data."""

    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None

    if text:
        control_count = sum(ord(character) < 32 and character not in "\n\r\t\f" for character in text)
        if control_count / len(text) > 0.01:
            return None
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_marker_lines(payload: str, source_name: str) -> list[PatternSpec]:
    patterns: list[PatternSpec] = []
    marker_index = 0
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        marker_index += 1
        rule_name = f"external-{source_name}-{marker_index}"
        if line.startswith("regex:"):
            expression = line.removeprefix("regex:").strip()
            if not expression:
                raise ValueError(f"empty regular expression in {source_name}")
            try:
                compiled = re.compile(expression)
            except re.error as error:
                raise ValueError(f"invalid regular expression in {source_name}") from error
        else:
            if len(line) < 4:
                raise ValueError(f"literal markers in {source_name} must contain at least 4 characters")
            compiled = re.compile(re.escape(line))
        if compiled.search("") is not None:
            raise ValueError(f"regular expressions in {source_name} must not match empty text")
        patterns.append(PatternSpec(rule_name, compiled))
    return patterns


def _load_external_patterns(paths: Sequence[Path], environment: dict[str, str]) -> list[PatternSpec]:
    patterns: list[PatternSpec] = []
    for index, path in enumerate(paths, start=1):
        try:
            payload = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(f"cannot read external denylist #{index}") from error
        patterns.extend(_parse_marker_lines(payload, f"file-{index}"))

    encoded = environment.get("PRIVATE_MARKERS_B64", "").strip()
    if encoded:
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as error:
            raise ValueError("PRIVATE_MARKERS_B64 must be base64-encoded UTF-8") from error
        patterns.extend(_parse_marker_lines(decoded, "environment"))
    return patterns


def _constant_string(node: ast.AST, bindings: dict[str, str] | None = None) -> str | None:
    bindings = bindings or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left, bindings)
        right = _constant_string(node.right, bindings)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                part = _constant_string(value.value, bindings)
            else:
                part = _constant_string(value, bindings)
            if part is None:
                return None
            parts.append(part)
        return "".join(parts)
    return None


def _python_constant_representations(text: str) -> Iterator[tuple[str, int, str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return

    bindings: dict[str, str] = {}
    for _ in range(2):
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value_node = node.value
            if value_node is None:
                continue
            value = _constant_string(value_node, bindings)
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = value

    seen: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        value = _constant_string(node, bindings)
        if value is None or len(value) < 4:
            continue
        key = (getattr(node, "lineno", 1), value)
        if key in seen:
            continue
        seen.add(key)
        yield value, key[0], "python-constant"


_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])")


def _base64_representations(text: str) -> Iterator[tuple[str, int, str]]:
    for match in _BASE64_TOKEN.finditer(text):
        token = match.group(0)
        if len(token) % 4:
            continue
        try:
            decoded_bytes = base64.b64decode(token, validate=True)
            decoded = decoded_bytes.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            continue
        if len(decoded) < 8:
            continue
        printable = sum(character.isprintable() or character in "\n\r\t" for character in decoded)
        if printable / len(decoded) < 0.95:
            continue
        line = text.count("\n", 0, match.start()) + 1
        yield decoded, line, "base64-decoded"


def _representations(
    path: Path,
    text: str,
    *,
    parse_python: bool = False,
) -> Iterator[tuple[str, int, str]]:
    yield text, 1, "raw"
    if parse_python or path.suffix.casefold() == ".py":
        for value, line, representation in _python_constant_representations(text):
            yield value, line, representation
            for decoded, relative_line, _ in _base64_representations(value):
                yield decoded, line + relative_line - 1, "python-constant-base64-decoded"
    yield from _base64_representations(text)


def _scan_text(
    path_label: str,
    path: Path,
    text: str,
    patterns: Sequence[PatternSpec],
    *,
    object_id: str = "",
    parse_python: bool = False,
) -> list[Finding]:
    findings: set[Finding] = set()
    for representation, base_line, representation_name in _representations(
        path,
        text,
        parse_python=parse_python,
    ):
        for pattern in patterns:
            for match in pattern.expression.finditer(representation):
                relative_line = representation.count("\n", 0, match.start())
                findings.add(
                    Finding(
                        path=path_label,
                        line=base_line + relative_line,
                        rule=pattern.name,
                        representation=representation_name,
                        object_id=object_id,
                    )
                )
    return sorted(findings)


def _git_root(start: Path) -> Path:
    process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ValueError("cannot locate a Git work tree")
    return Path(os.fsdecode(process.stdout).strip()).resolve()


def _tracked_paths(root: Path) -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ValueError("git ls-files failed")
    return [Path(os.fsdecode(item)) for item in process.stdout.split(b"\x00") if item]


def _history_git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PRIVATE_MARKERS_B64", None)
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _require_complete_history(root: Path) -> None:
    process = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=root,
        env=_history_git_environment(),
        check=False,
        capture_output=True,
    )
    state = process.stdout.strip()
    if process.returncode != 0 or state not in {b"true", b"false"}:
        raise ValueError("cannot determine whether the Git history is complete")
    if state == b"true":
        raise ValueError("--all-history requires a complete, non-shallow repository")


def _reachable_objects(root: Path) -> dict[str, str]:
    """Return reachable object IDs and diagnostic-only path hints across Git versions."""

    process = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root,
        env=_history_git_environment(),
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ValueError("git history enumeration failed")

    objects: dict[str, str] = {}
    # Older Git versions accept ``-z`` without using the newer NUL/object-metadata
    # format. The ordinary LF-delimited output is supported by both old and new
    # versions. Split only on LF: other control bytes can occur in a path hint.
    # Git may truncate a hint at an embedded LF; blob content is always read by
    # object ID below, never by this potentially incomplete/ambiguous hint.
    for record in process.stdout.split(b"\n"):
        if not record:
            continue
        object_bytes, separator, inline_path = record.partition(b" ")
        try:
            object_id = object_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("git history enumeration returned an invalid object ID") from error
        if re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", object_id) is None:
            raise ValueError("git history enumeration returned an invalid object ID")
        object_id = object_id.casefold()
        objects.setdefault(object_id, "")
        if separator:
            objects[object_id] = os.fsdecode(inline_path)
    return objects


def _reachable_blob_ids(root: Path, objects: dict[str, str]) -> list[str]:
    if not objects:
        return []
    payload = "".join(f"{object_id}\n" for object_id in objects).encode("ascii")
    process = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=root,
        env=_history_git_environment(),
        input=payload,
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise ValueError("git object classification failed")

    lines = process.stdout.splitlines()
    if len(lines) != len(objects):
        raise ValueError("git object classification returned an incomplete result")
    blob_ids: list[str] = []
    for expected_id, line in zip(objects, lines, strict=True):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError("git object classification returned an invalid result")
        actual_id = os.fsdecode(fields[0]).casefold()
        object_type = os.fsdecode(fields[1])
        if actual_id != expected_id:
            raise ValueError("git object classification returned an unexpected object ID")
        if object_type == "blob":
            blob_ids.append(actual_id)
        elif object_type not in {"commit", "tag", "tree"}:
            raise ValueError("git history contains an unavailable object")
    return blob_ids


def _stdin_paths() -> list[Path]:
    return [Path(os.fsdecode(item)) for item in sys.stdin.buffer.read().split(b"\x00") if item]


def _path_label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _redact_output(value: str, patterns: Sequence[PatternSpec]) -> str:
    redacted = value
    for pattern in patterns:
        redacted = pattern.expression.sub("<redacted>", redacted)
    return json.dumps(redacted, ensure_ascii=False)[1:-1]


def _resolve_candidates(root: Path, candidates: Iterable[Path]) -> list[Path]:
    resolved: dict[str, Path] = {}
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        key = os.path.normcase(str(path.resolve(strict=False)))
        resolved[key] = path
    return [resolved[key] for key in sorted(resolved)]


def scan_paths(
    root: Path,
    paths: Sequence[Path],
    patterns: Sequence[PatternSpec],
) -> tuple[list[Finding], int, int]:
    findings: list[Finding] = []
    text_count = 0
    binary_count = 0
    for path in paths:
        path_label = _path_label(path, root)
        if path.is_symlink():
            data = os.readlink(path).encode("utf-8", errors="surrogateescape")
        else:
            try:
                data = path.read_bytes()
            except OSError as error:
                safe_label = _redact_output(path_label, patterns)
                raise ValueError(f"cannot read tracked candidate {safe_label}") from error
        text = decode_text_bytes(data)
        if text is None:
            binary_count += 1
            continue
        text_count += 1
        findings.extend(_scan_text(path_label, path, text, patterns))
    return sorted(set(findings)), text_count, binary_count


def _finish_owned_history_reader(process: subprocess.Popen[bytes], owned_pid: int) -> None:
    """Finish this scan's verified child, preferring EOF over any targeted signal."""
    command = ["git", "cat-file", "--batch"]

    def verify_and_report(reason: str) -> None:
        if (
            type(owned_pid) is not int
            or owned_pid <= 0
            or owned_pid in {os.getpid(), os.getppid()}
            or process.pid != owned_pid
            or process.args != command
        ):
            raise RuntimeError("Refusing cleanup of an unverified history-reader process")
        print(
            f"History-reader cleanup: PID={owned_pid}, command=git cat-file --batch; {reason}",
            file=sys.stderr,
        )

    verify_and_report("scan failed; closing only this child's pipes to request normal completion")
    try:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    except BrokenPipeError:
        print("History-reader stdin was already closed by the child.", file=sys.stderr)
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        verify_and_report("no completion after EOF and 5 seconds; terminating only this owned child")
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        verify_and_report("targeted termination did not finish within 5 seconds; forcing only this owned child")
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)


def scan_history(
    root: Path,
    patterns: Sequence[PatternSpec],
) -> tuple[list[Finding], int, int]:
    """Scan each unique text blob reachable from any Git ref without checking it out."""

    _require_complete_history(root)
    objects = _reachable_objects(root)
    blob_ids = _reachable_blob_ids(root, objects)
    findings: list[Finding] = []
    text_count = 0
    binary_count = 0
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        env=_history_git_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    owned_pid = process.pid
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for object_id in blob_ids:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            fields = header.split()
            if len(fields) != 3:
                raise ValueError("git blob reader returned an invalid header")
            actual_id = os.fsdecode(fields[0]).casefold()
            object_type = os.fsdecode(fields[1])
            try:
                size = int(fields[2])
            except ValueError as error:
                raise ValueError("git blob reader returned an invalid size") from error
            if actual_id != object_id or object_type != "blob" or size < 0:
                raise ValueError("git blob reader returned an unexpected object")
            data = process.stdout.read(size)
            terminator = process.stdout.read(1)
            if len(data) != size or terminator != b"\n":
                raise ValueError("git blob reader returned incomplete content")

            text = decode_text_bytes(data)
            if text is None:
                binary_count += 1
                continue
            text_count += 1
            raw_path = objects[object_id] or "<unnamed-ref-object>"
            findings.extend(
                _scan_text(
                    raw_path,
                    Path(raw_path),
                    text,
                    patterns,
                    object_id=object_id,
                    parse_python=True,
                )
            )
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError("git blob reader failed")
    except BaseException as error:
        try:
            _finish_owned_history_reader(process, owned_pid)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as cleanup_error:
            raise error from cleanup_error
        raise
    return sorted(set(findings)), text_count, binary_count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="explicit files to scan")
    parser.add_argument("--all-tracked", action="store_true", help="scan every git-tracked file")
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="scan each unique blob reachable from every local Git ref",
    )
    parser.add_argument(
        "--files-from-stdin",
        action="store_true",
        help="read NUL-delimited file names from standard input",
    )
    parser.add_argument(
        "--extra-patterns",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="external UTF-8 denylist (literal lines, or regex:... lines)",
    )
    parser.add_argument(
        "--require-external-patterns",
        action="store_true",
        help="fail unless at least one external denylist pattern was loaded",
    )
    parser.add_argument("--root", type=Path, help="repository root; defaults to the current work tree")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    patterns: list[PatternSpec] = list(_generic_patterns())
    try:
        root = args.root.resolve() if args.root else _git_root(Path.cwd())
        candidates: list[Path] = list(args.paths)
        if args.all_tracked:
            candidates.extend(_tracked_paths(root))
        if args.files_from_stdin:
            candidates.extend(_stdin_paths())
        if not candidates and not args.all_history:
            raise ValueError(
                "select files with paths, --all-tracked, --all-history, or --files-from-stdin"
            )

        external_paths = [path if path.is_absolute() else Path.cwd() / path for path in args.extra_patterns]
        external_patterns = _load_external_patterns(external_paths, dict(os.environ))
        if args.require_external_patterns and not external_patterns:
            raise ValueError("an external private-marker denylist is required")
        patterns.extend(external_patterns)
        paths = _resolve_candidates(root, candidates)
        findings, text_count, binary_count = scan_paths(root, paths, patterns)
        history_text_count = 0
        history_binary_count = 0
        if args.all_history:
            history_findings, history_text_count, history_binary_count = scan_history(root, patterns)
            findings = sorted(set(findings).union(history_findings))
    except ValueError as error:
        message = _redact_output(f"privacy scan configuration error: {error}", patterns)
        print(message, file=sys.stderr)
        return 2

    if findings:
        affected_files = len({(finding.object_id, finding.path) for finding in findings})
        header = f"privacy scan failed: {len(findings)} finding(s) in {affected_files} file(s)"
        print(_redact_output(header, patterns), file=sys.stderr)
        for finding in findings:
            object_label = f"object {finding.object_id} " if finding.object_id else ""
            detail = (
                f"- {object_label}{finding.path}:{finding.line} "
                f"[{finding.rule}; {finding.representation}]"
            )
            print(_redact_output(detail, patterns), file=sys.stderr)
        return 1

    message = f"privacy scan passed: {text_count} text file(s), {binary_count} binary file(s) skipped"
    if args.all_history:
        message += (
            f"; {history_text_count} unique historical text blob(s), "
            f"{history_binary_count} binary blob(s) skipped"
        )
    print(_redact_output(message, patterns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
