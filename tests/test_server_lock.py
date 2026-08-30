from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _server_lock() -> tuple[dict[str, str], dict[str, set[str]]]:
    text = (ROOT / "constraints/server.txt").read_text(encoding="utf-8")
    assert "--index-url https://pypi.org/simple" in text
    assert "--extra-index-url https://download.pytorch.org/whl/cu126" in text
    pins: dict[str, str] = {}
    hashes: dict[str, set[str]] = {}
    for line in text.replace("\\\n", " ").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--index-url ", "--extra-index-url ")):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([^\s]+)((?:\s+--hash=sha256:[a-f0-9]{64})+)", line
        )
        assert match is not None, "Every server dependency needs an exact version and hashes"
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        assert name not in pins
        pins[name] = match[2]
        hashes[name] = set(re.findall(r"--hash=sha256:([a-f0-9]{64})", match[3]))
    return pins, hashes


def test_server_hash_lock_and_runtime_profile_have_identical_package_versions() -> None:
    pins, hashes = _server_lock()
    profile = json.loads((ROOT / "constraints/server.json").read_text(encoding="utf-8"))
    assert pins == profile["packages"]
    assert pins["torch"] == profile["torch"] == "2.13.0+cu126"
    assert profile["python"] == "3.12.14"
    assert profile["cuda"] == "12.6"
    assert profile["platform"] == "Linux"
    assert profile["machine"] == "x86_64"
    assert profile["environment"] == "conda"
    assert {
        "pip",
        "setuptools",
        "wheel",
        "triton",
        "cuda-toolkit",
        "cuda-bindings",
        "nvidia-cudnn-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
    }.issubset(pins)
    # Official index hash for this profile's CPython 3.12 Linux x86_64 CUDA wheel.
    assert "8695f3c6b7966d44560275b90c5c28e5091ba33ddbb1ab33b2173782ca1e9145" in hashes["torch"]


def test_server_lock_preserves_core_profile_and_bootstrap_input_pins() -> None:
    pins, _ = _server_lock()
    core = (ROOT / "constraints/py312.txt").read_text(encoding="utf-8")
    for line in core.splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        assert pins[canonical].split("+", 1)[0] == version
    source = (ROOT / "constraints/server.in").read_text(encoding="utf-8")
    assert "-r py312.txt" in source
    for name in ("pip", "setuptools", "wheel", "torch"):
        assert f"{name}=={pins[name]}" in source
