"""Synthetic, CPU-only offline browser smoke fixture. Never model results."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import tempfile
import zlib
from pathlib import Path

from asgcn_unet.offline_viewer import empty_payload, render_payload_html


def png(shade: int) -> bytes:
    def chunk(kind: bytes, value: bytes) -> bytes:
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value))
    # 64x48 RGB ramp is only a visible browser test pattern.
    pixels = b"".join(b"\0" + bytes(v for x in range(64) for v in
                                    (min(255, shade + x * 2), min(255, shade + y * 3), shade))
                      for y in range(48))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 48, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix="offline-smoke-", dir=project / "build"))
    payload = empty_payload("SYNTHETIC UI SMOKE · 모델 결과 아님")
    payload["notes"].insert(0, "SYNTHETIC DEBUG FIXTURE ONLY. 실제 학습/평가 결과가 아닙니다.")
    ids = []
    for name, shade in (("gt", 30), ("ann", 45), ("snn", 60)):
        data = png(shade)
        (root / f"{name}.png").write_bytes(data)
        key = hashlib.sha256(data).hexdigest()
        payload["images"][key] = {"mime": "image/png", "data_url": "data:image/png;base64,"
                                 + base64.b64encode(data).decode(), "sha256": key,
                                 "width": 64, "height": 48}
        ids.append(key)
    graph = {"nodes": [[0.1, 0.2, 0.1, 1], [0.7, 0.8, 0.9, -1], [0.5, 0.3, 0.7, 1]],
             "edges": [[0, 1], [1, 0], [0, 2]], "statistics": {"nodes": 3,
             "actual_directed_edges": 6, "displayed_edges": 3, "isolated_nodes": 0},
             "radius": 0.8, "position_dims": 3, "provenance_note": "SYNTHETIC UI TEST ONLY"}
    (root / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (root / "invalid.png").write_bytes(b"not PNG")
    attack = '</script><img src=x onerror="window.INJECTED=1">__OFFLINE_JS__'
    for dataset in ("hdr", "aid"):
        modes = [{"id": "ann", "label": "ann", "quality": {"frames": 2,
                  "micro": {"psnr": 10.125, "ssim": 0.5}, "macro": {"psnr": 11.5, "ssim": 0.6}},
                  "report_eligible": False, "benchmark": None},
                 {"id": "snn_standard_if_T4", "label": "snn_standard_if_T4",
                  "quality": {"frames": 2, "micro": {"psnr": 10.25, "ssim": 0.55},
                              "macro": {"psnr": 11.75, "ssim": 0.65}},
                  "report_eligible": False, "benchmark": None}]
        frames = [{"index": i, "sample_id": attack if i == 0 else "synthetic/1",
                   "group": "SYNTHETIC ONLY", "images": [
                       {"mode": m["id"], "target": ids[0], "prediction": ids[j + 1],
                        "metrics": {"psnr": 10.125 + j, "ssim": 0.5}, "report_eligible": False}
                       for j, m in enumerate(modes)], "graph": graph if i == 0 else None}
                  for i in range(2)]
        payload["datasets"].append({"id": dataset, "label": dataset + " · SYNTHETIC",
                                    "total_frames": 2, "modes": modes, "frames": frames})
    render_payload_html(payload, root / "smoke.html")
    print(root)


if __name__ == "__main__":
    main()
