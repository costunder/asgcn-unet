"""Manual browser QA with SYNTHETIC fixtures, never real evaluation results.

Run ``python -m tests.viewer_smoke`` and press Enter after browser inspection.
Only this test's loopback server and temporary fixture directory are closed.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import torch

from asgcn_unet.result_viewer import ResultViewer, ViewerHTTPServer
from tests.test_result_viewer import viewer_fixture


def main() -> None:
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="asgcn-viewer-smoke-") as temporary:
        root, config_path = viewer_fixture(Path(temporary))
        viewer = ResultViewer(root, configs={"hdr": config_path}, display_edges=12)
        viewer.warnings.insert(0, "SYNTHETIC CPU UI SMOKE TEST — NOT TRAINED MODEL RESULTS")
        with ViewerHTTPServer(viewer, port=0) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                print("SYNTHETIC CPU UI SMOKE TEST — NOT TRAINED MODEL RESULTS", flush=True)
                print(f"PID={os.getpid()} URL={server.url}", flush=True)
                input("Press Enter to finish browser QA and close only this test server: ")
            finally:
                server.shutdown()
                thread.join(timeout=5)
                viewer.close()


if __name__ == "__main__":
    main()
