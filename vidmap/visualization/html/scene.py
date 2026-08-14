"""Assemble the browser reconstruction viewer."""

from __future__ import annotations

import json
from pathlib import Path

THREE_JS_URL = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT_CONTROLS_URL = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
WIDE_LINE_URLS = (
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegmentsGeometry.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineGeometry.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineMaterial.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/LineSegments2.js",
    "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/lines/Line2.js",
)
SQL_JS_URL = "https://cdn.jsdelivr.net/npm/sql.js@1.13.0/dist/sql-wasm.js"
SQL_JS_WASM_URL = "https://cdn.jsdelivr.net/npm/sql.js@1.13.0/dist/sql-wasm.wasm"
LOOP_CLOSURE_MIN_SHARED_POINTS = 50
_BROWSER_DIRECTORY = Path(__file__).with_name("browser")
_HTML_TEMPLATE = (_BROWSER_DIRECTORY / "vidmap-viewer.html.in").read_text(encoding="utf-8")
_VIEWER_SCRIPT = (_BROWSER_DIRECTORY / "browser_scene.js").read_text(encoding="utf-8").rstrip("\n")


def render_viewer_html(*, embedded_run: dict[str, object] | None = None) -> str:
    """Return the browser viewer, optionally with one run embedded."""
    embedded = json.dumps(embedded_run, separators=(",", ":")).replace("<", "\\u003c")
    reconstruction_script = f"const VIDMAP_SQLITE_WASM_URL = {json.dumps(SQL_JS_WASM_URL)};\n" + (
        _BROWSER_DIRECTORY / "browser_reconstruction.js"
    ).read_text(encoding="utf-8")
    worker = json.dumps((_BROWSER_DIRECTORY / "browser_worker.js").read_text(encoding="utf-8"))
    return (
        _HTML_TEMPLATE.replace("__VIEWER_SCRIPT__", _VIEWER_SCRIPT)
        .replace("__THREE_JS_URL__", THREE_JS_URL)
        .replace("__ORBIT_CONTROLS_URL__", ORBIT_CONTROLS_URL)
        .replace(
            "__WIDE_LINE_SCRIPTS__",
            "\n".join(f'  <script src="{url}"></script>' for url in WIDE_LINE_URLS),
        )
        .replace("__SQLITE_SCRIPT__", f'  <script src="{SQL_JS_URL}"></script>')
        .replace("__RECONSTRUCTION_SCRIPT__", reconstruction_script)
        .replace("__BROWSER_WORKER_SOURCE__", worker)
        .replace("__LOOP_CLOSURE_MIN_SHARED_POINTS__", str(LOOP_CLOSURE_MIN_SHARED_POINTS))
        .replace("__VIDMAP_EMBEDDED_RUN__", embedded)
    )
