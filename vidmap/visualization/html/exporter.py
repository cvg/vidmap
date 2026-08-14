"""Write the interactive browser reconstruction viewer."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from . import scene

logger = logging.getLogger(__name__)


def write_html(output: str | Path, html: str) -> Path:
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(html)
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Visualization saved to %s", output)
    return output


def write_viewer_html(output: str | Path) -> Path:
    """Atomically write the reusable browser viewer."""
    return write_html(output, scene.render_viewer_html())
