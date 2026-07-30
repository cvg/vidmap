from __future__ import annotations

import os
import uuid
from pathlib import Path


def temporary_sibling(output: Path, suffix: str) -> Path:
    return output.parent / f".{output.name}.{uuid.uuid4().hex}{suffix}"


def finalize_output_file(temporary: Path, output: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(temporary, output)
        return
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise FileExistsError(f"Output already exists: {output} (pass overwrite=True to replace it)") from None
    except OSError as exc:
        raise RuntimeError(
            f"Could not finalize {temporary} as {output} without clobbering; " "retry with overwrite=True"
        ) from exc
    else:
        temporary.unlink()
