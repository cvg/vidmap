"""Storage policy for intermediate reconstruction checkpoints."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

INTERMEDIATE_RECONSTRUCTION_NAMES = (
    "rec-gp",
    "rec-ba",
    "rec-pre-point-refinement",
)


def remove_disabled_intermediate_reconstructions(
    output_dir: Path,
    *,
    persist: bool,
    post_point_refinement: bool,
) -> None:
    """Remove checkpoint directories that the current run will not produce."""

    disabled = (
        INTERMEDIATE_RECONSTRUCTION_NAMES
        if not persist
        else ("rec-pre-point-refinement",) if not post_point_refinement else ()
    )
    for name in disabled:
        path = Path(output_dir) / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


@contextmanager
def reconstruction_checkpoint_directory(
    output_dir: Path,
    name: str,
    *,
    persist: bool,
) -> Iterator[Path]:
    """Yield a persistent output directory or a transient internal checkpoint."""

    output_dir = Path(output_dir)
    if persist:
        checkpoint_dir = output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        yield checkpoint_dir
        return

    with TemporaryDirectory(prefix=f".{name}-", dir=output_dir) as temporary:
        yield Path(temporary)
