from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.mapper.inputs import MANIFEST_NAME, MapperInputs

if TYPE_CHECKING:
    from vidmap.datasets.base import PreparedTestset


def resolve_prepared_input(run: Path | None) -> tuple[Path, PreparedTestset] | None:
    """Resolve benchmark inputs from saved identity; local runs have no benchmark GT."""
    if run is None or not (run / "mapper_inputs" / MANIFEST_NAME).is_file():
        return None
    identity = MapperInputs.from_directory(run / "mapper_inputs").frontend_identity()
    if identity["dataset"] == "local":
        return None
    layout = get_dataset_layout(identity["dataset"])
    testsets = layout.read_testsets(identity["scene"], identity["mode"])
    if testsets is None or identity["testset_id"] not in testsets:
        return None
    selection = testsets[identity["testset_id"]]
    if selection.image_ids != tuple(identity["reference_image_ids"]):
        raise ValueError(f"Prepared testset differs from the saved run selection: {run}")
    root = layout.input_dir(identity["scene"], selection.recording, selection.variant)
    return (root, selection) if (root / "rec").is_dir() else None
