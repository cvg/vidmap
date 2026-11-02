"""CroCoDL dataset parser and session naming contract."""

from __future__ import annotations

from vidmap.datasets.base import validate_path_component
from vidmap.datasets.layouts import get_dataset_layout

CROCODL_LAYOUT = get_dataset_layout("crocodl")


def validate_ios_session(session_id: str) -> str:
    session_id = validate_path_component(session_id, "CroCoDL session ID")
    if not session_id.startswith("ios_"):
        raise ValueError(f"CroCoDL session {session_id!r} is not an iOS session")
    return session_id


def benchmark_scene_name(location: str) -> str | None:
    location = validate_path_component(location, "CroCoDL location")
    scene = f"ios-{location}"
    return scene if scene in CROCODL_LAYOUT.scenes else None


def parse_benchmark_scene(scene: str) -> str:
    scene = validate_path_component(scene, "CroCoDL scene")
    if scene not in CROCODL_LAYOUT.scenes:
        raise ValueError(f"CroCoDL scene {scene!r} is not registered")
    modality, separator, location = scene.partition("-")
    if not separator or modality != "ios":
        raise ValueError(f"CroCoDL scene {scene!r} is not an iOS scene")
    location = validate_path_component(location, "CroCoDL location")
    return location
