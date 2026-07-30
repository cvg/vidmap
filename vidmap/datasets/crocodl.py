"""CroCoDL dataset parser and session naming contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pycolmap

from vidmap.datasets.base import PreparedSceneParser, validate_path_component
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


def _session_from_testset(testset_desc: str) -> str:
    testset_desc = validate_path_component(testset_desc, "CroCoDL testset ID")
    session_id, separator, index = testset_desc.rpartition("-")
    if not separator or not session_id or not index.isdecimal():
        raise ValueError(f"CroCoDL testset ID {testset_desc!r} must end in a numeric subsequence index")
    return validate_path_component(session_id, "CroCoDL session ID")


class CroCoDLParser(PreparedSceneParser):
    """Read a CroCoDL session selected by a location-scoped testset key."""

    @classmethod
    def for_testsets(cls, scene: str, testset_descs: Iterable[str]) -> Iterator["CroCoDLParser"]:
        """Construct independent case parsers while loading each source session once."""
        reconstructions: dict[str, pycolmap.Reconstruction] = {}
        for testset_desc in testset_descs:
            session_id = _session_from_testset(testset_desc)
            if session_id in reconstructions:
                yield cls(scene, testset_id=session_id, _reconstruction=reconstructions[session_id])
                continue
            parser = cls(scene, testset_id=session_id)
            reconstructions[session_id] = parser.rec
            yield parser

    def __init__(
        self,
        scene: str,
        testset_id: str | None = None,
        *,
        _reconstruction: pycolmap.Reconstruction | None = None,
    ) -> None:
        self.layout = CROCODL_LAYOUT
        self.scene = scene
        location = parse_benchmark_scene(scene)
        self.session_id = testset_id

        if testset_id is None:
            self.scene_dir = self.layout.data_dir / scene
        else:
            testset_id = validate_ios_session(testset_id)
            self.scene_dir = self.layout.data_dir / location / testset_id
            if not self.scene_dir.is_dir() or not (self.scene_dir / "rec").is_dir():
                raise ValueError(
                    f"CroCoDL session {testset_id!r} not found in registered scene {scene!r}: {self.scene_dir}"
                )

        self.reconstruction_dir = self.scene_dir / "rec"
        self.images_dir = self.scene_dir / "images"
        self.rgb_dir = self.images_dir
        self.rec = _reconstruction if _reconstruction is not None else pycolmap.Reconstruction(self.reconstruction_dir)
