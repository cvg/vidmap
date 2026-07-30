"""Stable identity for a complete frontend-to-mapper boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vidmap.configuration.dump import config_to_dict
from vidmap.configuration.names import config_name_to_output_slug
from vidmap.frontend.cache import fingerprint

FRONTEND_IDENTITY_SCHEMA_VERSION = 2


def semantic_frontend_config(conf) -> dict[str, object]:
    """Serialize the current semantic frontend boundary contract."""

    return {
        "schema_version": FRONTEND_IDENTITY_SCHEMA_VERSION,
        "colmap_runtime": conf.colmap_runtime,
        "pipeline": config_to_dict(conf.pipeline),
        "deterministic_frontend": conf.run.deterministic_frontend,
    }


def semantic_frontend_fingerprint(conf) -> str:
    return fingerprint(semantic_frontend_config(conf))


def frontend_config_identity(conf) -> dict[str, object]:
    """Return the complete config-owned identity required by mapping."""

    if conf.name is None:
        raise ValueError("FrontendConfig requires its canonical root-relative name")
    return {
        "tag": config_name_to_output_slug(conf.name),
        "config_name": conf.name,
        "config_fingerprint": semantic_frontend_fingerprint(conf),
        "colmap_runtime": conf.colmap_runtime,
        "boundary_options": {
            "use_geocalib": conf.pipeline.use_geocalib,
            "view_graph_calibration": conf.pipeline.view_graph_calibration,
            "vgc_expand": False,
        },
    }


@dataclass(frozen=True)
class FrontendIdentity:
    """Semantic and target identity required to reuse mapper inputs."""

    tag: str
    config_name: str
    config_fingerprint: str
    colmap_runtime: str
    dataset: str
    scene: str
    mode: str
    testset_id: str
    reference_image_ids: tuple[int, ...]
    use_geocalib: bool
    view_graph_calibration: bool

    @classmethod
    def from_config(
        cls,
        conf,
        *,
        dataset: str,
        scene: str,
        mode: str,
        testset_id: str,
        reference_image_ids: Sequence[int],
    ) -> "FrontendIdentity":
        if conf.name is None:
            raise ValueError("FrontendConfig requires its canonical root-relative name")
        tag = config_name_to_output_slug(conf.name)
        return cls(
            tag=tag,
            config_name=conf.name,
            config_fingerprint=semantic_frontend_fingerprint(conf),
            colmap_runtime=conf.colmap_runtime,
            dataset=dataset,
            scene=scene,
            mode=mode,
            testset_id=testset_id,
            reference_image_ids=tuple(reference_image_ids),
            use_geocalib=conf.pipeline.use_geocalib,
            view_graph_calibration=conf.pipeline.view_graph_calibration,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": FRONTEND_IDENTITY_SCHEMA_VERSION,
            "tag": self.tag,
            "config_name": self.config_name,
            "config_fingerprint": self.config_fingerprint,
            "colmap_runtime": self.colmap_runtime,
            "dataset": self.dataset,
            "scene": self.scene,
            "mode": self.mode,
            "testset_id": self.testset_id,
            "reference_image_ids": list(self.reference_image_ids),
            "boundary_options": {
                "use_geocalib": self.use_geocalib,
                "view_graph_calibration": self.view_graph_calibration,
                "vgc_expand": False,
            },
        }
