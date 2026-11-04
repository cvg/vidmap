"""Cached runtime specifications for supported benchmark datasets."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from vidmap.datasets.names import get_dataset_definition

if TYPE_CHECKING:
    from vidmap.datasets.base import DatasetSpec


@cache
def get_dataset_spec(name: str) -> DatasetSpec:
    """Construct and cache one requested dataset specification."""
    definition = get_dataset_definition(name)

    from vidmap.datasets.base import DatasetCapabilities, DatasetPreparation, DatasetSpec
    from vidmap.datasets.layouts import get_dataset_layout

    layout = get_dataset_layout(name)
    preparer = None if definition.preparer_module is None else DatasetPreparation(definition.preparer_module)

    if name in {"spires", "hilti"}:
        from vidmap.datasets.prepare.helpers.ros_benchmarks import scene_is_prepared

        preparer = DatasetPreparation(definition.preparer_module, uses_mode=True)

        def missing_artifacts(layout, scenes, mode):
            return tuple(
                layout.scene_dir(scene) for scene in scenes if not scene_is_prepared(layout, scene, mode=mode)
            )

        return DatasetSpec(
            name=name,
            layout=layout,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )
    if name == "crocodl":

        def missing_artifacts(layout, scenes, mode):
            from vidmap.datasets.prepare.crocodl import official_scene_is_prepared

            return tuple(layout.scene_dir(scene) for scene in scenes if not official_scene_is_prepared(scene))

        return DatasetSpec(
            name=name,
            layout=layout,
            capabilities=DatasetCapabilities(
                allows_missing_testsets=True,
                ignores_unknown_testsets=True,
            ),
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )
    if name == "lamar":
        preparer = DatasetPreparation(definition.preparer_module, (("sample", "sample-300"), ("complete", "complete")))

        def missing_artifacts(layout, scenes, mode):
            from vidmap.datasets.prepare.lamar import lamar_scene_is_prepared

            profile = preparer.profile(mode)
            return tuple(
                layout.scene_dir(scene)
                for scene in scenes
                if not lamar_scene_is_prepared(layout.data_dir, scene, profile=profile)
            )

        return DatasetSpec(
            name=name,
            layout=layout,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )

    if name == "eth3d_slam":
        from vidmap.datasets.prepare.eth3d_slam import _scene_is_complete

        def missing_artifacts(layout, scenes, mode):
            return tuple(
                layout.scene_dir(scene) for scene in scenes if not _scene_is_complete(layout, scene, mode=mode)
            )

        return DatasetSpec(
            name=name,
            layout=layout,
            preparer=DatasetPreparation(definition.preparer_module, uses_mode=True),
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )

    if name == "euroc":
        preparer = DatasetPreparation(definition.preparer_module, uses_mode=True)

        def missing_artifacts(layout, scenes, mode):
            from vidmap.datasets.prepare.euroc import euroc_scene_is_prepared

            return tuple(
                layout.scene_dir(scene)
                for scene in scenes
                if not euroc_scene_is_prepared(layout.scene_dir(scene), mode=mode)
            )

        return DatasetSpec(
            name=name,
            layout=layout,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )
    return DatasetSpec(
        name=name,
        layout=layout,
        preparer=preparer,
        default_mode=definition.default_mode,
        default_evaluation_metric=definition.default_evaluation_metric,
    )
