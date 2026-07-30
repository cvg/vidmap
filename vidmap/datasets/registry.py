"""Cached runtime specifications for supported benchmark datasets."""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING

from vidmap.datasets.names import get_dataset_definition

if TYPE_CHECKING:
    from vidmap.datasets.base import DatasetSpec


@cache
def get_dataset_spec(name: str) -> DatasetSpec:
    """Construct and cache one requested dataset specification."""
    definition = get_dataset_definition(name)

    from vidmap.datasets.base import DatasetCapabilities, DatasetPreparation, DatasetSpec, PreparedSceneParser
    from vidmap.datasets.layouts import get_dataset_layout

    layout = get_dataset_layout(name)
    preparer = None if definition.preparer_module is None else DatasetPreparation(definition.preparer_module)

    if name == "crocodl":
        from vidmap.datasets.crocodl import CroCoDLParser

        def missing_artifacts(layout, scenes):
            from vidmap.datasets.prepare.crocodl import official_scene_is_prepared

            return tuple(layout.scene_dir(scene) for scene in scenes if not official_scene_is_prepared(scene))

        return DatasetSpec(
            name=name,
            parser=CroCoDLParser,
            layout=layout,
            capabilities=DatasetCapabilities(
                allows_missing_testsets=True,
                ignores_unknown_testsets=True,
            ),
            batch_case_parser=CroCoDLParser.for_testsets,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )
    if name == "lamar":
        from vidmap.datasets.lamar import LaMARParser

        def missing_artifacts(layout, scenes):
            from vidmap.datasets.prepare.lamar import lamar_scene_is_prepared

            return tuple(
                layout.scene_dir(scene) for scene in scenes if not lamar_scene_is_prepared(layout.scene_dir(scene))
            )

        return DatasetSpec(
            name=name,
            parser=LaMARParser,
            layout=layout,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )

    if name == "euroc":

        def missing_artifacts(layout, scenes):
            from vidmap.datasets.prepare.euroc import euroc_scene_is_prepared

            return tuple(
                layout.scene_dir(scene) for scene in scenes if not euroc_scene_is_prepared(layout.scene_dir(scene))
            )

        return DatasetSpec(
            name=name,
            parser=partial(PreparedSceneParser, layout),
            layout=layout,
            preparer=preparer,
            missing_artifacts=missing_artifacts,
            default_mode=definition.default_mode,
            default_evaluation_metric=definition.default_evaluation_metric,
        )
    return DatasetSpec(
        name=name,
        parser=partial(PreparedSceneParser, layout),
        layout=layout,
        preparer=preparer,
        default_mode=definition.default_mode,
        default_evaluation_metric=definition.default_evaluation_metric,
    )
