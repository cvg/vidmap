"""LaMAR dataset parser."""

from vidmap.datasets.base import PreparedSceneParser
from vidmap.datasets.layouts import get_dataset_layout

LAMAR_LAYOUT = get_dataset_layout("lamar")


class LaMARParser(PreparedSceneParser):
    """Read a prepared LaMAR ground-truth reconstruction."""

    def __init__(self, scene: str) -> None:
        super().__init__(LAMAR_LAYOUT, scene)
