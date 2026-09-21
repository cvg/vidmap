from copy import deepcopy

import pycolmap


def build_initial_reconstruction(
    scene_parser, reference_image_names: list[str] | None = None
) -> pycolmap.Reconstruction:
    """Copy selected images and their cameras from the prepared scene."""
    source = scene_parser.rec
    names = None if reference_image_names is None else set(reference_image_names)
    images = [image for image in source.images.values() if names is None or image.name in names]
    rec = pycolmap.Reconstruction()
    for camera_id in dict.fromkeys(image.camera_id for image in images):
        rec.add_camera_with_trivial_rig(deepcopy(source.cameras[camera_id]))
    for image in images:
        rec.add_image_with_trivial_frame(
            pycolmap.Image(image_id=image.image_id, name=image.name, camera_id=image.camera_id)
        )
    return rec
