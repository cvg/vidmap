from copy import deepcopy

import pycolmap


def build_initial_reconstruction(
    scene_parser, reference_image_names: list[str] | None = None
) -> pycolmap.Reconstruction:
    """Build the initial COLMAP reconstruction from a scene reconstruction."""
    if scene_parser.reconstruction_dir is not None:
        refrec = pycolmap.Reconstruction(scene_parser.reconstruction_dir)
    else:
        refrec = scene_parser.rec
    rec = pycolmap.Reconstruction()
    add_cameras = {camera.camera_id: camera for camera in refrec.cameras.values()}
    for camera in add_cameras.values():
        rec.add_camera_with_trivial_rig(deepcopy(camera))
    for imid, image in refrec.images.items():
        if reference_image_names is not None and image.name not in reference_image_names:
            continue
        image_ = pycolmap.Image(image_id=imid, name=image.name, camera_id=image.camera_id)
        rec.add_image_with_trivial_frame(image_)
    return rec
