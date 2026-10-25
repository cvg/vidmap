"""Frontend selection and pair policy over authoritative COLMAP membership."""

from vidmap.datasets.capture import PreparedCapture
from vidmap.frontend.options.rig import RigOptions


def resolve_rig_options(capture: PreparedCapture, options: RigOptions):
    """Resolve the admission sensor and ordered same-frame matching policy."""
    reference = capture.reference_camera_id if options.keyframe_camera is None else options.keyframe_camera
    if reference not in capture.rec.cameras:
        raise ValueError("The keyframing camera must belong to the capture")
    if any(cid not in capture.rec.cameras for pair in options.match_same_timestamp for cid in pair):
        raise ValueError("Same-timestamp matching references an unknown camera")
    for first, second in options.match_same_timestamp:
        if not any(
            rig.has_sensor(capture.rec.camera(first).sensor_id)
            and rig.has_sensor(capture.rec.camera(second).sensor_id)
            for rig in capture.rec.rigs.values()
        ):
            raise ValueError("Same-timestamp camera pairs must belong to the same rig")
    sequential = {frozenset(pair) for pair in options.flag_same_timestamp_as_sequential}
    return reference, tuple(
        (first, second, frozenset((first, second)) in sequential)
        for first, second in sorted(tuple(sorted(pair)) for pair in options.match_same_timestamp)
    )


def camera_streams(capture: PreparedCapture, names=None, reference_camera_id=None):
    """Derive temporal streams from native frames, without copying their graph."""
    selected = None if names is None else set(names)
    if selected is not None and (
        len(selected) != len(names) or not selected <= {im.name for im in capture.rec.images.values()}
    ):
        raise ValueError("Selected capture images must be unique and belong to the capture")
    if names is not None and reference_camera_id is None:
        cameras = tuple(dict.fromkeys(capture.image(name).camera_id for name in names))
    else:
        reference = capture.reference_camera_id if reference_camera_id is None else reference_camera_id
        if reference not in capture.rec.cameras:
            raise ValueError("The reference camera must belong to the capture")
        cameras = (reference,)
    cameras = (*cameras, *sorted(set(capture.rec.cameras) - set(cameras)))
    camera_order = {cid: index for index, cid in enumerate(cameras)}
    streams = []
    for fid in capture.frame_ids:
        frame = capture.rec.frame(fid)
        members = [capture.rec.image(data.id) for data in frame.image_ids]
        members.sort(key=lambda image: camera_order[image.camera_id])
        if not streams:
            streams = [[] for _ in members]
        for stream, image in zip(streams, members, strict=True):
            if selected is None or image.name in selected:
                stream.append(image.name)
    return tuple(tuple(stream) for stream in streams if stream)
