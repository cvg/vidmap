"""Explicit ETH3D-SLAM monocular benchmark inventory with public ground truth."""

from vidmap.datasets.video import VideoDatasetManifest, VideoSequence

ETH3D_SLAM_SCENES = (
    "cables_1",
    "cables_2",
    "cables_3",
    "camera_shake_1",
    "camera_shake_2",
    "camera_shake_3",
    "ceiling_1",
    "ceiling_2",
    "desk_3",
    "desk_changing_1",
    "einstein_1",
    "einstein_2",
    "einstein_flashlight",
    "einstein_global_light_changes_1",
    "einstein_global_light_changes_2",
    "einstein_global_light_changes_3",
    "kidnap_1",
    "sfm_lab_room_1",
    "sfm_lab_room_2",
    "large_loop_1",
    "mannequin_1",
    "mannequin_3",
    "mannequin_4",
    "mannequin_5",
    "mannequin_7",
    "mannequin_face_1",
    "mannequin_face_2",
    "mannequin_face_3",
    "mannequin_head",
    "motion_1",
    "planar_2",
    "planar_3",
    "plant_1",
    "plant_2",
    "plant_3",
    "plant_4",
    "plant_5",
    "plant_scene_1",
    "plant_scene_2",
    "plant_scene_3",
    "reflective_1",
    "repetitive",
    "sfm_bench",
    "sfm_garden",
    "sfm_house_loop",
    "sofa_1",
    "sofa_2",
    "sofa_3",
    "sofa_4",
    "sofa_shake",
    "table_3",
    "table_4",
    "table_7",
    "vicon_light_1",
    "vicon_light_2",
)

# Explicitly excluded by project policy. Keeping this list next to the runnable
# inventory prevents an official dark variant from being silently forgotten.
ETH3D_SLAM_EXCLUDED_DARK_SCENES = (
    "einstein_dark",
    "kidnap_dark",
    "plant_dark",
    "sofa_dark_1",
    "sofa_dark_2",
    "sofa_dark_3",
    "boxes_dark",
    "desk_dark_1",
    "desk_dark_2",
)

ETH3D_SLAM_MANIFEST = VideoDatasetManifest(
    "eth3d_slam",
    tuple(
        VideoSequence(
            name=scene,
            split="training",
            gt_quality="reference",
            condition_tags=("monocular", "non-dark", "ground-truth"),
        )
        for scene in ETH3D_SLAM_SCENES
    ),
)

if len(ETH3D_SLAM_SCENES) != 55:
    raise ValueError("ETH3D-SLAM benchmark inventory must contain 55 ground-truth scenes")
