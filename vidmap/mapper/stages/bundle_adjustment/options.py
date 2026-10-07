"""Configure upstream bundle adjustment for one VidMap solve."""

import pyceres
import pycolmap

from vidmap.mapper.native.solver_backend import apply_solver_backend
from vidmap.mapper.options.positioning import LossConfig


def build_bundle_adjustment_options(
    *,
    reconstruction,
    image_order,
    camera_ids,
    variable_point3D_ids,
    optimize_intrinsics,
    refine_principal_point,
    fix_rotations,
    fix_first_pose=True,
    fix_all_poses=False,
    reprojection_loss,
    reprojection_scale,
    reprojection_weight,
    num_threads,
    solver_backend,
):
    config = pycolmap.BundleAdjustmentConfig()
    for image_id in image_order:
        config.add_image(image_id)
    if not optimize_intrinsics:
        for camera_id in camera_ids:
            config.set_constant_cam_intrinsics(camera_id)
    for point_id in variable_point3D_ids:
        config.add_variable_point(point_id)
    if image_order and fix_first_pose:
        config.set_constant_rig_from_world_pose(reconstruction.images[image_order[0]].frame_id)
    options = pycolmap.BundleAdjustmentOptions(
        refine_focal_length=True,
        refine_principal_point=refine_principal_point,
        refine_extra_params=True,
        refine_points3D=True,
        min_track_length=0,
        refine_rig_from_world=not fix_all_poses,
        constant_rig_from_world_rotation=fix_rotations,
        refine_sensor_from_rig=False,
        print_summary=False,
    )
    loss = LossConfig(name=reprojection_loss, scale=reprojection_scale, weight=reprojection_weight)
    options.ceres.loss_function_type = getattr(pycolmap.LossFunctionType, loss.name.upper())
    options.ceres.loss_function_scale = loss.scale
    options.ceres.loss_function_weight = loss.weight
    options.ceres.use_gpu = solver_backend.use_cuda
    options.ceres.auto_select_solver_type = False
    options.ceres.min_num_residuals_for_cpu_multi_threading = 0
    options.ceres.solver_options = pyceres.SolverOptions()
    solver = options.ceres.solver_options
    solver.num_threads = -1 if num_threads is None else num_threads
    if solver.num_threads == 0:
        raise ValueError("num_threads must be nonzero")
    solver.max_num_iterations = 50
    solver.function_tolerance = 1e-6
    solver.gradient_tolerance = 1e-10
    solver.parameter_tolerance = 1e-8
    apply_solver_backend(solver, solver_backend)
    return options, config
