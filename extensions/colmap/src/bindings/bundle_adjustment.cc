#include "vidmap_native/bundle_adjustment.h"

#include "bindings.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindBundleAdjustment(py::module_& m) {
  py::class_<DepthConstraintRecord>(m, "DepthConstraintRecord")
      .def(py::init<>())
      .def_readwrite("image_id", &DepthConstraintRecord::image_id)
      .def_readwrite("point3D_id", &DepthConstraintRecord::point3D_id)
      .def_readwrite("depth", &DepthConstraintRecord::depth)
      .def_readwrite("loss", &DepthConstraintRecord::loss)
      .def("validate", &DepthConstraintRecord::Validate);

  py::class_<DepthScaleRecord>(m, "DepthScaleRecord")
      .def(py::init<>())
      .def_readwrite("image_id", &DepthScaleRecord::image_id)
      .def_readwrite("shift_scale", &DepthScaleRecord::shift_scale)
      .def_readwrite("fix_shift", &DepthScaleRecord::fix_shift)
      .def_readwrite("fix_scale", &DepthScaleRecord::fix_scale)
      .def_readwrite("use_scale_prior", &DepthScaleRecord::use_scale_prior)
      .def_readwrite("scale_prior_stddev",
                     &DepthScaleRecord::scale_prior_stddev)
      .def_readwrite("scale_prior_loss", &DepthScaleRecord::scale_prior_loss)
      .def("validate", &DepthScaleRecord::Validate);

  py::class_<LogFocalPriorRecord>(m, "LogFocalPriorRecord")
      .def(py::init<>())
      .def_readwrite("camera_id", &LogFocalPriorRecord::camera_id)
      .def_readwrite("observations", &LogFocalPriorRecord::observations)
      .def_readwrite("loss", &LogFocalPriorRecord::loss)
      .def("validate", &LogFocalPriorRecord::Validate);

  py::class_<BundleAdjustmentOptions>(m, "BundleAdjustmentOptions")
      .def(py::init<>())
      .def_readwrite("image_order", &BundleAdjustmentOptions::image_order)
      .def_readwrite("constant_camera_ids",
                     &BundleAdjustmentOptions::constant_camera_ids)
      .def_readwrite("variable_point3D_ids",
                     &BundleAdjustmentOptions::variable_point3D_ids)
      .def_readwrite("constant_point3D_ids",
                     &BundleAdjustmentOptions::constant_point3D_ids)
      .def_readwrite("reprojection_loss",
                     &BundleAdjustmentOptions::reprojection_loss)
      .def_readwrite("refine_focal_length",
                     &BundleAdjustmentOptions::refine_focal_length)
      .def_readwrite("refine_principal_point",
                     &BundleAdjustmentOptions::refine_principal_point)
      .def_readwrite("refine_extra_params",
                     &BundleAdjustmentOptions::refine_extra_params)
      .def_readwrite("refine_points3D",
                     &BundleAdjustmentOptions::refine_points3D)
      .def_readwrite("min_track_length",
                     &BundleAdjustmentOptions::min_track_length)
      .def_readwrite("fix_first_pose", &BundleAdjustmentOptions::fix_first_pose)
      .def_readwrite("fix_rotations", &BundleAdjustmentOptions::fix_rotations)
      .def_readwrite("fix_all_poses", &BundleAdjustmentOptions::fix_all_poses)
      .def_readwrite("use_log_depth_residual",
                     &BundleAdjustmentOptions::use_log_depth_residual)
      .def_readwrite("num_threads", &BundleAdjustmentOptions::num_threads)
      .def_readwrite("max_num_iterations",
                     &BundleAdjustmentOptions::max_num_iterations)
      .def_readwrite("function_tolerance",
                     &BundleAdjustmentOptions::function_tolerance)
      .def_readwrite("gradient_tolerance",
                     &BundleAdjustmentOptions::gradient_tolerance)
      .def_readwrite("parameter_tolerance",
                     &BundleAdjustmentOptions::parameter_tolerance)
      .def_readwrite("solver_backend", &BundleAdjustmentOptions::solver_backend)
      .def_readwrite("playback", &BundleAdjustmentOptions::playback)
      .def_readwrite("use_imu", &BundleAdjustmentOptions::use_imu)
      .def_readwrite("use_analytical_imu_cost",
                     &BundleAdjustmentOptions::use_analytical_imu_cost)
      .def_readwrite("refine_imu_scale",
                     &BundleAdjustmentOptions::refine_imu_scale)
      .def_readwrite("refine_gravity", &BundleAdjustmentOptions::refine_gravity)
      .def_readwrite("refine_imu_velocities",
                     &BundleAdjustmentOptions::refine_imu_velocities)
      .def_readwrite("refine_gyro_bias",
                     &BundleAdjustmentOptions::refine_gyro_bias)
      .def_readwrite("refine_accel_bias",
                     &BundleAdjustmentOptions::refine_accel_bias)
      .def_readwrite("refine_imu_from_cam_rotation",
                     &BundleAdjustmentOptions::refine_imu_from_cam_rotation)
      .def_readwrite("refine_imu_from_cam_translation",
                     &BundleAdjustmentOptions::refine_imu_from_cam_translation)
      .def_readwrite("auto_initialize_gravity",
                     &BundleAdjustmentOptions::auto_initialize_gravity)
      .def_readwrite("auto_initialize_imu_states",
                     &BundleAdjustmentOptions::auto_initialize_imu_states)
      .def_readwrite("imu_warm_start", &BundleAdjustmentOptions::imu_warm_start)
      .def_readwrite("apply_imu_alignment_to_problem",
                     &BundleAdjustmentOptions::apply_imu_alignment_to_problem)
      .def_readwrite("initial_log_scale",
                     &BundleAdjustmentOptions::initial_log_scale)
      .def_readwrite("initial_gravity_direction",
                     &BundleAdjustmentOptions::initial_gravity_direction)
      .def_readwrite("imu_from_cam", &BundleAdjustmentOptions::imu_from_cam)
      .def_readwrite("use_imu_from_cam_prior",
                     &BundleAdjustmentOptions::use_imu_from_cam_prior)
      .def_readwrite(
          "imu_from_cam_rotation_prior_stddev_deg",
          &BundleAdjustmentOptions::imu_from_cam_rotation_prior_stddev_deg)
      .def_readwrite(
          "imu_from_cam_translation_prior_stddev",
          &BundleAdjustmentOptions::imu_from_cam_translation_prior_stddev)
      .def_readwrite("reintegrate_angle_norm_thres",
                     &BundleAdjustmentOptions::reintegrate_angle_norm_thres)
      .def_readwrite("reintegrate_vel_norm_thres",
                     &BundleAdjustmentOptions::reintegrate_vel_norm_thres)
      .def("validate", &BundleAdjustmentOptions::Validate);

  py::class_<BundleAdjustmentDiagnostics>(m, "BundleAdjustmentDiagnostics")
      .def_readonly("num_reprojection_residuals",
                    &BundleAdjustmentDiagnostics::num_reprojection_residuals)
      .def_readonly("num_depth_residuals",
                    &BundleAdjustmentDiagnostics::num_depth_residuals)
      .def_readonly(
          "num_intrinsics_prior_residuals",
          &BundleAdjustmentDiagnostics::num_intrinsics_prior_residuals)
      .def_readonly("num_scale_prior_residuals",
                    &BundleAdjustmentDiagnostics::num_scale_prior_residuals)
      .def_readonly("num_imu_residuals",
                    &BundleAdjustmentDiagnostics::num_imu_residuals)
      .def_readonly(
          "num_imu_extrinsics_prior_residuals",
          &BundleAdjustmentDiagnostics::num_imu_extrinsics_prior_residuals)
      .def_readonly("num_residual_blocks",
                    &BundleAdjustmentDiagnostics::num_residual_blocks)
      .def_readonly("num_parameter_blocks",
                    &BundleAdjustmentDiagnostics::num_parameter_blocks)
      .def_readonly("num_parameters",
                    &BundleAdjustmentDiagnostics::num_parameters)
      .def_readonly("num_iterations",
                    &BundleAdjustmentDiagnostics::num_iterations)
      .def_readonly("termination_type",
                    &BundleAdjustmentDiagnostics::termination_type)
      .def_readonly("initial_cost", &BundleAdjustmentDiagnostics::initial_cost)
      .def_readonly("final_cost", &BundleAdjustmentDiagnostics::final_cost);

  py::class_<BundleAdjustmentResult>(m, "BundleAdjustmentResult")
      .def_readonly("success", &BundleAdjustmentResult::success)
      .def_readonly("depth_shift_scales",
                    &BundleAdjustmentResult::depth_shift_scales)
      .def_readonly("log_scale", &BundleAdjustmentResult::log_scale)
      .def_readonly("scale", &BundleAdjustmentResult::scale)
      .def_readonly("gravity_direction",
                    &BundleAdjustmentResult::gravity_direction)
      .def_readonly("gravity_in_world",
                    &BundleAdjustmentResult::gravity_in_world)
      .def_readonly("imu_from_cam", &BundleAdjustmentResult::imu_from_cam)
      .def_readonly("imu_states", &BundleAdjustmentResult::imu_states)
      .def_readonly("diagnostics", &BundleAdjustmentResult::diagnostics);

  m.def(
      "run_bundle_adjustment",
      [](const BundleAdjustmentOptions& options,
         const std::vector<DepthConstraintRecord>& depth_constraints,
         const std::vector<DepthScaleRecord>& depth_scales,
         const std::vector<LogFocalPriorRecord>& intrinsics_priors,
         MappingProblem* problem,
         const std::vector<ImuEdgeRecord>& imu_edges,
         const std::vector<ImuStateRecord>& imu_states) {
        py::gil_scoped_release release;
        return RunBundleAdjustment(options,
                                   depth_constraints,
                                   depth_scales,
                                   intrinsics_priors,
                                   problem,
                                   imu_edges,
                                   imu_states);
      },
      py::arg("options"),
      py::arg("depth_constraints"),
      py::arg("depth_scales"),
      py::arg("intrinsics_priors"),
      py::arg("problem"),
      py::arg("imu_edges") = std::vector<ImuEdgeRecord>{},
      py::arg("imu_states") = std::vector<ImuStateRecord>{});
}

}  // namespace vidmap
