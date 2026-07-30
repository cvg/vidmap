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

  py::class_<IntrinsicsPriorRecord>(m, "IntrinsicsPriorRecord")
      .def(py::init<>())
      .def_readwrite("camera_id", &IntrinsicsPriorRecord::camera_id)
      .def_readwrite("values", &IntrinsicsPriorRecord::values)
      .def_readwrite("stddevs", &IntrinsicsPriorRecord::stddevs)
      .def("validate", &IntrinsicsPriorRecord::Validate);

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
      .def_readwrite("fix_all_poses",
                     &BundleAdjustmentOptions::fix_all_poses)
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
      .def_readwrite("playback", &BundleAdjustmentOptions::playback)
      .def("validate", &BundleAdjustmentOptions::Validate);

  py::class_<BundleAdjustmentDiagnostics>(m, "BundleAdjustmentDiagnostics")
      .def_readonly("num_residual_blocks",
                    &BundleAdjustmentDiagnostics::num_residual_blocks)
      .def_readonly("initial_cost", &BundleAdjustmentDiagnostics::initial_cost)
      .def_readonly("final_cost", &BundleAdjustmentDiagnostics::final_cost);

  py::class_<BundleAdjustmentResult>(m, "BundleAdjustmentResult")
      .def_readonly("success", &BundleAdjustmentResult::success)
      .def_readonly("depth_shift_scales",
                    &BundleAdjustmentResult::depth_shift_scales)
      .def_readonly("diagnostics", &BundleAdjustmentResult::diagnostics);

  m.def(
      "run_bundle_adjustment",
      [](const BundleAdjustmentOptions& options,
         const std::vector<DepthConstraintRecord>& depth_constraints,
         const std::vector<DepthScaleRecord>& depth_scales,
         const std::vector<IntrinsicsPriorRecord>& intrinsics_priors,
         MappingProblem* problem) {
        py::gil_scoped_release release;
        return RunBundleAdjustment(options,
                                   depth_constraints,
                                   depth_scales,
                                   intrinsics_priors,
                                   problem);
      },
      py::arg("options"),
      py::arg("depth_constraints"),
      py::arg("depth_scales"),
      py::arg("intrinsics_priors"),
      py::arg("problem"));
}

}  // namespace vidmap
