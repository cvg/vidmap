#include "vidmap_native/video_rotation_averaging.h"

#include "bindings.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindVideoRotationAveraging(py::module_& m) {
  py::class_<VideoRotationAveragingOptions>(m, "RotationAveragingOptions")
      .def(py::init<>())
      .def_readwrite("random_seed", &VideoRotationAveragingOptions::random_seed)
      .def_readwrite("image_order_passes",
                     &VideoRotationAveragingOptions::image_order_passes)
      .def_readwrite("filter_unregistered_images",
                     &VideoRotationAveragingOptions::filter_unregistered)
      .def_readwrite("skip_risky_loop_closure_pairs",
                     &VideoRotationAveragingOptions::skip_risky_lc_pairs)
      .def_readwrite("max_rotation_error_deg",
                     &VideoRotationAveragingOptions::max_rotation_error_deg)
      .def_readwrite("tracking_huber_scale",
                     &VideoRotationAveragingOptions::video_tracking_huber_scale)
      .def_readwrite("loop_closure_cauchy_scale",
                     &VideoRotationAveragingOptions::video_lc_cauchy_scale)
      .def_readwrite("num_threads", &VideoRotationAveragingOptions::num_threads)
      .def_readwrite("max_num_iterations",
                     &VideoRotationAveragingOptions::max_num_iterations)
      .def_readwrite("use_imu", &VideoRotationAveragingOptions::use_imu)
      .def_readwrite("imu_from_cam",
                     &VideoRotationAveragingOptions::imu_from_cam)
      .def_readwrite("refine_gyro_bias",
                     &VideoRotationAveragingOptions::refine_gyro_bias)
      .def_readwrite("auto_initialize_gyro_bias",
                     &VideoRotationAveragingOptions::auto_initialize_gyro_bias)
      .def_readwrite("use_gyro_bias_prior",
                     &VideoRotationAveragingOptions::use_gyro_bias_prior)
      .def_readwrite("gyro_bias_prior",
                     &VideoRotationAveragingOptions::gyro_bias_prior)
      .def_readwrite("gyro_bias_prior_stddev",
                     &VideoRotationAveragingOptions::gyro_bias_prior_stddev)
      .def_readwrite(
          "apply_bias_prior_to_all_frames",
          &VideoRotationAveragingOptions::apply_bias_prior_to_all_frames)
      .def_readwrite("visual_rotation_stddev_deg",
                     &VideoRotationAveragingOptions::visual_rotation_stddev_deg)
      .def_readwrite(
          "imu_tracking_cauchy_scale_deg",
          &VideoRotationAveragingOptions::imu_tracking_cauchy_scale_deg)
      .def_readwrite(
          "reintegrate_angle_norm_thres",
          &VideoRotationAveragingOptions::reintegrate_angle_norm_thres)
      .def_readwrite("invalidate_outlier_pairs",
                     &VideoRotationAveragingOptions::invalidate_outlier_pairs)
      .def_readwrite(
          "salvage_outlier_translations",
          &VideoRotationAveragingOptions::salvage_outlier_translations)
      .def_readwrite(
          "salvage_epipolar_angle_thres_deg",
          &VideoRotationAveragingOptions::salvage_epipolar_angle_thres_deg)
      .def_readwrite("salvage_min_inlier_ratio",
                     &VideoRotationAveragingOptions::salvage_min_inlier_ratio)
      .def_readwrite("salvage_min_inliers",
                     &VideoRotationAveragingOptions::salvage_min_inliers)
      .def("validate", &VideoRotationAveragingOptions::Validate);

  py::class_<RotationAveragingResult>(m, "RotationAveragingResult")
      .def_readonly("success", &RotationAveragingResult::success)
      .def_readonly("registered_image_ids",
                    &RotationAveragingResult::registered_image_ids)
      .def_readonly("outlier_pair_ids",
                    &RotationAveragingResult::outlier_pair_ids)
      .def_readonly("salvaged_pair_ids",
                    &RotationAveragingResult::salvaged_pair_ids)
      .def_readonly("initial_gyro_bias",
                    &RotationAveragingResult::initial_gyro_bias)
      .def_readonly("initial_gravity_direction",
                    &RotationAveragingResult::initial_gravity_direction)
      .def_readonly("imu_states", &RotationAveragingResult::imu_states);

  m.def("run_video_rotation_averaging",
        &RunVideoRotationAveraging,
        py::arg("options"),
        py::arg("image_map_order"),
        py::arg("pair_map_order"),
        py::arg("problem"),
        py::arg("imu_edges") = std::vector<ImuEdgeRecord>{},
        py::arg("imu_states") = std::vector<ImuStateRecord>{});
}

}  // namespace vidmap
