#include "vidmap_native/view_graph.h"

#include "bindings.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindViewGraph(py::module_& m) {
  py::class_<InlierThresholdOptions>(m, "InlierThresholdOptions")
      .def(py::init<>())
      .def_readwrite("max_epipolar_error_essential",
                     &InlierThresholdOptions::max_epipolar_error_essential)
      .def_readwrite("max_epipolar_error_fundamental",
                     &InlierThresholdOptions::max_epipolar_error_fundamental)
      .def_readwrite("max_epipolar_error_homography",
                     &InlierThresholdOptions::max_epipolar_error_homography)
      .def_readwrite("min_angle_from_epipole_deg",
                     &InlierThresholdOptions::min_angle_from_epipole_deg)
      .def("validate", &InlierThresholdOptions::Validate);

  py::class_<ViewGraphCalibrationOptions>(m, "FocalCalibrationOptions")
      .def(py::init<>())
      .def_readwrite("min_focal_length_ratio",
                     &ViewGraphCalibrationOptions::min_focal_length_ratio)
      .def_readwrite("max_focal_length_ratio",
                     &ViewGraphCalibrationOptions::max_focal_length_ratio)
      .def_readwrite("max_calibration_error",
                     &ViewGraphCalibrationOptions::max_calibration_error)
      .def_readwrite("loss_function_scale",
                     &ViewGraphCalibrationOptions::loss_function_scale)
      .def_readwrite("num_threads", &ViewGraphCalibrationOptions::num_threads)
      .def_readwrite("max_num_iterations",
                     &ViewGraphCalibrationOptions::max_num_iterations)
      .def_readwrite("function_tolerance",
                     &ViewGraphCalibrationOptions::function_tolerance)
      .def("validate", &ViewGraphCalibrationOptions::Validate);

  py::class_<FocalLengthCalibResult>(m, "FocalCalibrationResult")
      .def(py::init<>())
      .def_readonly("success", &FocalLengthCalibResult::success);

  m.def("prepare_image_bearings", &PrepareImageBearings, py::arg("problem"));
  m.def("update_image_pair_configurations",
        &UpdateImagePairsConfig,
        py::arg("problem"));
  m.def("decompose_relative_poses", &DecomposeRelPose, py::arg("problem"));
  m.def("score_image_pair_inliers",
        &ImagePairsInlierCount,
        py::arg("options"),
        py::arg("clean_inliers"),
        py::arg("problem"));
  m.def("filter_pairs_by_inlier_count",
        &FilterPairsByInlierNum,
        py::arg("min_inlier_count"),
        py::arg("problem"));
  m.def("filter_pairs_by_inlier_ratio",
        &FilterPairsByInlierRatio,
        py::arg("min_inlier_ratio"),
        py::arg("problem"));
  m.def("calibrate_focal_lengths",
        &CalibrateFocalLengths,
        py::arg("options"),
        py::arg("problem"));
  m.def("apply_focal_calibration",
        &ApplyFocalCalibration,
        py::arg("options"),
        py::arg("result"),
        py::arg("problem"));
}

}  // namespace vidmap
