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

  m.def("prepare_image_bearings", &PrepareImageBearings);
  m.def("reclassify_calibrated_planar_pairs", &ReclassifyCalibratedPlanarPairs);
  m.def("score_image_pair_inliers", &ImagePairsInlierCount);
  m.def("filter_pairs_by_inlier_count", &FilterPairsByInlierNum);
  m.def("filter_pairs_by_inlier_ratio", &FilterPairsByInlierRatio);
  m.def("calibrate_focal_lengths",
        &CalibrateFocalLengths,
        py::arg("options"),
        py::arg("reconstruction"),
        py::arg("pose_graph"),
        py::arg("sidecars"),
        py::arg("focal_priors") = std::vector<LogFocalPriorRecord>{});
}
}  // namespace vidmap
