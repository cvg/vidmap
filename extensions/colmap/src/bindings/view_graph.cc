#include "vidmap_native/view_graph.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/geometry/essential_matrix.h"

#include "bindings.h"

#include <limits>
#include <vector>

#include <pybind11/eigen.h>
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

  m.def("filter_edges_by_relative_rotation",
        &colmap::FilterEdgesByRelativeRotation);
  m.def("prepare_image_bearings", &PrepareImageBearings);
  m.def("reclassify_calibrated_planar_pairs", &ReclassifyCalibratedPlanarPairs);
  m.def("score_image_pair_inliers", &ImagePairsInlierCount);
  m.def(
      "relative_pose_residuals",
      [](const colmap::Rigid3d& pose,
         const colmap::Camera& camera1,
         const Eigen::Ref<const Eigen::MatrixX2d>& points1,
         const colmap::Camera& camera2,
         const Eigen::Ref<const Eigen::MatrixX2d>& points2) {
        THROW_CHECK_EQ(points1.rows(), points2.rows());
        const Eigen::Matrix3d E = colmap::EssentialMatrixFromPose(pose);
        Eigen::VectorXd errors(points1.rows());
        for (Eigen::Index i = 0; i < points1.rows(); ++i) {
          const auto ray1 = camera1.CamRayFromImgWithJac(points1.row(i));
          const auto ray2 = camera2.CamRayFromImgWithJac(points2.row(i));
          errors[i] = ray1 && ray2 ? colmap::ComputeSquaredTangentSampsonError(
                                         *ray1, *ray2, E)
                                   : std::numeric_limits<double>::infinity();
        }
        return errors;
      },
      py::call_guard<py::gil_scoped_release>());
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
