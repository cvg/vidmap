#include "colmap/scene/point3d.h"

#include "bindings.h"
#include "vidmap_native/imu_types.h"
#include "vidmap_native/types.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindRecords(py::module_& m) {
  py::class_<CameraRecord>(m, "CameraRecord")
      .def(py::init<>())
      .def_readwrite("camera_id", &CameraRecord::camera_id)
      .def_readwrite("model_id", &CameraRecord::model_id)
      .def_readwrite("width", &CameraRecord::width)
      .def_readwrite("height", &CameraRecord::height)
      .def_readwrite("params", &CameraRecord::params)
      .def_readwrite("has_prior_focal_length",
                     &CameraRecord::has_prior_focal_length)
      .def("validate", &CameraRecord::Validate);

  py::class_<PoseRecord>(m, "PoseRecord")
      .def(py::init<>())
      .def_readwrite("has_pose", &PoseRecord::has_pose)
      .def_readwrite("rotation_xyzw", &PoseRecord::rotation_xyzw)
      .def_readwrite("translation", &PoseRecord::translation)
      .def("validate", &PoseRecord::Validate);

  py::class_<ImageRecord>(m, "ImageRecord")
      .def(py::init<>())
      .def_readwrite("image_id", &ImageRecord::image_id)
      .def_readwrite("camera_id", &ImageRecord::camera_id)
      .def_readwrite("frame_id", &ImageRecord::frame_id)
      .def_readwrite("name", &ImageRecord::name)
      .def_readwrite("pose", &ImageRecord::pose)
      .def_readwrite("keypoints", &ImageRecord::keypoints)
      .def_readwrite("bearings", &ImageRecord::bearings)
      .def_readwrite("depth_values", &ImageRecord::depth_values)
      .def_readwrite("depth_stddevs", &ImageRecord::depth_stddevs)
      .def_readwrite("depth_validity", &ImageRecord::depth_validity)
      .def_readwrite("angular_stddevs", &ImageRecord::angular_stddevs)
      .def_readwrite("is_inlier", &ImageRecord::is_inlier)
      .def_readwrite("is_track_anchor", &ImageRecord::is_track_anchor)
      .def_readwrite("is_depth_outlier", &ImageRecord::is_depth_outlier)
      .def_property_readonly("num_features", &ImageRecord::NumFeatures)
      .def("validate", &ImageRecord::Validate);

  py::class_<TwoViewGeometryRecord>(m, "TwoViewGeometryRecord")
      .def(py::init<>())
      .def_readwrite("configuration", &TwoViewGeometryRecord::configuration)
      .def_readwrite("has_essential", &TwoViewGeometryRecord::has_essential)
      .def_readwrite("has_fundamental", &TwoViewGeometryRecord::has_fundamental)
      .def_readwrite("has_homography", &TwoViewGeometryRecord::has_homography)
      .def_readwrite("essential", &TwoViewGeometryRecord::essential)
      .def_readwrite("fundamental", &TwoViewGeometryRecord::fundamental)
      .def_readwrite("homography", &TwoViewGeometryRecord::homography)
      .def_readwrite("cam2_from_cam1", &TwoViewGeometryRecord::cam2_from_cam1)
      .def("validate", &TwoViewGeometryRecord::Validate);

  py::class_<PairRecord>(m, "PairRecord")
      .def(py::init<>())
      .def_readwrite("pair_id", &PairRecord::pair_id)
      .def_readwrite("image_id1", &PairRecord::image_id1)
      .def_readwrite("image_id2", &PairRecord::image_id2)
      .def_readwrite("is_valid", &PairRecord::is_valid)
      .def_readwrite("geometry", &PairRecord::geometry)
      .def_readwrite("all_matches", &PairRecord::all_matches)
      .def_readwrite("inlier_indices", &PairRecord::inlier_indices)
      .def_readwrite("are_loop_closure", &PairRecord::are_loop_closure)
      .def("validate", &PairRecord::Validate);

  py::class_<TrackRecord>(m, "TrackRecord")
      .def(py::init<>())
      .def(py::init([](Point3DId point3D_id, const colmap::Point3D& point) {
        TrackRecord record;
        record.point3D_id = point3D_id;
        record.xyz = point.xyz;
        record.color = point.color;
        record.error = point.error;
        record.observations.resize(point.track.Length(), 2);
        for (std::size_t i = 0; i < point.track.Length(); ++i) {
          record.observations(i, 0) = point.track.Element(i).image_id;
          record.observations(i, 1) = point.track.Element(i).point2D_idx;
        }
        return record;
      }))
      .def_readwrite("point3D_id", &TrackRecord::point3D_id)
      .def_readwrite("xyz", &TrackRecord::xyz)
      .def_readwrite("color", &TrackRecord::color)
      .def_readwrite("error", &TrackRecord::error)
      .def_readwrite("observations", &TrackRecord::observations)
      .def_readwrite("loop_closure_observations",
                     &TrackRecord::loop_closure_observations)
      .def_readwrite("loop_closure_anchors", &TrackRecord::loop_closure_anchors)
      .def("validate", &TrackRecord::Validate);

  py::class_<ImuStateRecord>(m, "ImuStateRecord")
      .def(py::init<>())
      .def_readwrite("image_id", &ImuStateRecord::image_id)
      .def_readwrite("velocity", &ImuStateRecord::velocity)
      .def_readwrite("metric_velocity", &ImuStateRecord::metric_velocity)
      .def_readwrite("bias_gyro", &ImuStateRecord::bias_gyro)
      .def_readwrite("bias_accel", &ImuStateRecord::bias_accel)
      .def("to_vector", &ImuStateRecord::ToVector)
      .def_static("from_vector",
                  &ImuStateRecord::FromVector,
                  py::arg("image_id"),
                  py::arg("vec"),
                  py::arg("scale") = 1.0)
      .def("validate", &ImuStateRecord::Validate);

  py::class_<ImuEdgeRecord>(m, "ImuEdgeRecord")
      .def(py::init<>())
      .def_readwrite("image_id1", &ImuEdgeRecord::image_id1)
      .def_readwrite("image_id2", &ImuEdgeRecord::image_id2)
      .def_readwrite("data", &ImuEdgeRecord::data)
      .def(
          "set_integrator",
          [](ImuEdgeRecord& self, colmap::ImuPreintegrator* integrator) {
            self.integrator = integrator;
          },
          py::arg("integrator"),
          py::keep_alive<1, 2>())
      .def_property_readonly(
          "has_integrator",
          [](const ImuEdgeRecord& self) { return self.integrator != nullptr; })
      .def_property(
          "q_iori_1_xyzw",
          [](const ImuEdgeRecord& self) -> Eigen::Vector4d {
            return self.q_iori_1_xyzw.coeffs();
          },
          [](ImuEdgeRecord& self, const Eigen::Vector4d& xyzw) {
            self.q_iori_1_xyzw.coeffs() = xyzw;
          })
      .def_property(
          "q_iori_2_xyzw",
          [](const ImuEdgeRecord& self) -> Eigen::Vector4d {
            return self.q_iori_2_xyzw.coeffs();
          },
          [](ImuEdgeRecord& self, const Eigen::Vector4d& xyzw) {
            self.q_iori_2_xyzw.coeffs() = xyzw;
          })
      .def_readwrite("loss", &ImuEdgeRecord::loss)
      .def("validate", &ImuEdgeRecord::Validate);
}

}  // namespace vidmap
