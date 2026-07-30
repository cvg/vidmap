#include "bindings.h"
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
      .def_readwrite("point3D_id", &TrackRecord::point3D_id)
      .def_readwrite("xyz", &TrackRecord::xyz)
      .def_readwrite("color", &TrackRecord::color)
      .def_readwrite("error", &TrackRecord::error)
      .def_readwrite("observations", &TrackRecord::observations)
      .def_readwrite("loop_closure_observations",
                     &TrackRecord::loop_closure_observations)
      .def_readwrite("loop_closure_anchors", &TrackRecord::loop_closure_anchors)
      .def("validate", &TrackRecord::Validate);
}

}  // namespace vidmap
