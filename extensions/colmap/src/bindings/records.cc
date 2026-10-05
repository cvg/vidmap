#include "bindings.h"
#include "vidmap_native/focal_prior.h"
#include "vidmap_native/types.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace vidmap {
void BindRecords(py::module_& m) {
  py::class_<ImageData>(m, "ImageData")
      .def(py::init<>())
      .def_readwrite("bearings", &ImageData::bearings)
      .def_readwrite("depth_values", &ImageData::depth_values)
      .def_readwrite("depth_stddevs", &ImageData::depth_stddevs)
      .def_readwrite("depth_validity", &ImageData::depth_validity)
      .def_readwrite("is_depth_outlier", &ImageData::is_depth_outlier);
  py::class_<PairData>(m, "PairData")
      .def(py::init<>())
      .def_readwrite("geometry", &PairData::geometry)
      .def_readwrite("has_relative_pose", &PairData::has_relative_pose)
      .def_readwrite("all_matches", &PairData::all_matches)
      .def_readwrite("inlier_indices", &PairData::inlier_indices)
      .def_readwrite("are_loop_closure", &PairData::are_loop_closure)
      .def("validate", &PairData::Validate);
  py::class_<TrackData>(m, "TrackData")
      .def(py::init<>())
      .def_readwrite("loop_closure_observations",
                     &TrackData::loop_closure_observations)
      .def_readwrite("loop_closure_anchors", &TrackData::loop_closure_anchors);
  py::class_<LogFocalPriorRecord>(m, "LogFocalPriorRecord")
      .def(py::init<>())
      .def_readwrite("camera_id", &LogFocalPriorRecord::camera_id)
      .def_readwrite("observations", &LogFocalPriorRecord::observations)
      .def_readwrite("loss", &LogFocalPriorRecord::loss)
      .def("validate", &LogFocalPriorRecord::Validate);
}
}  // namespace vidmap
