#include "vidmap_native/mapping_problem.h"

#include "bindings.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindMappingProblem(py::module_& m) {
  py::class_<MappingProblem>(m, "MappingProblem")
      .def(py::init<>())
      .def("add_camera", &MappingProblem::AddCamera)
      .def("add_image", &MappingProblem::AddImage)
      .def("add_pair", &MappingProblem::AddPair)
      .def("add_track", &MappingProblem::AddTrack)
      .def("update_camera", &MappingProblem::UpdateCamera)
      .def("update_image", &MappingProblem::UpdateImage)
      .def("update_pair", &MappingProblem::UpdatePair)
      .def("update_track", &MappingProblem::UpdateTrack)
      .def("clear_tracks", &MappingProblem::ClearTracks)
      .def("camera", &MappingProblem::Camera)
      .def("image", &MappingProblem::Image)
      .def("pair", &MappingProblem::Pair)
      .def("track", &MappingProblem::Track)
      .def_property_readonly("camera_ids", &MappingProblem::CameraIds)
      .def_property_readonly("image_ids", &MappingProblem::ImageIds)
      .def_property_readonly("pair_ids", &MappingProblem::PairIds)
      .def_property_readonly("point3D_ids", &MappingProblem::Point3DIds)
      .def_property_readonly("num_cameras", &MappingProblem::NumCameras)
      .def_property_readonly("num_images", &MappingProblem::NumImages)
      .def_property_readonly("num_pairs", &MappingProblem::NumPairs)
      .def_property_readonly("num_tracks", &MappingProblem::NumTracks)
      .def("validate", &MappingProblem::Validate);
}

}  // namespace vidmap
