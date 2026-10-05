#include "vidmap_native/tracks.h"

#include "bindings.h"
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

void BindTracks(py::module_& m) {
  py::class_<TrackEstablishmentOptions>(m, "TrackEstablishmentOptions")
      .def(py::init<>())
      .def_readwrite(
          "intra_image_consistency_threshold",
          &TrackEstablishmentOptions::intra_image_consistency_threshold)
      .def_readwrite("min_num_views_per_track",
                     &TrackEstablishmentOptions::min_num_views_per_track)
      .def_readwrite("max_num_views_per_track",
                     &TrackEstablishmentOptions::max_num_views_per_track)
      .def_readwrite("two_view_depth_gate",
                     &TrackEstablishmentOptions::two_view_depth_gate);

  py::class_<TrackEstablishmentResult>(m, "TrackEstablishmentResult")
      .def_readonly("num_full_tracks",
                    &TrackEstablishmentResult::num_full_tracks)
      .def_readonly("num_tracks", &TrackEstablishmentResult::num_tracks)
      .def_readonly("full_tracks", &TrackEstablishmentResult::full_tracks)
      .def_readonly("full_track_data",
                    &TrackEstablishmentResult::full_track_data);
  m.def("establish_tracks",
        &EstablishAndCommitTracks,
        py::arg("reconstruction"),
        py::arg("pose_graph"),
        py::arg("sidecars"),
        py::arg("image_order"),
        py::arg("pair_order"),
        py::arg("options"),
        py::arg("loop_closure_second_pass") = false,
        py::arg("include_loop_closure_observations") = true,
        py::arg("capture_tracks") = false);
  m.def("create_correspondence_graph", &CreateCorrespondenceGraph);
  m.def("filter_correspondence_graph", &FilterCorrespondenceGraph);
}
}  // namespace vidmap
