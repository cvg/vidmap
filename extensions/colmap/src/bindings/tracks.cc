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
      .def_readwrite("required_tracks_per_view",
                     &TrackEstablishmentOptions::required_tracks_per_view)
      .def("validate", &TrackEstablishmentOptions::Validate);

  py::class_<TrackProblemFilterOptions>(m, "TrackProblemFilterOptions")
      .def(py::init<>())
      .def_readwrite("min_num_views_per_track",
                     &TrackProblemFilterOptions::min_num_views_per_track)
      .def_readwrite("max_num_views_per_track",
                     &TrackProblemFilterOptions::max_num_views_per_track)
      .def_readwrite("two_view_depth_gate",
                     &TrackProblemFilterOptions::two_view_depth_gate)
      .def("validate", &TrackProblemFilterOptions::Validate);

  py::class_<TrackFilterResult>(m, "TrackFilterResult")
      .def_readonly("tracks", &TrackFilterResult::tracks)
      .def_readonly("counter", &TrackFilterResult::counter);

  m.def("establish_full_tracks",
        &EstablishTracksFromCorrGraph,
        py::arg("problem"),
        py::arg("image_order"),
        py::arg("pair_order"),
        py::arg("options"),
        py::arg("loop_closure_second_pass") = false,
        py::arg("loop_closure_pair_order") = std::vector<PairId>{});
  m.def("filter_tracks_for_problem",
        &FilterTracksForProblem,
        py::arg("problem"),
        py::arg("registered_image_ids"),
        py::arg("tracks_full"),
        py::arg("options"));
  m.def("filter_tracks_by_angle",
        &FilterTracksByAngle,
        py::arg("problem"),
        py::arg("tracks"),
        py::arg("max_angle_error_deg") = 1.0);
  m.def("filter_tracks_by_triangulation_angle",
        &FilterTrackTriangulationAngle,
        py::arg("problem"),
        py::arg("tracks"),
        py::arg("min_angle_deg") = 1.0);
}

}  // namespace vidmap
