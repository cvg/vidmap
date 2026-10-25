#include "vidmap_native/tracks.h"

#include "colmap/scene/correspondence_graph.h"

#include <unordered_map>
#include <unordered_set>

#include "bindings.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {

// Keep COLMAP traversal in native code and materialize each pair once.
py::dict BuildTransitiveCorrespondences(
    const colmap::CorrespondenceGraph& graph,
    const std::vector<colmap::image_t>& images,
    const std::vector<size_t>& counts,
    size_t depth) {
  if (images.size() != counts.size())
    throw std::invalid_argument("image and observation inventories differ");
  struct Entry {
    colmap::image_t first, second;
    std::vector<std::pair<colmap::point2D_t, colmap::point2D_t>> rows;
    std::unordered_set<uint64_t> unique;
  };
  std::vector<Entry> entries;
  {
    py::gil_scoped_release release;
    std::unordered_map<uint64_t, size_t> indices;
    std::vector<colmap::CorrespondenceGraph::Correspondence> corrs;
    for (size_t i = 0; i < images.size(); ++i) {
      const auto image = images[i];
      if (!graph.ExistsImage(image)) continue;
      for (colmap::point2D_t point = 0; point < counts[i]; ++point) {
        graph.ExtractTransitiveCorrespondences(image, point, depth, &corrs);
        for (const auto& corr : corrs) {
          const uint64_t forward = (uint64_t(image) << 32) | corr.image_id;
          const uint64_t reverse = (uint64_t(corr.image_id) << 32) | image;
          if (indices.count(reverse)) continue;
          const auto [it, inserted] = indices.emplace(forward, entries.size());
          if (inserted) entries.push_back({image, corr.image_id, {}, {}});
          auto& entry = entries[it->second];
          const uint64_t row = (uint64_t(point) << 32) | corr.point2D_idx;
          if (entry.unique.insert(row).second)
            entry.rows.emplace_back(point, corr.point2D_idx);
        }
      }
    }
    for (auto& entry : entries)
      std::unordered_set<uint64_t>().swap(entry.unique);
  }
  py::dict result;
  for (const auto& entry : entries) {
    // Preserve the established Python tuple-set row ordering exactly.
    py::set unique;
    for (const auto& row : entry.rows)
      unique.add(py::make_tuple(row.first, row.second));
    py::array_t<int64_t> array(
        {static_cast<py::ssize_t>(entry.rows.size()), py::ssize_t(2)});
    auto rows = array.mutable_unchecked<2>();
    py::ssize_t i = 0;
    for (const auto& row : unique) {
      const auto pair = py::cast<std::pair<int64_t, int64_t>>(row);
      rows(i, 0) = pair.first;
      rows(i++, 1) = pair.second;
    }
    result[py::make_tuple(entry.first, entry.second)] = std::move(array);
  }
  return result;
}

void BindTracks(py::module_& m) {
  m.def("build_transitive_correspondences", &BuildTransitiveCorrespondences);
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
