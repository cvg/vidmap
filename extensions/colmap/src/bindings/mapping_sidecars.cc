#include "vidmap_native/mapping_sidecars.h"

#include "bindings.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
namespace vidmap {
void BindMappingSidecars(py::module_& m) {
  py::class_<MappingSidecars>(m, "MappingSidecars")
      .def(py::init<>())
      .def("add_image",
           [](MappingSidecars& self, ImageId id, const ImageData& data) {
             self.images.emplace(id, data);
           })
      .def("add_pair",
           [](MappingSidecars& self, PairId id, const PairData& data) {
             self.pairs.emplace(id, data);
           })
      .def("set_track",
           [](MappingSidecars& self, Point3DId id, const TrackData& data) {
             self.tracks[id] = data;
           })
      .def(
          "image",
          [](MappingSidecars& self, ImageId id) -> ImageData& {
            return self.images.at(id);
          },
          py::return_value_policy::reference_internal)
      .def(
          "pair",
          [](MappingSidecars& self, PairId id) -> PairData& {
            return self.pairs.at(id);
          },
          py::return_value_policy::reference_internal)
      .def(
          "track",
          [](MappingSidecars& self, Point3DId id) -> TrackData& {
            return self.tracks.at(id);
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly("track_ids",
                             [](const MappingSidecars& self) {
                               std::vector<Point3DId> ids;
                               for (const auto& [id, data] : self.tracks)
                                 ids.push_back(id);
                               return ids;
                             })
      .def("clear_pairs", [](MappingSidecars& self) { self.pairs.clear(); })
      .def("clear_tracks", [](MappingSidecars& self) { self.tracks.clear(); })
      .def("validate", &MappingSidecars::Validate);
  m.def(
      "point2D_coords",
      [](const colmap::Image& image) {
        Eigen::MatrixXd coords(image.NumPoints2D(), 2);
        for (colmap::point2D_t idx = 0; idx < image.NumPoints2D(); ++idx) {
          coords.row(idx) = image.Point2D(idx).xy;
        }
        return coords;
      },
      py::arg("image"),
      "Get an Nx2 numpy array of xy coordinates for points2D.");
  m.def("image_point3D_ids", [](const colmap::Image& image) {
    py::array_t<colmap::point3D_t> ids(image.NumPoints2D());
    for (std::size_t i = 0; i < image.NumPoints2D(); ++i)
      ids.mutable_data()[i] = image.Point2D(i).point3D_id;
    return ids;
  });
  m.def("point3D_table", [](const colmap::Reconstruction& reconstruction) {
    const py::ssize_t count = reconstruction.NumPoints3D();
    py::array_t<colmap::point3D_t> ids(count);
    py::array_t<double> xyz({count, py::ssize_t(3)});
    py::array_t<std::int64_t> lengths(count);
    std::size_t i = 0;
    for (const auto& [id, point] : reconstruction.Points3D()) {
      ids.mutable_data()[i] = id;
      for (int j = 0; j < 3; ++j) xyz.mutable_data()[3 * i + j] = point.xyz[j];
      lengths.mutable_data()[i++] = point.track.Length();
    }
    return py::make_tuple(ids, xyz, lengths);
  });
  m.def(
      "playback_coordinates",
      [](const colmap::Reconstruction& reconstruction,
         const std::vector<colmap::image_t>& image_ids,
         const py::array_t<colmap::point3D_t>& point_ids,
         const py::dict& centers) {
        MatrixX3d camera_values(image_ids.size(), 3),
            point_values(point_ids.size(), 3);
        for (size_t i = 0; i < image_ids.size(); ++i) {
          const Eigen::Vector3d center =
              centers.empty()
                  ? reconstruction.Image(image_ids[i]).ProjectionCenter()
                  : centers[py::int_(image_ids[i])].cast<Eigen::Vector3d>();
          camera_values.row(i) = center.transpose();
        }
        for (py::ssize_t i = 0; i < point_ids.size(); ++i)
          point_values.row(i) =
              reconstruction.Point3D(*point_ids.data(i)).xyz.transpose();
        return py::make_tuple(camera_values, point_values);
      },
      py::arg("reconstruction"),
      py::arg("image_ids"),
      py::arg("point_ids"),
      py::arg("centers") = py::dict());
  m.def(
      "publish_geometry",
      [](const colmap::Reconstruction& source,
         colmap::Reconstruction& target,
         bool update_poses) {
        for (const auto& [id, camera] : source.Cameras())
          target.Camera(id).params = camera.params;
        if (update_poses) {
          for (const auto& [id, frame] : source.Frames()) {
            if (frame.HasPose())
              target.Frame(id).SetRigFromWorld(frame.RigFromWorld());
          }
        }
        for (const auto& [id, point] : source.Points3D())
          target.Point3D(id).xyz = point.xyz;
      },
      py::arg("source"),
      py::arg("target"),
      py::arg("update_poses") = true);
}
}  // namespace vidmap
