#include "vidmap_native/solver_playback.h"

#include <algorithm>
#include <memory>

#include "bindings.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace vidmap {
namespace {

template <typename Value>
py::array_t<Value> VectorArray(const std::vector<Value>& values) {
  py::array_t<Value> result(values.size());
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

py::dict CaptureDict(const SolverPlaybackCapture& value) {
  py::dict capture;
  capture["phase"] = value.phase;
  capture["iteration"] = value.iteration;
  capture["image_ids"] = VectorArray(value.image_ids);
  capture["centers"] = value.centers;
  capture["point_ids"] = VectorArray(value.point3D_ids);
  capture["points_xyz"] = value.points_xyz;
  capture["lc_pairs"] = value.loop_closure_pairs;
  capture["lc_support_count"] = VectorArray(value.loop_closure_support_counts);
  capture["lc_raw_score"] = value.loop_closure_raw_scores;
  return capture;
}

}  // namespace

void BindSolverPlayback(py::module_& m) {
  py::class_<SolverPlaybackOptions>(m, "SolverPlaybackOptions")
      .def(py::init<>())
      .def_readwrite("snapshot_every_n_iterations",
                     &SolverPlaybackOptions::snapshot_every_n_iterations)
      .def_readwrite("image_ids", &SolverPlaybackOptions::image_ids)
      .def_readwrite("point3D_ids", &SolverPlaybackOptions::point3D_ids)
      .def_property(
          "callback",
          [](const SolverPlaybackOptions&) { return py::none(); },
          [](SolverPlaybackOptions& options, const py::object& callback) {
            if (callback.is_none()) {
              options.callback = {};
              return;
            }
            if (!PyCallable_Check(callback.ptr())) {
              throw py::type_error("callback must be callable");
            }
            auto holder = std::shared_ptr<py::object>(
                new py::object(callback), [](py::object* value) {
                  py::gil_scoped_acquire acquire;
                  delete value;
                });
            options.callback = [holder](const SolverPlaybackCapture& value) {
              py::gil_scoped_acquire acquire;
              (*holder)(CaptureDict(value));
            };
          })
      .def("validate", &SolverPlaybackOptions::Validate);
}

}  // namespace vidmap
