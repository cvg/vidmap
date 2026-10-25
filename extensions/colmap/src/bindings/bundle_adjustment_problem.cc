#include "colmap/scene/camera.h"

#include <memory>
#include <vector>

#include "stages/intrinsics_prior.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(bundle_adjustment, m) {
  py::module_::import("pyceres");
  m.def("focal_prior_cost",
        [](const colmap::Camera& camera, double focal, double stddev) {
          const auto indices = camera.FocalLengthIdxs();
          return std::shared_ptr<ceres::CostFunction>(
              new vidmap::LogMeanFocalPriorCostFunction(
                  camera.params.size(),
                  std::vector<std::size_t>(indices.begin(), indices.end()),
                  focal,
                  stddev));
        });
}
