#include "vidmap_native/global_positioning.h"

#include "bindings.h"

namespace py = pybind11;

namespace vidmap {

void BindGlobalPositioning(py::module_& m) {
  py::enum_<MetricDepthResidualType>(m, "MetricDepthResidualType")
      .value("LINEAR", MetricDepthResidualType::kLinear)
      .value("LOG", MetricDepthResidualType::kLog)
      .value("LOG_LINEAR", MetricDepthResidualType::kLogLinear);
}

}  // namespace vidmap
