#include "bindings.h"
#include "vidmap_native/solver_backend.h"
#include <pybind11/pybind11.h>

namespace {

constexpr int kApiVersion = 5;

}  // namespace

#ifndef VIDMAP_COLMAP_REVISION
#error "VIDMAP_COLMAP_REVISION must be defined by CMake"
#endif

PYBIND11_MODULE(_core, m) {
  m.doc() = "VidMap native mapping algorithms";
  m.attr("__api_version__") = kApiVersion;
  m.attr("__colmap_revision__") = VIDMAP_COLMAP_REVISION;
  m.attr("__ceres_version__") = vidmap::CeresVersion();
  m.attr("__cuda_dense_solver_available__") =
      vidmap::IsCudaDenseSolverAvailable();
  m.attr("__cuda_sparse_solver_available__") =
      vidmap::IsCudaSparseSolverAvailable();
  vidmap::BindRecords(m);
  vidmap::BindSolverPlayback(m);
  vidmap::BindMappingProblem(m);
  vidmap::BindViewGraph(m);
  vidmap::BindTracks(m);
  vidmap::BindVideoRotationAveraging(m);
  vidmap::BindGlobalPositioning(m);
  vidmap::BindBundleAdjustment(m);
}
