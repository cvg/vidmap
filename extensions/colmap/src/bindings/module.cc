#include "bindings.h"
#include <pybind11/pybind11.h>

#ifndef VIDMAP_COLMAP_REVISION
#error "VIDMAP_COLMAP_REVISION must be defined by CMake"
#endif

PYBIND11_MODULE(_core, m) {
  m.doc() = "VidMap native mapping algorithms";
  m.attr("__colmap_revision__") = VIDMAP_COLMAP_REVISION;
  vidmap::BindRecords(m);
  vidmap::BindSolverPlayback(m);
  vidmap::BindMappingProblem(m);
  vidmap::BindViewGraph(m);
  vidmap::BindTracks(m);
  vidmap::BindVideoRotationAveraging(m);
  vidmap::BindGlobalPositioning(m);
  vidmap::BindBundleAdjustment(m);
}
