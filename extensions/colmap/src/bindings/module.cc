#include "bindings.h"
#include <pybind11/pybind11.h>

#ifndef VIDMAP_COLMAP_REVISION
#error "VIDMAP_COLMAP_REVISION must be defined by CMake"
#endif

PYBIND11_MODULE(_core, m) {
  m.doc() = "VidMap native mapping algorithms";
  m.attr("__colmap_revision__") = VIDMAP_COLMAP_REVISION;
  pybind11::module_::import("pycolmap");
  pybind11::module_::import("pyceres");
  vidmap::BindRecords(m);
  vidmap::BindMappingSidecars(m);
  vidmap::BindViewGraph(m);
  vidmap::BindTracks(m);
  vidmap::BindVideoRotationAveraging(m);
  vidmap::BindGlobalPositioning(m);
}
