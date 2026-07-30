#pragma once

#include <pybind11/pybind11.h>

namespace vidmap {

void BindRecords(pybind11::module_& module);
void BindSolverPlayback(pybind11::module_& module);
void BindMappingProblem(pybind11::module_& module);
void BindViewGraph(pybind11::module_& module);
void BindTracks(pybind11::module_& module);
void BindVideoRotationAveraging(pybind11::module_& module);
void BindGlobalPositioning(pybind11::module_& module);
void BindBundleAdjustment(pybind11::module_& module);

}  // namespace vidmap
