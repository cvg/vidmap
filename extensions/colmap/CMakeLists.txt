cmake_minimum_required(VERSION 3.28)

project(vidmap_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Ceres may live outside the loader's search path, so keep the directories it was
# linked from in the installed module's RPATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 CONFIG REQUIRED)

set(VIDMAP_COLMAP_VERSION "4.3.0")
set(
  VIDMAP_COLMAP_REVISION
  "ea063e6874583429143dd5c53d86c3e68c895e8f"
  CACHE STRING
  "Upstream COLMAP revision used by vidmap_native"
)
find_package(colmap CONFIG REQUIRED)
if(NOT colmap_VERSION VERSION_EQUAL VIDMAP_COLMAP_VERSION)
  message(
    WARNING
    "VidMap was tested with COLMAP ${VIDMAP_COLMAP_VERSION}; building against "
    "${colmap_VERSION}. Unsupported versions may fail to compile or run."
  )
endif()
set(VIDMAP_COLMAP_ESTIMATORS_TARGET colmap::colmap_estimators)
get_target_property(
  VIDMAP_COLMAP_INCLUDE_DIRS
  colmap::colmap
  INTERFACE_INCLUDE_DIRECTORIES
)

set(
  VIDMAP_NATIVE_ALGORITHM_SOURCES
    src/types.cc
    src/mapping_sidecars.cc
    src/stages/image_pair_inliers.cc
    src/stages/view_graph_manipulation.cc
    src/stages/view_graph_calibration.cc
    src/stages/track_establishment.cc
    src/stages/video_rotation_averaging.cc
)

add_library(vidmap_native_algorithms STATIC ${VIDMAP_NATIVE_ALGORITHM_SOURCES})
set_target_properties(vidmap_native_algorithms PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  vidmap_native_algorithms
  PRIVATE
    src
  PUBLIC
    ${VIDMAP_COLMAP_INCLUDE_DIRS}
)
target_link_libraries(vidmap_native_algorithms PUBLIC ${VIDMAP_COLMAP_ESTIMATORS_TARGET})

# COLMAP's static libraries can reference Abseil's logging internals without listing
# Abseil in their exported configuration, which leaves the module with undefined
# absl::log_internal symbols at import time.
find_package(absl CONFIG QUIET)
if(absl_FOUND)
  target_link_libraries(vidmap_native_algorithms PUBLIC absl::log absl::check)
endif()

set(
  VIDMAP_NATIVE_BINDING_SOURCES
    src/bindings/module.cc
    src/bindings/records.cc
    src/bindings/mapping_sidecars.cc
    src/bindings/view_graph.cc
    src/bindings/tracks.cc
    src/bindings/video_rotation_averaging.cc
    src/bindings/global_positioning.cc
)

pybind11_add_module(
  vidmap_native_core
  NO_EXTRAS
  MODULE
    ${VIDMAP_NATIVE_BINDING_SOURCES}
)
set_target_properties(
  vidmap_native_core
  PROPERTIES
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vidmap_native"
)
target_include_directories(
  vidmap_native_core
  PRIVATE
    src
)
target_link_libraries(vidmap_native_core PRIVATE vidmap_native_algorithms)
target_compile_definitions(
  vidmap_native_core
  PRIVATE VIDMAP_COLMAP_REVISION="${VIDMAP_COLMAP_REVISION}"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_vidmap_export_map "${CMAKE_CURRENT_BINARY_DIR}/vidmap_native.exports")
  file(WRITE "${_vidmap_export_map}" "{\n  global: PyInit__core;\n  local: *;\n};\n")
  target_link_options(
    vidmap_native_core
    PRIVATE
      "LINKER:--exclude-libs,ALL"
      "LINKER:--version-script=${_vidmap_export_map}"
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(
    vidmap_native_api_invariants_test
    tests/cpp/native_api_invariants_test.cc
  )
  target_link_libraries(
    vidmap_native_api_invariants_test
    PRIVATE
      vidmap_native_algorithms
  )
  target_include_directories(vidmap_native_api_invariants_test PRIVATE src)
  add_test(
    NAME vidmap_native_api_invariants_test
    COMMAND vidmap_native_api_invariants_test
  )
endif()

install(TARGETS vidmap_native_core LIBRARY DESTINATION vidmap_native)

pybind11_add_module(vidmap_gp_costs NO_EXTRAS src/bindings/global_positioning_problem.cc)
set_target_properties(vidmap_gp_costs PROPERTIES OUTPUT_NAME global_positioning)
target_include_directories(vidmap_gp_costs PRIVATE src)
target_link_libraries(vidmap_gp_costs PRIVATE vidmap_native_algorithms)
install(TARGETS vidmap_gp_costs LIBRARY DESTINATION vidmap_native)

pybind11_add_module(vidmap_ba_costs NO_EXTRAS src/bindings/bundle_adjustment_problem.cc)
set_target_properties(vidmap_ba_costs PROPERTIES OUTPUT_NAME bundle_adjustment)
target_include_directories(vidmap_ba_costs PRIVATE src ${VIDMAP_COLMAP_INCLUDE_DIRS})
target_link_libraries(vidmap_ba_costs PRIVATE ${VIDMAP_COLMAP_ESTIMATORS_TARGET})
install(TARGETS vidmap_ba_costs LIBRARY DESTINATION vidmap_native)
