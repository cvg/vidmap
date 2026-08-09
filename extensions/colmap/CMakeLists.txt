cmake_minimum_required(VERSION 3.28)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/python/vidmap_native/_version.py" VIDMAP_VERSION_SOURCE)
string(REGEX MATCH "__version__ = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"" _ "${VIDMAP_VERSION_SOURCE}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Cannot read the VidMap version")
endif()
project(vidmap_native VERSION "${CMAKE_MATCH_1}" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Ceres may live outside the loader's search path, so keep the directories it was
# linked from in the installed module's RPATH.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.0.2 CONFIG REQUIRED)

set(VIDMAP_COLMAP_VERSION "4.1.0")
set(
  VIDMAP_COLMAP_REVISION
  "fa8e3b3ff591552855f8ad2806723c80f963f69c"
  CACHE STRING
  "Official COLMAP release revision used by vidmap_native"
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
    src/solver_playback.cc
    src/ceres_loss.cc
    src/solver_backend.cc
    src/conversion.cc
    src/mapping_problem.cc
    src/stages/image_pair_inliers.cc
    src/stages/view_graph_manipulation.cc
    src/stages/view_graph_calibration.cc
    src/stages/track_records.cc
    src/stages/track_establishment.cc
    src/stages/track_filter.cc
    src/stages/video_rotation_averaging.cc
    src/stages/global_positioning.cc
    src/stages/bundle_adjustment.cc
)

add_library(vidmap_native_algorithms STATIC ${VIDMAP_NATIVE_ALGORITHM_SOURCES})
set_target_properties(vidmap_native_algorithms PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  vidmap_native_algorithms
  PRIVATE
    src
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
    src/bindings/solver_playback.cc
    src/bindings/mapping_problem.cc
    src/bindings/view_graph.cc
    src/bindings/tracks.cc
    src/bindings/video_rotation_averaging.cc
    src/bindings/global_positioning.cc
    src/bindings/bundle_adjustment.cc
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
