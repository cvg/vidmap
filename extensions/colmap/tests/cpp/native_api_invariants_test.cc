#include <stdexcept>
#include <string>

#include "vidmap_native/bundle_adjustment.h"
#include "vidmap_native/global_positioning.h"
#include "vidmap_native/tracks.h"
#include "vidmap_native/types.h"
#include "vidmap_native/video_rotation_averaging.h"
#include "vidmap_native/view_graph.h"

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callable>
void CheckInvalidArgument(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename Callable>
void CheckInvalidArgumentContains(Callable&& callable,
                                  const std::string& expected,
                                  const std::string& message) {
  try {
    callable();
  } catch (const std::invalid_argument& error) {
    Check(std::string(error.what()).find(expected) != std::string::npos,
          message + ": unexpected error: " + error.what());
    return;
  }
  throw std::runtime_error(message);
}

void TestStableIdentifiers() {
  Check(vidmap::CanonicalPairId(1, 2) == vidmap::CanonicalPairId(2, 1),
        "canonical pair IDs must be orientation independent");
  Check(vidmap::EncodeObservationKey(7, 11) ==
            (static_cast<vidmap::Point3DId>(7) << 32 | 11),
        "track observation encoding changed");
}

void TestSolverDefaults() {
  const vidmap::GlobalPositionerOptions options;
  Check(
      options.parameter_ordering == vidmap::GlobalPositioningOrdering::kGrouped,
      "global positioning ordering default changed");
  Check(options.center_mode == vidmap::GlobalPositioningCenterMode::kFrame,
        "global positioning center default changed");
  Check(options.solver_backend.linear_solver ==
            vidmap::LinearSolverType::kSparseSchur,
        "linear solver default changed");
  Check(!options.solver_backend.use_cuda, "CUDA solver default changed");
  options.Validate();
}

void TestSolverBackendCapabilities() {
  Check(std::string(vidmap::CeresVersion()).find('.') != std::string::npos,
        "Ceres version is not exposed");

  vidmap::SolverBackendOptions dense_options;
  dense_options.linear_solver = vidmap::LinearSolverType::kDenseSchur;
  dense_options.use_cuda = true;
  if (vidmap::IsCudaDenseSolverAvailable()) {
    dense_options.Validate();
    ceres::Solver::Options ceres_options;
    dense_options.Apply(&ceres_options);
    Check(ceres_options.linear_solver_type == ceres::DENSE_SCHUR,
          "dense Schur selection was not applied");
    Check(ceres_options.dense_linear_algebra_library_type == ceres::CUDA,
          "dense CUDA backend was not applied");
  } else {
    CheckInvalidArgumentContains([&] { dense_options.Validate(); },
                                 "CUDA support",
                                 "unavailable dense CUDA was accepted");
  }

  vidmap::SolverBackendOptions sparse_options;
  sparse_options.linear_solver = vidmap::LinearSolverType::kSparseSchur;
  sparse_options.use_cuda = true;
  if (vidmap::IsCudaSparseSolverAvailable()) {
    sparse_options.Validate();
    ceres::Solver::Options ceres_options;
    sparse_options.Apply(&ceres_options);
    Check(ceres_options.linear_solver_type == ceres::SPARSE_SCHUR,
          "sparse Schur selection was not applied");
    Check(
        ceres_options.sparse_linear_algebra_library_type == ceres::CUDA_SPARSE,
        "sparse CUDA backend was not applied");
  } else {
    CheckInvalidArgumentContains([&] { sparse_options.Validate(); },
                                 "Ceres 2.3",
                                 "unavailable sparse CUDA was accepted");
  }

  vidmap::SolverBackendOptions iterative_options;
  iterative_options.linear_solver = vidmap::LinearSolverType::kIterativeSchur;
  iterative_options.use_cuda = true;
  CheckInvalidArgumentContains([&] { iterative_options.Validate(); },
                               "CPU-only",
                               "iterative CUDA was accepted");

  vidmap::GlobalPositionerOptions global_options;
  global_options.solver_backend.linear_solver =
      vidmap::LinearSolverType::kDenseSchur;
  CheckInvalidArgumentContains([&] { global_options.Validate(); },
                               "not supported for global positioning",
                               "dense global positioning was accepted");
}

void TestOptionValidation() {
  vidmap::TrackEstablishmentOptions track_options;
  track_options.required_tracks_per_view = -1;
  CheckInvalidArgument([&] { track_options.Validate(); },
                       "negative track quota was accepted");

  vidmap::InlierThresholdOptions inlier_options;
  inlier_options.min_angle_from_epipole_deg = 181.0;
  CheckInvalidArgument([&] { inlier_options.Validate(); },
                       "invalid epipole angle was accepted");

  vidmap::VideoRotationAveragingOptions rotation_options;
  rotation_options.num_threads = 0;
  CheckInvalidArgument([&] { rotation_options.Validate(); },
                       "zero rotation-averaging threads were accepted");

  vidmap::BundleAdjustmentOptions bundle_options;
  bundle_options.max_num_iterations = 0;
  CheckInvalidArgument([&] { bundle_options.Validate(); },
                       "zero bundle-adjustment iterations were accepted");
}

}  // namespace

int main() {
  TestStableIdentifiers();
  TestSolverDefaults();
  TestSolverBackendCapabilities();
  TestOptionValidation();
  return 0;
}
