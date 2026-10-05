#include "colmap/estimators/cost_functions/calibration.h"
#include "colmap/util/threading.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stages/intrinsics_prior.h"
#include "vidmap_native/view_graph.h"
#include <ceres/ceres.h>

namespace vidmap {
namespace {

constexpr double kFocalLengthLowerBound = 1e-3;

void ValidateCalibrationOptions(
    const colmap::ViewGraphCalibrationOptions& options) {
  if (options.min_focal_length_ratio <= 0.0 ||
      options.max_focal_length_ratio < options.min_focal_length_ratio ||
      options.max_calibration_error < 0.0 ||
      options.loss_function_scale < 0.0 ||
      options.solver_options.max_num_iterations <= 0 ||
      options.solver_options.function_tolerance < 0.0 ||
      options.solver_options.num_threads == 0) {
    throw std::invalid_argument("invalid focal calibration options");
  }
}

}  // namespace

std::size_t CalibrateFocalLengths(
    const colmap::ViewGraphCalibrationOptions& options,
    colmap::Reconstruction& reconstruction,
    colmap::PoseGraph& graph,
    const MappingSidecars& sidecars,
    const std::vector<LogFocalPriorRecord>& focal_priors) {
  ValidateCalibrationOptions(options);
  sidecars.Validate(reconstruction);
  std::unordered_set<CameraId> prior_camera_ids;
  for (const auto& prior : focal_priors) {
    prior.Validate();
    if (!prior_camera_ids.insert(prior.camera_id).second) {
      throw std::invalid_argument("VGC focal priors require unique cameras");
    }
  }
  struct FocalLengthCalibInput {
    PairId pair_id;
    CameraId camera_id1;
    CameraId camera_id2;
    Eigen::Matrix3d F;
  };

  const auto& cameras = reconstruction.Cameras();
  std::unordered_map<CameraId, double> focal_lengths;
  focal_lengths.reserve(reconstruction.NumCameras());
  for (const auto& [camera_id, camera] : cameras) {
    focal_lengths.emplace(camera_id, camera.MeanFocalLength());
  }
  for (const auto& prior : focal_priors) {
    reconstruction.Camera(prior.camera_id);
  }

  std::vector<FocalLengthCalibInput> inputs;
  for (const auto& [pair_id, pair] : sidecars.pairs) {
    const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
    if (!graph.IsValid(pair_id) || !pair.geometry.F ||
        (pair.geometry.config != colmap::TwoViewGeometry::CALIBRATED &&
         pair.geometry.config != colmap::TwoViewGeometry::UNCALIBRATED)) {
      continue;
    }
    inputs.push_back({pair_id,
                      reconstruction.Image(id1).CameraId(),
                      reconstruction.Image(id2).CameraId(),
                      *pair.geometry.F});
  }

  if (inputs.empty()) return 0;

  auto loss_function = options.CreateLossFunction();
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  std::vector<ceres::ResidualBlockId> fetzer_blocks;
  fetzer_blocks.reserve(inputs.size());
  for (const FocalLengthCalibInput& input : inputs) {
    ceres::ResidualBlockId block = nullptr;
    if (input.camera_id1 == input.camera_id2) {
      block = problem.AddResidualBlock(
          colmap::FetzerFocalLengthSameCameraCostFunctor::Create(
              input.F, cameras.at(input.camera_id1).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths.at(input.camera_id1));
    } else {
      block = problem.AddResidualBlock(
          colmap::FetzerFocalLengthCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              cameras.at(input.camera_id2).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths.at(input.camera_id1),
          &focal_lengths.at(input.camera_id2));
    }
    fetzer_blocks.push_back(block);
  }

  for (const auto& prior : focal_priors) {
    double* focal = &focal_lengths.at(prior.camera_id);
    for (Eigen::Index row = 0; row < prior.observations.rows(); ++row) {
      problem.AddResidualBlock(
          new LogMeanFocalPriorCostFunction(
              1, {0}, prior.observations(row, 0), prior.observations(row, 1)),
          prior.loss.get(),
          focal);
    }
  }

  std::size_t num_cameras = 0;
  for (auto& [camera_id, camera] : cameras) {
    double* focal = &focal_lengths.at(camera_id);
    if (!problem.HasParameterBlock(focal)) continue;
    problem.SetParameterLowerBound(focal, 0, kFocalLengthLowerBound);
    if (camera.has_prior_focal_length) {
      problem.SetParameterBlockConstant(focal);
    } else {
      ++num_cameras;
    }
  }

  if (num_cameras > 0) {
    ceres::Solver::Options solver_options = options.solver_options;
    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(options.solver_options.num_threads);
    solver_options.linear_solver_type = cameras.size() < 50
                                            ? ceres::DENSE_NORMAL_CHOLESKY
                                            : ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
      throw std::runtime_error("View graph calibration failed");
    }
  }

  for (auto& [camera_id, focal] : focal_lengths) {
    if (problem.HasParameterBlock(&focal)) {
      const double initial = cameras.at(camera_id).MeanFocalLength();
      const double ratio = focal / initial;
      if (ratio < options.min_focal_length_ratio ||
          ratio > options.max_focal_length_ratio) {
        focal = initial;
      }
    }
  }

  ceres::Problem::EvaluateOptions evaluate_options;
  evaluate_options.num_threads =
      colmap::GetEffectiveNumThreads(options.solver_options.num_threads);
  evaluate_options.apply_loss_function = false;
  evaluate_options.residual_blocks = fetzer_blocks;
  std::vector<double> residuals;
  if (!problem.Evaluate(
          evaluate_options, nullptr, &residuals, nullptr, nullptr)) {
    throw std::runtime_error("View graph calibration evaluation failed");
  }
  for (const auto& [camera_id, focal] : focal_lengths) {
    auto& camera = reconstruction.Camera(camera_id);
    if (!camera.has_prior_focal_length) camera.SetFocalLength(focal);
  }

  const double max_error_sq =
      options.max_calibration_error * options.max_calibration_error;
  std::size_t invalidated = 0;
  std::size_t residual_index = 0;
  for (const auto& input : inputs) {
    const double error_sq =
        residuals[residual_index] * residuals[residual_index] +
        residuals[residual_index + 1] * residuals[residual_index + 1];
    residual_index += 2;
    const PairId pair_id = input.pair_id;
    if (error_sq <= max_error_sq) continue;
    if (graph.IsValid(pair_id)) {
      graph.SetInvalidEdge(pair_id);
      ++invalidated;
    }
  }
  return invalidated;
}

}  // namespace vidmap
