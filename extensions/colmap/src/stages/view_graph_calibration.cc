#include "colmap/estimators/cost_functions/calibration.h"
#include "colmap/util/threading.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vidmap_native/conversion.h"
#include "vidmap_native/view_graph.h"
#include <ceres/ceres.h>

namespace vidmap {
namespace {

constexpr double kFocalLengthLowerBound = 1e-3;

}  // namespace

void ViewGraphCalibrationOptions::Validate() const {
  if (min_focal_length_ratio <= 0.0 ||
      max_focal_length_ratio < min_focal_length_ratio ||
      max_calibration_error < 0.0 || loss_function_scale < 0.0 ||
      max_num_iterations <= 0 || function_tolerance < 0.0 || num_threads == 0) {
    throw std::invalid_argument("invalid focal calibration options");
  }
}

FocalLengthCalibResult CalibrateFocalLengths(
    const ViewGraphCalibrationOptions& options,
    const MappingProblem& mapping_problem) {
  options.Validate();
  mapping_problem.Validate();
  struct FocalLengthState {
    double optimized = 0.0;
    double initial = 0.0;
  };
  struct FocalLengthCalibInput {
    PairId pair_id;
    CameraId camera_id1;
    CameraId camera_id2;
    Eigen::Matrix3d F;
  };

  std::unordered_map<CameraId, colmap::Camera> cameras;
  std::unordered_map<CameraId, FocalLengthState> focal_lengths;
  cameras.reserve(mapping_problem.NumCameras());
  focal_lengths.reserve(mapping_problem.NumCameras());
  for (const CameraId camera_id : mapping_problem.CameraIds()) {
    colmap::Camera camera = ToColmapCamera(mapping_problem.Camera(camera_id));
    const double focal = camera.MeanFocalLength();
    cameras.emplace(camera_id, std::move(camera));
    focal_lengths.emplace(camera_id, FocalLengthState{focal, focal});
  }

  std::vector<FocalLengthCalibInput> inputs;
  for (const PairId pair_id : mapping_problem.PairIds()) {
    const PairRecord& pair = mapping_problem.Pair(pair_id);
    if (!pair.is_valid || !pair.geometry.has_fundamental ||
        (pair.geometry.configuration != colmap::TwoViewGeometry::CALIBRATED &&
         pair.geometry.configuration !=
             colmap::TwoViewGeometry::UNCALIBRATED)) {
      continue;
    }
    inputs.push_back({pair_id,
                      mapping_problem.Image(pair.image_id1).camera_id,
                      mapping_problem.Image(pair.image_id2).camera_id,
                      pair.geometry.fundamental});
  }

  FocalLengthCalibResult result;
  if (inputs.empty()) {
    result.success = true;
    return result;
  }

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);
  auto loss_function =
      std::make_unique<ceres::CauchyLoss>(options.loss_function_scale);
  for (const FocalLengthCalibInput& input : inputs) {
    if (input.camera_id1 == input.camera_id2) {
      problem.AddResidualBlock(
          colmap::FetzerFocalLengthSameCameraCostFunctor::Create(
              input.F, cameras.at(input.camera_id1).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths.at(input.camera_id1).optimized);
    } else {
      problem.AddResidualBlock(
          colmap::FetzerFocalLengthCostFunctor::Create(
              input.F,
              cameras.at(input.camera_id1).PrincipalPoint(),
              cameras.at(input.camera_id2).PrincipalPoint()),
          loss_function.get(),
          &focal_lengths.at(input.camera_id1).optimized,
          &focal_lengths.at(input.camera_id2).optimized);
    }
  }

  std::size_t num_cameras = 0;
  for (auto& [camera_id, camera] : cameras) {
    double* focal = &focal_lengths.at(camera_id).optimized;
    if (!problem.HasParameterBlock(focal)) continue;
    problem.SetParameterLowerBound(focal, 0, kFocalLengthLowerBound);
    if (camera.has_prior_focal_length) {
      problem.SetParameterBlockConstant(focal);
    } else {
      ++num_cameras;
    }
  }

  if (num_cameras > 0) {
    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.function_tolerance = options.function_tolerance;
    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(options.num_threads);
    solver_options.linear_solver_type = cameras.size() < 50
                                            ? ceres::DENSE_NORMAL_CHOLESKY
                                            : ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable()) {
      return result;
    }
  }

  for (auto& [camera_id, focal] : focal_lengths) {
    if (problem.HasParameterBlock(&focal.optimized)) {
      const double ratio = focal.optimized / focal.initial;
      if (ratio < options.min_focal_length_ratio ||
          ratio > options.max_focal_length_ratio) {
        focal.optimized = focal.initial;
      }
    }
    result.focal_lengths[camera_id] = focal.optimized;
  }

  ceres::Problem::EvaluateOptions evaluate_options;
  evaluate_options.num_threads =
      colmap::GetEffectiveNumThreads(options.num_threads);
  evaluate_options.apply_loss_function = false;
  std::vector<double> residuals;
  problem.Evaluate(evaluate_options, nullptr, &residuals, nullptr, nullptr);
  std::size_t residual_index = 0;
  for (const FocalLengthCalibInput& input : inputs) {
    result.calibration_errors_sq[input.pair_id] =
        residuals[residual_index] * residuals[residual_index] +
        residuals[residual_index + 1] * residuals[residual_index + 1];
    residual_index += 2;
  }
  result.success = true;
  return result;
}

std::size_t ApplyFocalCalibration(const ViewGraphCalibrationOptions& options,
                                  const FocalLengthCalibResult& result,
                                  MappingProblem* problem) {
  options.Validate();
  if (!result.success) {
    throw std::invalid_argument("cannot apply unsuccessful calibration");
  }
  for (const auto& [camera_id, focal] : result.focal_lengths) {
    CameraRecord camera = problem->Camera(camera_id);
    if (camera.has_prior_focal_length) continue;
    colmap::Camera converted = ToColmapCamera(camera);
    converted.SetFocalLength(focal);
    camera.params = Eigen::Map<const VectorXd>(converted.params.data(),
                                               converted.params.size());
    problem->UpdateCamera(camera);
  }

  const double max_error_sq =
      options.max_calibration_error * options.max_calibration_error;
  std::size_t invalidated = 0;
  for (const auto& [pair_id, error_sq] : result.calibration_errors_sq) {
    if (error_sq <= max_error_sq) continue;
    PairRecord pair = problem->Pair(pair_id);
    if (pair.is_valid) {
      pair.is_valid = false;
      problem->UpdatePair(pair);
      ++invalidated;
    }
  }
  return invalidated;
}

}  // namespace vidmap
