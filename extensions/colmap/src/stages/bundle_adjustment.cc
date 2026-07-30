// Bundle adjustment over VidMap-owned value records using COLMAP and Ceres
// primitives.
#include "vidmap_native/bundle_adjustment.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "depth_prior.h"
#include "intrinsics_prior.h"
#include "solver_playback.h"
#include "vidmap_native/conversion.h"
#include <ceres/ceres.h>

namespace vidmap {
namespace {

class DelegatingManifold final : public ceres::Manifold {
 public:
  explicit DelegatingManifold(std::unique_ptr<ceres::Manifold> manifold)
      : manifold_(std::move(manifold)) {}

  bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override {
    return manifold_->Plus(x, delta, x_plus_delta);
  }

  bool PlusJacobian(const double* x, double* jacobian) const override {
    return manifold_->PlusJacobian(x, jacobian);
  }

  bool Minus(const double* y,
             const double* x,
             double* y_minus_x) const override {
    return manifold_->Minus(y, x, y_minus_x);
  }

  bool MinusJacobian(const double* x, double* jacobian) const override {
    return manifold_->MinusJacobian(x, jacobian);
  }

  int AmbientSize() const override { return manifold_->AmbientSize(); }
  int TangentSize() const override { return manifold_->TangentSize(); }

 private:
  std::unique_ptr<ceres::Manifold> manifold_;
};

std::unique_ptr<ceres::Manifold> WrapSubsetManifold(
    int size, const std::vector<int>& constant_indices) {
  return std::make_unique<DelegatingManifold>(
      colmap::CreateSubsetManifold(size, constant_indices));
}

class DefaultBundleAdjuster {
 public:
  DefaultBundleAdjuster(
      const BundleAdjustmentOptions& options,
      const std::vector<DepthConstraintRecord>& depth_constraints,
      const std::vector<DepthScaleRecord>& depth_scales,
      const std::vector<IntrinsicsPriorRecord>& intrinsics_priors,
      MappingProblem* mapping_problem)
      : options_(options),
        depth_constraints_(depth_constraints),
        depth_scale_records_(depth_scales),
        intrinsics_priors_(intrinsics_priors),
        mapping_problem_(mapping_problem) {}

  BundleAdjustmentResult Solve() {
    options_.Validate();
    mapping_problem_->Validate();
    ValidateInputs();
    SetupProblem();
    // AddPointToProblem assumes that AddImageToProblem is called first. Do not
    // change the order of these instructions.
    const std::unordered_set<ImageId> image_ids(image_order_.begin(),
                                                image_order_.end());
    for (const ImageId image_id : image_ids) {
      AddImageToProblem(image_id);
    }
    for (const Point3DId point3D_id : options_.variable_point3D_ids) {
      AddPointToProblem(point3D_id, image_ids);
    }
    for (const Point3DId point3D_id : options_.constant_point3D_ids) {
      AddPointToProblem(point3D_id, image_ids);
    }
    ParameterizeCameras();
    ParameterizePoses();
    ParameterizePoints();
    AddIntrinsicsPriors();
    AddDepthConstraints();

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(options_.num_threads);
    solver_options.max_num_iterations = options_.max_num_iterations;
    solver_options.function_tolerance = options_.function_tolerance;
    solver_options.gradient_tolerance = options_.gradient_tolerance;
    solver_options.parameter_tolerance = options_.parameter_tolerance;
    ceres::Solver::Summary summary;
    try {
      SolveWithPlayback(
          options_.playback,
          solver_options,
          problem_.get(),
          [this](const char* phase, const int iteration) {
            WritePlaybackCapture(phase, iteration);
          },
          &summary);
    } catch (...) {
      WriteBackSolution();
      throw;
    }
    PopulateResult(summary);
    WriteBackSolution();
    return result_;
  }

 private:
  using PointAssociation = std::optional<Point3DId>;

  void ValidateInputs() {
    image_order_ = options_.image_order.empty() ? mapping_problem_->ImageIds()
                                                : options_.image_order;
    std::unordered_set<ImageId> seen_images;
    seen_images.reserve(image_order_.size());
    for (const ImageId image_id : image_order_) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      if (!seen_images.insert(image_id).second) {
        throw std::invalid_argument("duplicate image in BA image order");
      }
      if (!image.pose.has_pose) {
        throw std::invalid_argument("BA image does not have a pose");
      }
    }
    for (const DepthConstraintRecord& constraint : depth_constraints_) {
      constraint.Validate();
      mapping_problem_->Image(constraint.image_id);
      mapping_problem_->Track(constraint.point3D_id);
    }
    std::unordered_set<ImageId> depth_scale_images;
    for (const DepthScaleRecord& scale : depth_scale_records_) {
      scale.Validate();
      mapping_problem_->Image(scale.image_id);
      if (!depth_scale_images.insert(scale.image_id).second) {
        throw std::invalid_argument("duplicate BA depth scale record");
      }
    }
    for (const IntrinsicsPriorRecord& prior : intrinsics_priors_) {
      prior.Validate();
      const CameraRecord& camera = mapping_problem_->Camera(prior.camera_id);
      if (prior.values.size() != camera.params.size()) {
        throw std::invalid_argument("intrinsics prior dimension mismatch");
      }
    }
    const std::unordered_set<Point3DId> variable_point3D_ids(
        options_.variable_point3D_ids.begin(),
        options_.variable_point3D_ids.end());
    if (variable_point3D_ids.size() != options_.variable_point3D_ids.size()) {
      throw std::invalid_argument("duplicate variable BA point");
    }
    std::unordered_set<Point3DId> constant_point3D_ids;
    constant_point3D_ids.reserve(options_.constant_point3D_ids.size());
    for (const Point3DId point3D_id : options_.constant_point3D_ids) {
      if (!constant_point3D_ids.insert(point3D_id).second) {
        throw std::invalid_argument("duplicate constant BA point");
      }
      if (variable_point3D_ids.count(point3D_id) != 0) {
        throw std::invalid_argument(
            "BA point cannot be both variable and constant");
      }
    }
    for (const Point3DId point3D_id : variable_point3D_ids) {
      mapping_problem_->Track(point3D_id);
    }
    for (const Point3DId point3D_id : constant_point3D_ids) {
      mapping_problem_->Track(point3D_id);
    }
  }

  void SetupProblem() {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_unique<ceres::Problem>(problem_options);
    loss_function_ = options_.reprojection_loss.Create();

    camera_params_.clear();
    for (const CameraId camera_id : mapping_problem_->CameraIds()) {
      camera_params_.emplace(camera_id,
                             mapping_problem_->Camera(camera_id).params);
    }
    pose_params_.clear();
    associations_.clear();
    for (const ImageId image_id : mapping_problem_->ImageIds()) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      Eigen::Matrix<double, 7, 1> pose;
      pose.head<4>() = image.pose.rotation_xyzw;
      pose.tail<3>() = image.pose.translation;
      pose_params_.emplace(image_id, pose);
      associations_.emplace(
          image_id,
          std::vector<PointAssociation>(image.NumFeatures(), std::nullopt));
    }
    point_xyz_.clear();
    track_lengths_.clear();
    for (const Point3DId point3D_id : mapping_problem_->Point3DIds()) {
      const TrackRecord& track = mapping_problem_->Track(point3D_id);
      point_xyz_.emplace(point3D_id, track.xyz);
      track_lengths_.emplace(
          point3D_id, static_cast<std::size_t>(track.observations.rows()));
      for (Eigen::Index row = 0; row < track.observations.rows(); ++row) {
        const ImageId image_id = track.observations(row, 0);
        const std::uint32_t point2D_idx = track.observations(row, 1);
        auto association_it = associations_.find(image_id);
        if (association_it == associations_.end()) continue;
        if (point2D_idx >= association_it->second.size()) {
          throw std::invalid_argument("BA observation index is out of bounds");
        }
        PointAssociation& association = association_it->second[point2D_idx];
        if (association.has_value() && association.value() != point3D_id) {
          throw std::invalid_argument("feature belongs to multiple BA tracks");
        }
        association = point3D_id;
      }
    }

    depth_shift_scales_.clear();
    depth_scale_record_by_image_.clear();
    for (const DepthScaleRecord& record : depth_scale_records_) {
      depth_shift_scales_.emplace(record.image_id, record.shift_scale);
      depth_scale_record_by_image_.emplace(record.image_id, &record);
    }
    depth_constraints_by_image_.clear();
    for (const DepthConstraintRecord& constraint : depth_constraints_) {
      depth_constraints_by_image_[constraint.image_id].push_back(&constraint);
    }

    parameterized_camera_ids_.clear();
    automatically_constant_camera_ids_.clear();
    parameterized_image_ids_.clear();
    point3D_num_observations_.clear();
    owned_losses_.clear();
    result_ = BundleAdjustmentResult();
  }

  void AddImageToProblem(const ImageId image_id) {
    const ImageRecord& image = mapping_problem_->Image(image_id);
    const CameraRecord& camera = mapping_problem_->Camera(image.camera_id);
    const auto& image_associations = associations_.at(image_id);
    int num_observations = 0;
    for (std::size_t point2D_idx = 0; point2D_idx < image_associations.size();
         ++point2D_idx) {
      if (!image_associations[point2D_idx].has_value()) continue;
      const Point3DId point3D_id = image_associations[point2D_idx].value();
      const std::size_t track_length = track_lengths_.at(point3D_id);
      if (track_length <= 1) {
        throw std::invalid_argument("BA track must have at least two views");
      }
      if (options_.min_track_length > 0 &&
          static_cast<int>(track_length) < options_.min_track_length) {
        continue;
      }
      ceres::CostFunction* cost =
          colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
              static_cast<colmap::CameraModelId>(camera.model_id),
              Eigen::Vector2d(image.keypoints.row(point2D_idx)));
      problem_->AddResidualBlock(cost,
                                 loss_function_.get(),
                                 point_xyz_.at(point3D_id).data(),
                                 pose_params_.at(image_id).data(),
                                 camera_params_.at(image.camera_id).data());
      ++point3D_num_observations_[point3D_id];
      ++num_observations;
      ++result_.diagnostics.num_reprojection_residuals;
    }
    if (num_observations > 0) {
      parameterized_camera_ids_.insert(image.camera_id);
      parameterized_image_ids_.insert(image_id);
    }
  }

  void AddPointToProblem(const Point3DId point3D_id,
                         const std::unordered_set<ImageId>& problem_image_ids) {
    const TrackRecord& track = mapping_problem_->Track(point3D_id);
    if (options_.min_track_length > 0 &&
        track.observations.rows() < options_.min_track_length) {
      return;
    }
    std::size_t& num_observations = point3D_num_observations_[point3D_id];
    if (num_observations ==
        static_cast<std::size_t>(track.observations.rows())) {
      return;
    }
    for (Eigen::Index row = 0; row < track.observations.rows(); ++row) {
      const ImageId image_id = track.observations(row, 0);
      if (problem_image_ids.count(image_id) != 0) continue;
      const std::uint32_t point2D_idx = track.observations(row, 1);
      const ImageRecord& image = mapping_problem_->Image(image_id);
      if (!image.pose.has_pose) {
        throw std::invalid_argument(
            "variable BA point references an unposed image");
      }
      const CameraRecord& camera = mapping_problem_->Camera(image.camera_id);
      ceres::CostFunction* cost = colmap::CreateCameraCostFunction<
          colmap::ReprojErrorConstantPoseCostFunctor>(
          static_cast<colmap::CameraModelId>(camera.model_id),
          Eigen::Vector2d(image.keypoints.row(point2D_idx)),
          ToColmapPose(image.pose));
      problem_->AddResidualBlock(cost,
                                 loss_function_.get(),
                                 point_xyz_.at(point3D_id).data(),
                                 camera_params_.at(image.camera_id).data());
      ++num_observations;
      ++result_.diagnostics.num_reprojection_residuals;
      if (parameterized_camera_ids_.insert(image.camera_id).second) {
        automatically_constant_camera_ids_.insert(image.camera_id);
      }
    }
  }

  void ParameterizeCameras() {
    const std::unordered_set<CameraId> constant_camera_ids(
        options_.constant_camera_ids.begin(),
        options_.constant_camera_ids.end());
    const bool constant_camera = !options_.refine_focal_length &&
                                 !options_.refine_principal_point &&
                                 !options_.refine_extra_params;
    for (const CameraId camera_id : parameterized_camera_ids_) {
      VectorXd& params = camera_params_.at(camera_id);
      if (constant_camera || constant_camera_ids.count(camera_id) != 0) {
        problem_->SetParameterBlockConstant(params.data());
        continue;
      }
      if (automatically_constant_camera_ids_.count(camera_id) != 0) {
        problem_->SetParameterBlockConstant(params.data());
        continue;
      }
      const colmap::Camera camera =
          ToColmapCamera(mapping_problem_->Camera(camera_id));
      std::vector<int> constant_indices;
      if (!options_.refine_focal_length) {
        for (const std::size_t index : camera.FocalLengthIdxs()) {
          constant_indices.push_back(static_cast<int>(index));
        }
      }
      if (!options_.refine_principal_point) {
        for (const std::size_t index : camera.PrincipalPointIdxs()) {
          constant_indices.push_back(static_cast<int>(index));
        }
      }
      if (!options_.refine_extra_params) {
        for (const std::size_t index : camera.ExtraParamsIdxs()) {
          constant_indices.push_back(static_cast<int>(index));
        }
      }
      if (!constant_indices.empty()) {
        colmap::SetManifold(
            problem_.get(),
            params.data(),
            colmap::CreateSubsetManifold(params.size(), constant_indices));
      }
    }
  }

  void ParameterizePoses() {
    for (const ImageId image_id : parameterized_image_ids_) {
      Eigen::Matrix<double, 7, 1>& pose = pose_params_.at(image_id);
      Eigen::Map<Eigen::Quaterniond>(pose.data()).normalize();
      if (options_.fix_all_poses) {
        problem_->SetParameterBlockConstant(pose.data());
      } else {
        colmap::SetManifold(problem_.get(),
                            pose.data(),
                            colmap::CreateProductManifold(
                                colmap::CreateEigenQuaternionManifold(),
                                colmap::CreateEuclideanManifold<3>()));
      }
    }
  }

  void ParameterizePoints() {
    const std::unordered_set<Point3DId> variable_point3D_ids(
        options_.variable_point3D_ids.begin(),
        options_.variable_point3D_ids.end());
    for (const auto& [point3D_id, num_observations] :
         point3D_num_observations_) {
      Eigen::Vector3d& xyz = point_xyz_.at(point3D_id);
      if (!options_.refine_points3D ||
          (track_lengths_.at(point3D_id) > num_observations &&
           variable_point3D_ids.count(point3D_id) == 0)) {
        problem_->SetParameterBlockConstant(xyz.data());
      }
    }
    for (const Point3DId point3D_id : options_.constant_point3D_ids) {
      Eigen::Vector3d& xyz = point_xyz_.at(point3D_id);
      if (problem_->HasParameterBlock(xyz.data())) {
        problem_->SetParameterBlockConstant(xyz.data());
      }
    }
  }

  void AddIntrinsicsPriors() {
    for (const IntrinsicsPriorRecord& prior : intrinsics_priors_) {
      problem_->AddResidualBlock(
          new IntrinsicsPriorCostFunction(prior.values, prior.stddevs),
          nullptr,
          camera_params_.at(prior.camera_id).data());
      ++result_.diagnostics.num_intrinsics_prior_residuals;
    }
  }

  void AddDepthConstraints() {
    for (std::size_t image_index = 0; image_index < image_order_.size();
         ++image_index) {
      const ImageId image_id = image_order_[image_index];
      Eigen::Matrix<double, 7, 1>& pose = pose_params_.at(image_id);
      if (!problem_->HasParameterBlock(pose.data())) continue;
      if (options_.fix_all_poses ||
          (image_index == 0 && options_.fix_first_pose)) {
        problem_->SetParameterBlockConstant(pose.data());
      } else if (options_.fix_rotations) {
        colmap::SetManifold(
            problem_.get(), pose.data(), WrapSubsetManifold(7, {0, 1, 2, 3}));
      }

      const auto constraints_it = depth_constraints_by_image_.find(image_id);
      if (constraints_it == depth_constraints_by_image_.end() ||
          constraints_it->second.empty()) {
        continue;
      }
      const auto scale_record_it = depth_scale_record_by_image_.find(image_id);
      if (scale_record_it == depth_scale_record_by_image_.end()) {
        throw std::invalid_argument(
            "depth constraints require a shift/scale record");
      }
      const DepthScaleRecord& scale_record = *scale_record_it->second;
      Eigen::Vector2d& shift_scale = depth_shift_scales_.at(image_id);
      for (const DepthConstraintRecord* constraint : constraints_it->second) {
        ceres::CostFunction* cost =
            options_.use_log_depth_residual
                ? LogScaledDepthErrorCostFunctor::Create(constraint->depth)
                : ScaledDepthErrorCostFunctor::Create(constraint->depth);
        owned_losses_.push_back(constraint->loss.Create());
        problem_->AddResidualBlock(cost,
                                   owned_losses_.back().get(),
                                   pose.data(),
                                   point_xyz_.at(constraint->point3D_id).data(),
                                   shift_scale.data());
        ++result_.diagnostics.num_depth_residuals;
      }
      SetDepthScaleManifold(scale_record, &shift_scale);

      if (scale_record.use_scale_prior) {
        owned_losses_.push_back(scale_record.scale_prior_loss.Create());
        problem_->AddResidualBlock(
            new ScalePriorCostFunction(1.0 / scale_record.scale_prior_stddev),
            owned_losses_.back().get(),
            shift_scale.data());
        ++result_.diagnostics.num_scale_prior_residuals;
      }
      SetDepthScaleManifold(scale_record, &shift_scale);
    }
  }

  void SetDepthScaleManifold(const DepthScaleRecord& record,
                             Eigen::Vector2d* shift_scale) {
    if (record.fix_shift && record.fix_scale) {
      problem_->SetParameterBlockConstant(shift_scale->data());
      return;
    }
    std::vector<int> fixed_indices;
    if (record.fix_shift) fixed_indices.push_back(0);
    if (record.fix_scale) fixed_indices.push_back(1);
    if (!fixed_indices.empty()) {
      colmap::SetManifold(problem_.get(),
                          shift_scale->data(),
                          WrapSubsetManifold(2, fixed_indices));
    }
  }

  void PreparePlaybackSelection() {
    if (playback_selection_ready_) return;

    playback_image_ids_ = options_.playback.image_ids;
    if (playback_image_ids_.empty()) {
      for (const ImageId image_id : mapping_problem_->ImageIds()) {
        if (mapping_problem_->Image(image_id).pose.has_pose) {
          playback_image_ids_.push_back(image_id);
        }
      }
    }
    for (const ImageId image_id : playback_image_ids_) {
      if (pose_params_.count(image_id) == 0 ||
          !mapping_problem_->Image(image_id).pose.has_pose) {
        throw std::invalid_argument(
            "playback image is not available in bundle adjustment");
      }
    }

    playback_point3D_ids_ = options_.playback.point3D_ids;
    if (playback_point3D_ids_.empty()) {
      for (const auto& [point3D_id, xyz] : point_xyz_) {
        playback_point3D_ids_.push_back(point3D_id);
      }
      playback_point3D_ids_ =
          SelectPlaybackPoints(std::move(playback_point3D_ids_));
    }
    for (const Point3DId point3D_id : playback_point3D_ids_) {
      if (point_xyz_.count(point3D_id) == 0) {
        throw std::invalid_argument(
            "playback point is not available in bundle adjustment");
      }
    }
    playback_selection_ready_ = true;
  }

  void WritePlaybackCapture(const char* phase, const int iteration) {
    PreparePlaybackSelection();
    SolverPlaybackCapture capture;
    capture.phase = phase;
    capture.iteration = iteration;
    capture.image_ids = playback_image_ids_;
    capture.centers.resize(playback_image_ids_.size(), 3);
    for (std::size_t index = 0; index < playback_image_ids_.size(); ++index) {
      const auto& pose = pose_params_.at(playback_image_ids_[index]);
      const Eigen::Map<const Eigen::Quaterniond> rotation(pose.data());
      capture.centers.row(index) =
          (rotation.conjugate() * -pose.tail<3>()).transpose();
    }
    capture.point3D_ids = playback_point3D_ids_;
    capture.points_xyz.resize(playback_point3D_ids_.size(), 3);
    for (std::size_t index = 0; index < playback_point3D_ids_.size(); ++index) {
      capture.points_xyz.row(index) =
          point_xyz_.at(playback_point3D_ids_[index]).transpose();
    }
    capture.loop_closure_pairs.resize(0, 2);
    capture.loop_closure_raw_scores.resize(0);
    options_.playback.callback(capture);
  }

  void PopulateResult(const ceres::Solver::Summary& summary) {
    result_.success = summary.IsSolutionUsable();
    result_.depth_shift_scales = depth_shift_scales_;
    BundleAdjustmentDiagnostics& diagnostics = result_.diagnostics;
    diagnostics.num_residual_blocks = summary.num_residual_blocks;
    diagnostics.num_parameter_blocks = summary.num_parameter_blocks;
    diagnostics.num_parameters = summary.num_parameters;
    diagnostics.num_iterations = static_cast<int>(summary.iterations.size());
    diagnostics.termination_type = static_cast<int>(summary.termination_type);
    diagnostics.initial_cost = summary.initial_cost;
    diagnostics.final_cost = summary.final_cost;
  }

  void WriteBackSolution() {
    for (const auto& [camera_id, params] : camera_params_) {
      CameraRecord camera = mapping_problem_->Camera(camera_id);
      camera.params = params;
      mapping_problem_->UpdateCamera(camera);
    }
    if (!options_.fix_all_poses) {
      for (const auto& [image_id, pose] : pose_params_) {
        ImageRecord image = mapping_problem_->Image(image_id);
        image.pose.rotation_xyzw = pose.head<4>();
        image.pose.translation = pose.tail<3>();
        mapping_problem_->UpdateImage(image);
      }
    }
    for (const auto& [point3D_id, xyz] : point_xyz_) {
      TrackRecord track = mapping_problem_->Track(point3D_id);
      track.xyz = xyz;
      mapping_problem_->UpdateTrack(track);
    }
  }

  const BundleAdjustmentOptions& options_;
  const std::vector<DepthConstraintRecord>& depth_constraints_;
  const std::vector<DepthScaleRecord>& depth_scale_records_;
  const std::vector<IntrinsicsPriorRecord>& intrinsics_priors_;
  MappingProblem* mapping_problem_;

  std::vector<ImageId> image_order_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::vector<std::unique_ptr<ceres::LossFunction>> owned_losses_;
  std::map<CameraId, VectorXd> camera_params_;
  std::map<ImageId, Eigen::Matrix<double, 7, 1>> pose_params_;
  std::map<Point3DId, Eigen::Vector3d> point_xyz_;
  std::map<ImageId, std::vector<PointAssociation>> associations_;
  std::map<Point3DId, std::size_t> track_lengths_;
  std::unordered_map<Point3DId, std::size_t> point3D_num_observations_;
  std::set<CameraId> parameterized_camera_ids_;
  std::set<CameraId> automatically_constant_camera_ids_;
  std::set<ImageId> parameterized_image_ids_;
  std::map<ImageId, Eigen::Vector2d> depth_shift_scales_;
  std::unordered_map<ImageId, const DepthScaleRecord*>
      depth_scale_record_by_image_;
  std::unordered_map<ImageId, std::vector<const DepthConstraintRecord*>>
      depth_constraints_by_image_;
  std::vector<ImageId> playback_image_ids_;
  std::vector<Point3DId> playback_point3D_ids_;
  bool playback_selection_ready_ = false;
  BundleAdjustmentResult result_;
};

}  // namespace

void DepthConstraintRecord::Validate() const {
  if (!std::isfinite(depth) || depth <= 0.0) {
    throw std::invalid_argument("invalid BA depth constraint");
  }
  loss.Validate();
}

void DepthScaleRecord::Validate() const {
  if (!shift_scale.allFinite() || !std::isfinite(scale_prior_stddev) ||
      scale_prior_stddev <= 0.0) {
    throw std::invalid_argument("invalid BA depth scale record");
  }
  scale_prior_loss.Validate();
}

void IntrinsicsPriorRecord::Validate() const {
  if (values.size() == 0 || values.size() != stddevs.size() ||
      !values.allFinite() || !stddevs.allFinite() ||
      (stddevs.array() <= 0.0).any()) {
    throw std::invalid_argument("invalid BA intrinsics prior");
  }
}

void BundleAdjustmentOptions::Validate() const {
  playback.Validate();
  reprojection_loss.Validate();
  if (min_track_length < 0 || num_threads == 0 || max_num_iterations <= 0 ||
      !std::isfinite(function_tolerance) || function_tolerance < 0.0 ||
      !std::isfinite(gradient_tolerance) || gradient_tolerance < 0.0 ||
      !std::isfinite(parameter_tolerance) || parameter_tolerance < 0.0) {
    throw std::invalid_argument("invalid bundle adjustment options");
  }
}

BundleAdjustmentResult RunBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const std::vector<DepthConstraintRecord>& depth_constraints,
    const std::vector<DepthScaleRecord>& depth_scales,
    const std::vector<IntrinsicsPriorRecord>& intrinsics_priors,
    MappingProblem* problem) {
  if (problem == nullptr) {
    throw std::invalid_argument("mapping problem must not be null");
  }
  return DefaultBundleAdjuster(options,
                               depth_constraints,
                               depth_scales,
                               intrinsics_priors,
                               problem)
      .Solve();
}

}  // namespace vidmap
