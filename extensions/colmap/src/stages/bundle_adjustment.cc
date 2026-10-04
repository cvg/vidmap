// Bundle adjustment over VidMap-owned value records using COLMAP and Ceres
// primitives.
#include "vidmap_native/bundle_adjustment.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/pose_prior.h"
#include "colmap/estimators/cost_functions/reprojection_error.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/imu_preintegration.h"
#include "colmap/estimators/imu_preintegration_cost.h"
#include "colmap/math/math.h"
#include "colmap/sensor/models.h"
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
#include <Eigen/Dense>
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
      const std::vector<LogFocalPriorRecord>& intrinsics_priors,
      const std::vector<ImuEdgeRecord>& imu_edges,
      const std::vector<ImuStateRecord>& imu_states,
      MappingProblem* mapping_problem)
      : options_(options),
        depth_constraints_(depth_constraints),
        depth_scale_records_(depth_scales),
        intrinsics_priors_(intrinsics_priors),
        imu_edges_(imu_edges),
        imu_states_(imu_states),
        mapping_problem_(mapping_problem) {}

  BundleAdjustmentResult Solve() {
    options_.Validate();
    mapping_problem_->Validate();
    ValidateInputs();
    SetupProblem();

    // Record input camera orientations, centers, and variable 3D points so we
    // can project out any global Sim(3) gauge drift between the free SfM poses
    // and (log_scale_, gravity_direction_) when no metric depth priors anchor
    // the SfM coordinate frame.
    std::map<ImageId, Eigen::Matrix3d> initial_rotations_wc;
    std::map<ImageId, Eigen::Vector3d> initial_centers;
    std::map<Point3DId, Eigen::Vector3d> initial_points;
    if (has_imu_ && !options_.fix_all_poses) {
      for (const ImageId image_id : imu_image_ids_) {
        initial_rotations_wc.emplace(image_id,
                                     pose_params_.at(image_id)
                                         .rotation()
                                         .conjugate()
                                         .toRotationMatrix());
        initial_centers.emplace(image_id, GetCameraCenter(image_id));
      }
      for (const Point3DId point3D_id : options_.variable_point3D_ids) {
        auto it = point_xyz_.find(point3D_id);
        if (it != point_xyz_.end()) {
          initial_points.emplace(point3D_id, it->second);
        }
      }
    }

    const bool wants_extrinsics_refinement =
        has_imu_ && (options_.refine_imu_from_cam_rotation ||
                     options_.refine_imu_from_cam_translation);
    const bool project_free_pose_scale_gauge =
        has_imu_ && !options_.fix_all_poses && options_.refine_imu_scale &&
        depth_constraints_.empty();

    ceres::Solver::Summary summary;
    try {
      summary = BuildAndSolveMainProblem(
          /*allow_extrinsics_refinement=*/wants_extrinsics_refinement);
      if (project_free_pose_scale_gauge && summary.IsSolutionUsable()) {
        ProjectFreePoseScaleGaugeToLogScale(
            initial_rotations_wc, initial_centers, initial_points);
      }
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

  Eigen::Vector3d GetCameraCenter(const ImageId image_id) const {
    return pose_params_.at(image_id).TgtOriginInSrc();
  }

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
    std::unordered_set<CameraId> focal_prior_cameras;
    for (const LogFocalPriorRecord& prior : intrinsics_priors_) {
      if (!focal_prior_cameras.insert(prior.camera_id).second) {
        throw std::invalid_argument("duplicate BA log-focal prior camera");
      }
      prior.Validate();
      mapping_problem_->Camera(prior.camera_id);
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
    if (options_.use_imu) {
      for (const ImuEdgeRecord& edge : imu_edges_) {
        edge.Validate();
        const ImageRecord& img1 = mapping_problem_->Image(edge.image_id1);
        const ImageRecord& img2 = mapping_problem_->Image(edge.image_id2);
        if (!img1.pose.has_pose || !img2.pose.has_pose) {
          throw std::invalid_argument("IMU edge references an unposed image");
        }
      }
      std::unordered_set<ImageId> seen_imu_states;
      for (const ImuStateRecord& state : imu_states_) {
        state.Validate();
        mapping_problem_->Image(state.image_id);
        if (!seen_imu_states.insert(state.image_id).second) {
          throw std::invalid_argument("duplicate BA IMU state record");
        }
      }
    }
  }

  void SetupProblem() {
    camera_params_.clear();
    for (const CameraId camera_id : mapping_problem_->CameraIds()) {
      camera_params_.emplace(camera_id,
                             mapping_problem_->Camera(camera_id).params);
    }
    pose_params_.clear();
    associations_.clear();
    for (const ImageId image_id : mapping_problem_->ImageIds()) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      colmap::Rigid3d pose = ToColmapPose(image.pose);
      pose.rotation().normalize();
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

    has_imu_ = options_.use_imu && !imu_edges_.empty();
    mutable_imu_edges_ = imu_edges_;
    imu_image_ids_.clear();
    imu_state_params_.clear();
    InitializeImuParameters();

    result_ = BundleAdjustmentResult();
  }

  ceres::Solver::Options CreateImuInitSolverOptions(
      const int max_iterations) const {
    ceres::Solver::Options solver_options;
    options_.solver_backend.Apply(&solver_options);
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.max_num_iterations = max_iterations;
    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(options_.num_threads);
    return solver_options;
  }

  void CollectImuImageIdsAndInitStates(const Eigen::Vector3d& bg_init,
                                       const Eigen::Vector3d& ba_init) {
    std::unordered_set<ImageId> edge_image_set;
    for (const ImuEdgeRecord& edge : mutable_imu_edges_) {
      edge_image_set.insert(edge.image_id1);
      edge_image_set.insert(edge.image_id2);
    }
    for (const ImageId image_id : image_order_) {
      if (edge_image_set.count(image_id) != 0) {
        imu_image_ids_.push_back(image_id);
        edge_image_set.erase(image_id);
      }
    }
    std::vector<ImageId> remaining(edge_image_set.begin(),
                                   edge_image_set.end());
    std::sort(remaining.begin(), remaining.end());
    imu_image_ids_.insert(
        imu_image_ids_.end(), remaining.begin(), remaining.end());

    for (const ImuStateRecord& state : imu_states_) {
      imu_state_params_.emplace(state.image_id, state.ToVector());
    }
    for (const ImageId image_id : imu_image_ids_) {
      if (imu_state_params_.count(image_id) == 0) {
        Eigen::Matrix<double, 9, 1> state = Eigen::Matrix<double, 9, 1>::Zero();
        state.segment<3>(3) = bg_init;
        state.segment<3>(6) = ba_init;
        imu_state_params_.emplace(image_id, state);
      }
    }
  }

  void ApplyUniformBiasesAndReintegrate(const Eigen::Vector3d& bg,
                                        const Eigen::Vector3d& ba) {
    for (const ImageId image_id : imu_image_ids_) {
      Eigen::Matrix<double, 9, 1>& state = imu_state_params_.at(image_id);
      state.segment<3>(3) = bg;
      state.segment<3>(6) = ba;
    }
    Eigen::Vector6d biases;
    biases.head<3>() = bg;
    biases.tail<3>() = ba;
    for (ImuEdgeRecord& edge : mutable_imu_edges_) {
      if (edge.integrator != nullptr) {
        edge.integrator->Reintegrate(biases);
        edge.integrator->Update(&edge.data);
      }
    }
  }

  void InitializeImuParameters() {
    log_scale_ = options_.initial_log_scale;
    gravity_direction_ = options_.initial_gravity_direction.normalized();
    if (options_.imu_from_cam.has_pose) {
      imu_from_cam_metric_ = ToColmapPose(options_.imu_from_cam);
      imu_from_cam_metric_.rotation().normalize();
    } else {
      imu_from_cam_metric_ = colmap::Rigid3d();
    }
    imu_from_cam_params_ = imu_from_cam_metric_;

    if (!has_imu_) {
      return;
    }

    gravity_magnitude_ = mutable_imu_edges_.front().data.gravity_magnitude;
    if (!std::isfinite(gravity_magnitude_) || gravity_magnitude_ <= 0.0) {
      gravity_magnitude_ = 9.81;
    }

    Eigen::Vector3d bg_init = options_.gyro_bias_prior;
    Eigen::Vector3d ba_init = options_.accel_bias_prior;
    effective_gyro_bias_prior_ = options_.gyro_bias_prior;
    effective_accel_bias_prior_ = options_.accel_bias_prior;
    CollectImuImageIdsAndInitStates(bg_init, ba_init);

    // Stage 1: Estimate initial gyroscope bias via InertialRotationCostFunctor.
    if (options_.auto_initialize_imu_states && imu_states_.empty() &&
        options_.refine_gyro_bias) {
      InitializeGyroBiasWithCeres(&bg_init, ba_init);
    }

    // Stage 2: Estimate gravity direction, metric scale, per-frame velocities,
    // and accelerometer bias via InertialGlobalPositioningCostFunctor.
    const bool init_gravity =
        options_.auto_initialize_gravity && options_.refine_gravity;
    const bool init_states =
        options_.auto_initialize_imu_states && imu_states_.empty();
    if (init_gravity || init_states) {
      InitializeScaleGravityAndStatesWithCeres(
          init_gravity, init_states, bg_init, &ba_init);
    }
  }

  void InitializeGyroBiasWithCeres(Eigen::Vector3d* bg_init,
                                   const Eigen::Vector3d& ba_init) {
    const Eigen::Quaterniond q_IC = imu_from_cam_metric_.rotation();
    std::unordered_map<ImageId, Eigen::Vector3d> aa_cw;
    aa_cw.reserve(imu_image_ids_.size());
    for (const ImageId image_id : imu_image_ids_) {
      const Eigen::AngleAxisd aa(
          pose_params_.at(image_id).rotation().normalized());
      aa_cw.emplace(image_id, aa.angle() * aa.axis());
    }

    ceres::Problem problem;
    const double visual_rot_var = options_.fix_all_poses ? 0.0 : (5e-3 * 5e-3);
    for (const ImuEdgeRecord& edge : mutable_imu_edges_) {
      Eigen::Matrix<double, 6, 6> cov_6x6 =
          colmap::ExtractRotationGyroBiasCovariance(edge.data.covariance);
      if (visual_rot_var > 0.0) {
        cov_6x6.topLeftCorner<3, 3>().diagonal().array() += visual_rot_var;
      }
      const Eigen::Matrix<double, 6, 6> sqrt_info_6x6 =
          colmap::ComputeSubBlockSqrtInformation<6>(cov_6x6);
      ceres::CostFunction* cost =
          colmap::InertialRotationCostFunctor::Create(&edge.data,
                                                      q_IC,
                                                      sqrt_info_6x6,
                                                      edge.q_iori_1_xyzw,
                                                      edge.q_iori_2_xyzw);
      problem.AddResidualBlock(cost,
                               nullptr,
                               aa_cw.at(edge.image_id1).data(),
                               imu_state_params_.at(edge.image_id1).data(),
                               aa_cw.at(edge.image_id2).data(),
                               imu_state_params_.at(edge.image_id2).data());
    }
    for (auto& [image_id, aa] : aa_cw) {
      problem.SetParameterBlockConstant(aa.data());
    }
    for (const ImageId image_id : imu_image_ids_) {
      double* state_ptr = imu_state_params_.at(image_id).data();
      if (!problem.HasParameterBlock(state_ptr)) continue;
      colmap::SetManifold(&problem,
                          state_ptr,
                          std::make_unique<DelegatingManifold>(
                              colmap::CreateImuStateGyroOnlyManifold()));
    }

    ceres::Solver::Summary summary;
    ceres::Solve(CreateImuInitSolverOptions(25), &problem, &summary);

    Eigen::Vector3d bg_mean = Eigen::Vector3d::Zero();
    for (const ImageId image_id : imu_image_ids_) {
      bg_mean += imu_state_params_.at(image_id).segment<3>(3);
    }
    bg_mean /= static_cast<double>(imu_image_ids_.size());

    if (summary.IsSolutionUsable() && bg_mean.allFinite() &&
        bg_mean.norm() < 1.0) {
      *bg_init = bg_mean;
      if (options_.gyro_bias_prior.isZero(1e-12) &&
          options_.apply_bias_prior_to_all_frames) {
        effective_gyro_bias_prior_ = *bg_init;
      }
      ApplyUniformBiasesAndReintegrate(*bg_init, ba_init);
    }
  }

  void SeedGravityDirectionAndVelocities(const bool init_gravity,
                                         const bool init_states,
                                         const Eigen::Vector3d& bg_init,
                                         const Eigen::Vector3d& ba_init) {
    const Eigen::Quaterniond q_CI = imu_from_cam_metric_.rotation().conjugate();
    Eigen::Vector3d dv_telescoping = Eigen::Vector3d::Zero();
    for (const ImuEdgeRecord& edge : mutable_imu_edges_) {
      const Eigen::Quaterniond q_wb_1 =
          pose_params_.at(edge.image_id1).rotation().conjugate() *
          edge.q_iori_1_xyzw * q_CI;
      const Eigen::Vector3d dbg = bg_init - edge.data.biases.head<3>();
      const Eigen::Vector3d dba = ba_init - edge.data.biases.tail<3>();
      const Eigen::Vector3d dv_corr =
          edge.data.delta_v + edge.data.dv_dbg * dbg + edge.data.dv_dba * dba;
      dv_telescoping += q_wb_1 * dv_corr;

      if (init_states && options_.refine_imu_velocities) {
        const double dt = std::max(edge.data.delta_t, 1e-3);
        const Eigen::Vector3d v_fd = (GetCameraCenter(edge.image_id2) -
                                      GetCameraCenter(edge.image_id1)) /
                                     dt;
        imu_state_params_.at(edge.image_id1).head<3>() = v_fd;
        imu_state_params_.at(edge.image_id2).head<3>() = v_fd;
      }
    }
    if (init_gravity && dv_telescoping.norm() > 1e-6) {
      gravity_direction_ = (-dv_telescoping).normalized();
    }
  }

  bool SolveInertialPositioningStep(
      const bool init_gravity,
      const bool init_states,
      const bool optimize_accel_bias,
      const Eigen::Vector3d& ba_prior,
      const int max_iterations,
      std::unordered_map<ImageId, Eigen::Vector3d>* centers) {
    const double visual_pos_var = options_.fix_all_poses ? 0.0 : (1e-2 * 1e-2);

    ceres::Problem problem;
    for (const ImuEdgeRecord& edge : mutable_imu_edges_) {
      const Eigen::Quaterniond q_cw_phys_1 =
          (edge.q_iori_1_xyzw.conjugate() *
           pose_params_.at(edge.image_id1).rotation())
              .normalized();
      const Eigen::Quaterniond q_cw_phys_2 =
          (edge.q_iori_2_xyzw.conjugate() *
           pose_params_.at(edge.image_id2).rotation())
              .normalized();
      Eigen::Matrix<double, 9, 9> cov_9x9 =
          colmap::ExtractPositionVelocityAccelBiasCovariance(
              edge.data.covariance);
      if (visual_pos_var > 0.0) {
        cov_9x9.topLeftCorner<3, 3>().diagonal().array() += visual_pos_var;
      }
      const Eigen::Matrix<double, 9, 9> sqrt_info_9x9 =
          colmap::ComputeSubBlockSqrtInformation<9>(cov_9x9);
      ceres::CostFunction* cost =
          colmap::InertialGlobalPositioningCostFunctor::Create(
              &edge.data,
              imu_from_cam_metric_,
              q_cw_phys_1,
              q_cw_phys_2,
              sqrt_info_9x9,
              /*metric_imu_from_cam=*/true);
      problem.AddResidualBlock(cost,
                               nullptr,
                               &log_scale_,
                               gravity_direction_.data(),
                               centers->at(edge.image_id1).data(),
                               imu_state_params_.at(edge.image_id1).data(),
                               centers->at(edge.image_id2).data(),
                               imu_state_params_.at(edge.image_id2).data());
    }

    if (!init_states || !options_.refine_imu_scale) {
      problem.SetParameterBlockConstant(&log_scale_);
    }
    gravity_direction_.normalize();
    if (!init_gravity) {
      problem.SetParameterBlockConstant(gravity_direction_.data());
    } else {
      colmap::SetManifold(&problem,
                          gravity_direction_.data(),
                          colmap::CreateSphereManifold<3>());
    }

    for (const ImageId image_id : imu_image_ids_) {
      double* c_ptr = centers->at(image_id).data();
      if (problem.HasParameterBlock(c_ptr)) {
        problem.SetParameterBlockConstant(c_ptr);
      }
    }

    std::vector<int> constant_state_indices = {3, 4, 5};
    if (!init_states || !options_.refine_imu_velocities) {
      constant_state_indices.insert(constant_state_indices.end(), {0, 1, 2});
    }
    if (!init_states || !optimize_accel_bias || !options_.refine_accel_bias) {
      constant_state_indices.insert(constant_state_indices.end(), {6, 7, 8});
    }

    for (std::size_t idx = 0; idx < imu_image_ids_.size(); ++idx) {
      const ImageId image_id = imu_image_ids_[idx];
      double* state_ptr = imu_state_params_.at(image_id).data();
      if (!problem.HasParameterBlock(state_ptr)) continue;
      if (constant_state_indices.size() == 9) {
        problem.SetParameterBlockConstant(state_ptr);
      } else {
        colmap::SetManifold(
            &problem, state_ptr, WrapSubsetManifold(9, constant_state_indices));
      }
      if (idx == 0 && optimize_accel_bias && init_states &&
          options_.refine_accel_bias) {
        problem.AddResidualBlock(
            colmap::BiasPriorCostFunctor<9>::CreateAccel(ba_prior, 0.1),
            nullptr,
            state_ptr);
      }
    }

    ceres::Solver::Summary summary;
    ceres::Solve(
        CreateImuInitSolverOptions(max_iterations), &problem, &summary);
    gravity_direction_.normalize();
    return summary.IsSolutionUsable();
  }

  void InitializeScaleGravityAndStatesWithCeres(const bool init_gravity,
                                                const bool init_states,
                                                const Eigen::Vector3d& bg_init,
                                                Eigen::Vector3d* ba_init) {
    SeedGravityDirectionAndVelocities(
        init_gravity, init_states, bg_init, *ba_init);

    std::unordered_map<ImageId, Eigen::Vector3d> centers;
    centers.reserve(imu_image_ids_.size());
    for (const ImageId image_id : imu_image_ids_) {
      centers.emplace(image_id, GetCameraCenter(image_id));
    }

    // Sub-pass 2a: Optimize scale, gravity direction, and velocities with
    // accelerometer bias held constant to avoid gravity/bias collinearity.
    SolveInertialPositioningStep(init_gravity,
                                 init_states,
                                 /*optimize_accel_bias=*/false,
                                 *ba_init,
                                 /*max_iterations=*/25,
                                 &centers);

    // Sub-pass 2b: Jointly refine accelerometer bias, scale, gravity, and
    // velocities when accelerometer bias refinement is enabled.
    if (init_states && options_.refine_accel_bias &&
        SolveInertialPositioningStep(init_gravity,
                                     init_states,
                                     /*optimize_accel_bias=*/true,
                                     *ba_init,
                                     /*max_iterations=*/25,
                                     &centers)) {
      Eigen::Vector3d ba_mean = Eigen::Vector3d::Zero();
      for (const ImageId image_id : imu_image_ids_) {
        ba_mean += imu_state_params_.at(image_id).tail<3>();
      }
      ba_mean /= static_cast<double>(imu_image_ids_.size());
      if (ba_mean.allFinite() && ba_mean.norm() < 2.0) {
        *ba_init = ba_mean;
        if (options_.accel_bias_prior.isZero(1e-12) &&
            options_.apply_bias_prior_to_all_frames) {
          effective_accel_bias_prior_ = *ba_init;
        }
        ApplyUniformBiasesAndReintegrate(bg_init, *ba_init);
      }
    }
  }

  // In free-pose VI-BA without metric depth priors, visual reprojection
  // residuals are invariant to a global Sim(3) transformation of the SfM world
  // frame, while IMU residuals constrain only the product (scale * c_sfm) and
  // (R_world * gravity_direction_). During joint optimization, part of the
  // scale or rotation update can move into the free SfM camera centers and 3D
  // points rather than staying purely in log_scale_ and gravity_direction_.
  // This helper aligns the optimized SfM trajectory back to the input SfM
  // gauge and transfers that Sim(3) delta into log_scale_ and
  // gravity_direction_.
  void ProjectFreePoseScaleGaugeToLogScale(
      const std::map<ImageId, Eigen::Matrix3d>& initial_rotations_wc,
      const std::map<ImageId, Eigen::Vector3d>& initial_centers,
      const std::map<Point3DId, Eigen::Vector3d>& initial_points) {
    if (imu_image_ids_.size() < 2) return;

    Eigen::Vector3d ref_in = Eigen::Vector3d::Zero();
    Eigen::Vector3d ref_out = Eigen::Vector3d::Zero();
    Eigen::Matrix3d M_rot = Eigen::Matrix3d::Zero();
    for (const ImageId image_id : imu_image_ids_) {
      ref_in += initial_centers.at(image_id);
      ref_out += GetCameraCenter(image_id);
      const Eigen::Matrix3d R_wc_out =
          pose_params_.at(image_id).rotation().conjugate().toRotationMatrix();
      M_rot.noalias() +=
          initial_rotations_wc.at(image_id) * R_wc_out.transpose();
    }
    const double inv_n = 1.0 / static_cast<double>(imu_image_ids_.size());
    ref_in *= inv_n;
    ref_out *= inv_n;

    Eigen::Matrix3d M_pos = Eigen::Matrix3d::Zero();
    for (const ImageId image_id : imu_image_ids_) {
      const Eigen::Vector3d c_in = initial_centers.at(image_id) - ref_in;
      const Eigen::Vector3d c_out = GetCameraCenter(image_id) - ref_out;
      M_pos.noalias() += c_in * c_out.transpose();
    }
    if (imu_image_ids_.size() < 50) {
      for (const auto& [point3D_id, xyz_in] : initial_points) {
        const Eigen::Vector3d p_in = xyz_in - ref_in;
        const Eigen::Vector3d p_out = point_xyz_.at(point3D_id) - ref_out;
        M_pos.noalias() += p_in * p_out.transpose();
      }
    }

    Eigen::Matrix3d dR = Eigen::Matrix3d::Identity();
    const Eigen::Matrix3d M_comb =
        (imu_image_ids_.size() >= 50)
            ? M_rot
            : (M_pos + std::max(1e-6, 1e-2 * M_pos.norm() * inv_n) * M_rot);
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        M_comb, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d S_sign = Eigen::Matrix3d::Identity();
    if ((U * V.transpose()).determinant() < 0.0) {
      S_sign(2, 2) = -1.0;
    }
    const Eigen::Matrix3d dR_cand = U * S_sign * V.transpose();
    if (dR_cand.allFinite()) {
      dR = dR_cand;
    }
    const Eigen::Quaterniond q_dR(dR);

    const int chord_len =
        std::clamp<int>(static_cast<int>(imu_image_ids_.size()) / 2, 1, 15);
    double dot_in_in = 0.0;
    double dot_out_in = 0.0;
    std::vector<double> chord_ratios;
    for (std::size_t idx = 0;
         idx + static_cast<std::size_t>(chord_len) < imu_image_ids_.size();
         ++idx) {
      const ImageId id1 = imu_image_ids_[idx];
      const ImageId id2 =
          imu_image_ids_[idx + static_cast<std::size_t>(chord_len)];
      const Eigen::Vector3d d_in =
          initial_centers.at(id2) - initial_centers.at(id1);
      const Eigen::Vector3d d_out = GetCameraCenter(id2) - GetCameraCenter(id1);
      const double in_sq = d_in.squaredNorm();
      const double out_dot_in = (dR * d_out).dot(d_in);
      dot_in_in += in_sq;
      dot_out_in += out_dot_in;
      if (in_sq > 1e-4 && out_dot_in > 0.0) {
        chord_ratios.push_back(out_dot_in / in_sq);
      }
    }
    if (imu_image_ids_.size() < 50) {
      for (const auto& [point3D_id, xyz_in] : initial_points) {
        const Eigen::Vector3d p_in = xyz_in - ref_in;
        const Eigen::Vector3d p_out = point_xyz_.at(point3D_id) - ref_out;
        dot_in_in += p_in.squaredNorm();
        dot_out_in += (dR * p_out).dot(p_in);
      }
    }
    if (dot_out_in <= 1e-12 || dot_in_in <= 1e-12) return;

    double delta_s = dot_out_in / dot_in_in;
    if (imu_image_ids_.size() >= 50 && !chord_ratios.empty()) {
      const std::size_t mid = chord_ratios.size() / 2;
      std::nth_element(
          chord_ratios.begin(), chord_ratios.begin() + mid, chord_ratios.end());
      delta_s = chord_ratios[mid];
    }
    if (!std::isfinite(delta_s) || delta_s <= 0.1 || delta_s >= 10.0) return;

    for (auto& [image_id, pose] : pose_params_) {
      const Eigen::Vector3d c_out = pose.TgtOriginInSrc();
      const Eigen::Quaterniond q_cw_new =
          (pose.rotation() * q_dR.conjugate()).normalized();
      const Eigen::Vector3d c_new = ref_in + (dR * (c_out - ref_out)) / delta_s;
      pose.rotation() = q_cw_new;
      pose.translation() = -(q_cw_new * c_new);
    }
    for (auto& [point3D_id, xyz] : point_xyz_) {
      xyz = ref_in + (dR * (xyz - ref_out)) / delta_s;
    }
    for (auto& [image_id, state] : imu_state_params_) {
      state.head<3>() = (dR * state.head<3>()) / delta_s;
    }
    gravity_direction_ = (dR * gravity_direction_).normalized();
    log_scale_ += std::log(delta_s);
  }

  ceres::Solver::Summary BuildAndSolveMainProblem(
      const bool allow_extrinsics_refinement) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_unique<ceres::Problem>(problem_options);
    loss_function_ = options_.reprojection_loss.Create();

    parameterized_camera_ids_.clear();
    automatically_constant_camera_ids_.clear();
    parameterized_image_ids_.clear();
    point3D_num_observations_.clear();
    owned_losses_.clear();
    result_.diagnostics = BundleAdjustmentDiagnostics();

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

    colmap::ImuReintegrationOptions reint_options;
    reint_options.reintegrate_angle_norm_thres =
        options_.reintegrate_angle_norm_thres;
    reint_options.reintegrate_vel_norm_thres =
        options_.reintegrate_vel_norm_thres;
    colmap::ImuReintegrationCallback reint_callback(reint_options);
    bool has_reint = false;
    if (has_imu_) {
      AddImuConstraints(&reint_callback, &has_reint);
    }

    ParameterizeCameras();
    ParameterizePoses();
    ParameterizePoints();
    if (has_imu_) {
      ParameterizeImuBlocks(problem_.get(),
                            allow_extrinsics_refinement,
                            /*count_diagnostics=*/true);
    }
    AddIntrinsicsPriors();
    AddDepthConstraints();

    ceres::Solver::Options solver_options;
    options_.solver_backend.Apply(&solver_options);
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.num_threads =
        colmap::GetEffectiveNumThreads(options_.num_threads);
    solver_options.max_num_iterations = options_.max_num_iterations;
    solver_options.function_tolerance = options_.function_tolerance;
    solver_options.gradient_tolerance = options_.gradient_tolerance;
    solver_options.parameter_tolerance = options_.parameter_tolerance;
    if (has_reint) {
      solver_options.callbacks.push_back(&reint_callback);
      solver_options.update_state_every_iteration = true;
    }

    ceres::Solver::Summary summary;
    SolveWithPlayback(
        options_.playback,
        solver_options,
        problem_.get(),
        [this](const char* phase, const int iteration) {
          WritePlaybackCapture(phase, iteration);
        },
        &summary);
    if (has_imu_) {
      gravity_direction_.normalize();
    }
    return summary;
  }

  void AddImuConstraints(colmap::ImuReintegrationCallback* reint_callback,
                         bool* has_reint) {
    for (ImuEdgeRecord& edge : mutable_imu_edges_) {
      ceres::CostFunction* cost =
          options_.use_analytical_imu_cost
              ? static_cast<ceres::CostFunction*>(
                    new colmap::
                        AnalyticalVisualCentricImuPreintegrationCostFunction(
                            &edge.data,
                            edge.q_iori_1_xyzw,
                            edge.q_iori_2_xyzw,
                            /*metric_imu_from_cam=*/true))
              : colmap::VisualCentricImuPreintegrationCostFunctor::Create(
                    &edge.data,
                    edge.q_iori_1_xyzw,
                    edge.q_iori_2_xyzw,
                    /*metric_imu_from_cam=*/true);
      owned_losses_.push_back(edge.loss.Create());
      problem_->AddResidualBlock(cost,
                                 owned_losses_.back().get(),
                                 &log_scale_,
                                 gravity_direction_.data(),
                                 imu_from_cam_params_.params.data(),
                                 pose_params_.at(edge.image_id1).params.data(),
                                 imu_state_params_.at(edge.image_id1).data(),
                                 pose_params_.at(edge.image_id2).params.data(),
                                 imu_state_params_.at(edge.image_id2).data());
      parameterized_image_ids_.insert(edge.image_id1);
      parameterized_image_ids_.insert(edge.image_id2);
      ++result_.diagnostics.num_imu_residuals;
      if (edge.integrator != nullptr) {
        reint_callback->AddEdge(edge.integrator,
                                &edge.data,
                                imu_state_params_.at(edge.image_id1).data());
        *has_reint = true;
      }
    }
  }

  void ParameterizeImuBlocks(ceres::Problem* problem,
                             const bool allow_extrinsics_refinement,
                             const bool count_diagnostics) {
    // 1. Scale parameter block.
    if (!options_.refine_imu_scale) {
      problem->SetParameterBlockConstant(&log_scale_);
    }

    // 2. Gravity direction on S^2.
    gravity_direction_.normalize();
    if (!options_.refine_gravity) {
      problem->SetParameterBlockConstant(gravity_direction_.data());
    } else {
      colmap::SetManifold(problem,
                          gravity_direction_.data(),
                          colmap::CreateSphereManifold<3>());
    }

    // 3. IMU-from-camera extrinsics [qx, qy, qz, qw, tx, ty, tz] in metric
    // units.
    imu_from_cam_params_.rotation().normalize();
    const bool opt_rot =
        allow_extrinsics_refinement && options_.refine_imu_from_cam_rotation;
    const bool opt_trans =
        allow_extrinsics_refinement && options_.refine_imu_from_cam_translation;
    if (!opt_rot && !opt_trans) {
      problem->SetParameterBlockConstant(imu_from_cam_params_.params.data());
    } else {
      if (opt_rot && opt_trans) {
        colmap::SetManifold(problem,
                            imu_from_cam_params_.params.data(),
                            colmap::CreateProductManifold(
                                colmap::CreateEigenQuaternionManifold(),
                                colmap::CreateEuclideanManifold<3>()));
      } else if (opt_rot) {
        colmap::SetManifold(problem,
                            imu_from_cam_params_.params.data(),
                            colmap::CreateProductManifold(
                                colmap::CreateEigenQuaternionManifold(),
                                colmap::CreateSubsetManifold(3, {0, 1, 2})));
      } else {
        colmap::SetManifold(problem,
                            imu_from_cam_params_.params.data(),
                            WrapSubsetManifold(7, {0, 1, 2, 3}));
      }

      if (options_.use_imu_from_cam_prior) {
        const double rot_std_rad = std::max(
            colmap::DegToRad(options_.imu_from_cam_rotation_prior_stddev_deg),
            1e-6);
        const double trans_std_m =
            std::max(options_.imu_from_cam_translation_prior_stddev, 1e-6);
        Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Zero();
        cov.diagonal().head<3>().setConstant(rot_std_rad * rot_std_rad);
        cov.diagonal().tail<3>().setConstant(trans_std_m * trans_std_m);

        ceres::CostFunction* prior_cost = colmap::CovarianceWeightedCostFunctor<
            colmap::AbsolutePosePriorCostFunctor>::Create(cov,
                                                          imu_from_cam_metric_);
        problem->AddResidualBlock(
            prior_cost, nullptr, imu_from_cam_params_.params.data());
        if (count_diagnostics) {
          ++result_.diagnostics.num_imu_extrinsics_prior_residuals;
        }
      }
    }

    // 4. Per-frame 9D IMU states [v(3), bg(3), ba(3)] and bias priors.
    std::vector<int> constant_state_indices;
    if (!options_.refine_imu_velocities) {
      constant_state_indices.insert(constant_state_indices.end(), {0, 1, 2});
    }
    if (!options_.refine_gyro_bias) {
      constant_state_indices.insert(constant_state_indices.end(), {3, 4, 5});
    }
    if (!options_.refine_accel_bias) {
      constant_state_indices.insert(constant_state_indices.end(), {6, 7, 8});
    }

    for (std::size_t idx = 0; idx < imu_image_ids_.size(); ++idx) {
      const ImageId image_id = imu_image_ids_[idx];
      Eigen::Matrix<double, 9, 1>& state = imu_state_params_.at(image_id);
      if (!problem->HasParameterBlock(state.data())) continue;

      if (constant_state_indices.size() == 9) {
        problem->SetParameterBlockConstant(state.data());
      } else if (!constant_state_indices.empty()) {
        colmap::SetManifold(problem,
                            state.data(),
                            WrapSubsetManifold(9, constant_state_indices));
      }

      const bool add_prior_for_frame =
          (idx == 0) || options_.apply_bias_prior_to_all_frames;
      if (add_prior_for_frame) {
        if (options_.use_gyro_bias_prior && options_.refine_gyro_bias) {
          problem->AddResidualBlock(
              colmap::BiasPriorCostFunctor<9>::CreateGyro(
                  effective_gyro_bias_prior_, options_.gyro_bias_prior_stddev),
              nullptr,
              state.data());
          if (count_diagnostics) {
            ++result_.diagnostics.num_imu_bias_prior_residuals;
          }
        }
        if (options_.use_accel_bias_prior && options_.refine_accel_bias) {
          problem->AddResidualBlock(
              colmap::BiasPriorCostFunctor<9>::CreateAccel(
                  effective_accel_bias_prior_,
                  options_.accel_bias_prior_stddev),
              nullptr,
              state.data());
          if (count_diagnostics) {
            ++result_.diagnostics.num_imu_bias_prior_residuals;
          }
        }
      }
    }
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
                                 pose_params_.at(image_id).params.data(),
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
      colmap::Rigid3d& pose = pose_params_.at(image_id);
      pose.rotation().normalize();
      if (options_.fix_all_poses) {
        problem_->SetParameterBlockConstant(pose.params.data());
      } else {
        colmap::SetManifold(problem_.get(),
                            pose.params.data(),
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
    constexpr double kFocalLengthLowerBound = 1e-3;
    for (const LogFocalPriorRecord& prior : intrinsics_priors_) {
      VectorXd& params = camera_params_.at(prior.camera_id);
      if (!problem_->HasParameterBlock(params.data())) {
        // Cameras without surviving reprojection observations stay unchanged.
        continue;
      }
      const auto model_id = static_cast<colmap::CameraModelId>(
          mapping_problem_->Camera(prior.camera_id).model_id);
      const auto focal_index_span =
          colmap::CameraModelFocalLengthIdxs(model_id);
      const std::vector<std::size_t> focal_indices(focal_index_span.begin(),
                                                   focal_index_span.end());
      for (const std::size_t index : focal_indices) {
        problem_->SetParameterLowerBound(
            params.data(), static_cast<int>(index), kFocalLengthLowerBound);
      }
      owned_losses_.push_back(prior.loss.Create());
      for (Eigen::Index row = 0; row < prior.observations.rows(); ++row) {
        problem_->AddResidualBlock(
            new LogMeanFocalPriorCostFunction(params.size(),
                                              focal_indices,
                                              prior.observations(row, 0),
                                              prior.observations(row, 1)),
            owned_losses_.back().get(),
            params.data());
        ++result_.diagnostics.num_intrinsics_prior_residuals;
      }
    }
  }

  void AddDepthConstraints() {
    for (std::size_t image_index = 0; image_index < image_order_.size();
         ++image_index) {
      const ImageId image_id = image_order_[image_index];
      colmap::Rigid3d& pose = pose_params_.at(image_id);
      if (!problem_->HasParameterBlock(pose.params.data())) continue;
      if (options_.fix_all_poses ||
          (image_index == 0 && options_.fix_first_pose)) {
        problem_->SetParameterBlockConstant(pose.params.data());
      } else if (options_.fix_rotations) {
        colmap::SetManifold(problem_.get(),
                            pose.params.data(),
                            WrapSubsetManifold(7, {0, 1, 2, 3}));
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
                                   pose.params.data(),
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
      capture.centers.row(index) =
          GetCameraCenter(playback_image_ids_[index]).transpose();
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
    result_.log_scale = log_scale_;
    result_.scale = std::exp(log_scale_);
    result_.gravity_direction = gravity_direction_.normalized();
    result_.gravity_in_world = result_.gravity_direction * gravity_magnitude_;
    colmap::Rigid3d metric_imu_from_cam = imu_from_cam_params_;
    metric_imu_from_cam.rotation().normalize();
    result_.imu_from_cam = FromColmapPose(metric_imu_from_cam);
    result_.imu_from_cam.has_pose = has_imu_ || options_.imu_from_cam.has_pose;
    result_.imu_states.clear();
    for (const auto& [image_id, state_vec] : imu_state_params_) {
      result_.imu_states.emplace(
          image_id,
          ImuStateRecord::FromVector(image_id, state_vec, result_.scale));
    }
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

    Eigen::Quaterniond q_align = Eigen::Quaterniond::Identity();
    double scale_align = 1.0;
    if (has_imu_ && options_.apply_imu_alignment_to_problem) {
      scale_align = std::exp(log_scale_);
      // Align solved gravity direction to canonical Y-down [0, 1, 0].
      q_align = Eigen::Quaterniond::FromTwoVectors(
          gravity_direction_.normalized(), Eigen::Vector3d::UnitY());
    }

    if (!options_.fix_all_poses ||
        (has_imu_ && options_.apply_imu_alignment_to_problem)) {
      for (const auto& [image_id, pose] : pose_params_) {
        ImageRecord image = mapping_problem_->Image(image_id);
        colmap::Rigid3d aligned_pose = pose;
        aligned_pose.rotation() =
            (pose.rotation() * q_align.conjugate()).normalized();
        aligned_pose.translation() = scale_align * pose.translation();
        image.pose = FromColmapPose(aligned_pose);
        mapping_problem_->UpdateImage(image);
      }
    }
    for (const auto& [point3D_id, xyz] : point_xyz_) {
      TrackRecord track = mapping_problem_->Track(point3D_id);
      track.xyz = scale_align * (q_align * xyz);
      mapping_problem_->UpdateTrack(track);
    }
  }

  const BundleAdjustmentOptions& options_;
  const std::vector<DepthConstraintRecord>& depth_constraints_;
  const std::vector<DepthScaleRecord>& depth_scale_records_;
  const std::vector<LogFocalPriorRecord>& intrinsics_priors_;
  const std::vector<ImuEdgeRecord>& imu_edges_;
  const std::vector<ImuStateRecord>& imu_states_;
  MappingProblem* mapping_problem_;

  std::vector<ImageId> image_order_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unique_ptr<ceres::LossFunction> loss_function_;
  std::vector<std::unique_ptr<ceres::LossFunction>> owned_losses_;
  std::map<CameraId, VectorXd> camera_params_;
  std::map<ImageId, colmap::Rigid3d> pose_params_;
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

  bool has_imu_ = false;
  double gravity_magnitude_ = 9.81;
  double log_scale_ = 0.0;
  Eigen::Vector3d gravity_direction_ = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d effective_gyro_bias_prior_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d effective_accel_bias_prior_ = Eigen::Vector3d::Zero();
  colmap::Rigid3d imu_from_cam_metric_;
  colmap::Rigid3d imu_from_cam_params_;
  std::vector<ImuEdgeRecord> mutable_imu_edges_;
  std::vector<ImageId> imu_image_ids_;
  std::map<ImageId, Eigen::Matrix<double, 9, 1>> imu_state_params_;

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

void ImuStateRecord::Validate() const {
  if (image_id == std::numeric_limits<ImageId>::max() ||
      !velocity.allFinite() || !metric_velocity.allFinite() ||
      !bias_gyro.allFinite() || !bias_accel.allFinite()) {
    throw std::invalid_argument("invalid BA IMU state record");
  }
}

void ImuEdgeRecord::Validate() const {
  if (image_id1 == std::numeric_limits<ImageId>::max() ||
      image_id2 == std::numeric_limits<ImageId>::max() ||
      image_id1 == image_id2) {
    throw std::invalid_argument(
        "IMU edge image IDs must be distinct and valid");
  }
  if (!std::isfinite(data.delta_t) || data.delta_t <= 0.0 ||
      !data.delta_p.allFinite() || !data.delta_v.allFinite() ||
      !data.biases.allFinite() || !data.sqrt_information.allFinite() ||
      data.sqrt_information.isZero()) {
    throw std::invalid_argument("invalid PreintegratedImuData in BA IMU edge");
  }
  if (!q_iori_1_xyzw.coeffs().allFinite() || q_iori_1_xyzw.norm() <= 1e-12 ||
      !q_iori_2_xyzw.coeffs().allFinite() || q_iori_2_xyzw.norm() <= 1e-12) {
    throw std::invalid_argument(
        "invalid stabilization quaternion in BA IMU edge");
  }
  loss.Validate();
}

void BundleAdjustmentOptions::Validate() const {
  playback.Validate();
  reprojection_loss.Validate();
  solver_backend.Validate();
  imu_from_cam.Validate();
  if (min_track_length < 0 || num_threads == 0 || max_num_iterations <= 0 ||
      !std::isfinite(function_tolerance) || function_tolerance < 0.0 ||
      !std::isfinite(gradient_tolerance) || gradient_tolerance < 0.0 ||
      !std::isfinite(parameter_tolerance) || parameter_tolerance < 0.0) {
    throw std::invalid_argument("invalid bundle adjustment options");
  }
  if (use_imu) {
    if (!std::isfinite(initial_log_scale) ||
        !initial_gravity_direction.allFinite() ||
        initial_gravity_direction.norm() <= 1e-12 ||
        !gyro_bias_prior.allFinite() ||
        !std::isfinite(gyro_bias_prior_stddev) ||
        gyro_bias_prior_stddev <= 0.0 || !accel_bias_prior.allFinite() ||
        !std::isfinite(accel_bias_prior_stddev) ||
        accel_bias_prior_stddev <= 0.0 ||
        !std::isfinite(imu_from_cam_rotation_prior_stddev_deg) ||
        imu_from_cam_rotation_prior_stddev_deg <= 0.0 ||
        !std::isfinite(imu_from_cam_translation_prior_stddev) ||
        imu_from_cam_translation_prior_stddev <= 0.0 ||
        !std::isfinite(reintegrate_angle_norm_thres) ||
        reintegrate_angle_norm_thres < 0.0 ||
        !std::isfinite(reintegrate_vel_norm_thres) ||
        reintegrate_vel_norm_thres < 0.0) {
      throw std::invalid_argument("invalid VI-BA options");
    }
  }
}

BundleAdjustmentResult RunBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const std::vector<DepthConstraintRecord>& depth_constraints,
    const std::vector<DepthScaleRecord>& depth_scales,
    const std::vector<LogFocalPriorRecord>& intrinsics_priors,
    MappingProblem* problem,
    const std::vector<ImuEdgeRecord>& imu_edges,
    const std::vector<ImuStateRecord>& imu_states) {
  if (problem == nullptr) {
    throw std::invalid_argument("mapping problem must not be null");
  }
  return DefaultBundleAdjuster(options,
                               depth_constraints,
                               depth_scales,
                               intrinsics_priors,
                               imu_edges,
                               imu_states,
                               problem)
      .Solve();
}

}  // namespace vidmap
