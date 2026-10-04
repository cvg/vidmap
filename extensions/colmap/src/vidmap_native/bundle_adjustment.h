#pragma once

#include "colmap/estimators/imu_preintegration.h"

#include <map>
#include <vector>

#include "vidmap_native/focal_prior.h"
#include "vidmap_native/mapping_problem.h"
#include "vidmap_native/solver_backend.h"
#include "vidmap_native/solver_playback.h"

namespace vidmap {

struct DepthConstraintRecord {
  ImageId image_id = 0;
  Point3DId point3D_id = 0;
  double depth = 0.0;
  LossConfig loss;

  void Validate() const;
};

struct DepthScaleRecord {
  ImageId image_id = 0;
  Eigen::Vector2d shift_scale = Eigen::Vector2d::Zero();
  bool fix_shift = true;
  bool fix_scale = false;
  bool use_scale_prior = false;
  double scale_prior_stddev = 1.0;
  LossConfig scale_prior_loss;

  void Validate() const;
};

struct ImuStateRecord {
  ImageId image_id = 0;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d metric_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 9, 1> ToVector() const {
    Eigen::Matrix<double, 9, 1> vec;
    vec.segment<3>(0) = velocity;
    vec.segment<3>(3) = bias_gyro;
    vec.segment<3>(6) = bias_accel;
    return vec;
  }

  static ImuStateRecord FromVector(ImageId image_id,
                                   const Eigen::Matrix<double, 9, 1>& vec,
                                   double scale = 1.0) {
    ImuStateRecord record;
    record.image_id = image_id;
    record.velocity = vec.segment<3>(0);
    record.metric_velocity = record.velocity * scale;
    record.bias_gyro = vec.segment<3>(3);
    record.bias_accel = vec.segment<3>(6);
    return record;
  }

  void Validate() const;
};

struct ImuEdgeRecord {
  ImageId image_id1 = 0;
  ImageId image_id2 = 0;
  colmap::PreintegratedImuData data;
  colmap::ImuPreintegrator* integrator = nullptr;
  Eigen::Quaterniond q_iori_1_xyzw = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_iori_2_xyzw = Eigen::Quaterniond::Identity();
  LossConfig loss;

  void Validate() const;
};

struct BundleAdjustmentOptions {
  std::vector<ImageId> image_order;
  std::vector<CameraId> constant_camera_ids;
  std::vector<Point3DId> variable_point3D_ids;
  std::vector<Point3DId> constant_point3D_ids;
  LossConfig reprojection_loss;
  bool refine_focal_length = true;
  bool refine_principal_point = false;
  bool refine_extra_params = true;
  bool refine_points3D = true;
  int min_track_length = 0;
  bool fix_first_pose = true;
  bool fix_rotations = false;
  bool fix_all_poses = false;
  bool use_log_depth_residual = true;
  int num_threads = 1;
  int max_num_iterations = 50;
  double function_tolerance = 1e-6;
  double gradient_tolerance = 1e-10;
  double parameter_tolerance = 1e-8;
  SolverBackendOptions solver_backend;
  SolverPlaybackOptions playback;

  // Visual-Inertial Bundle Adjustment (VI-BA) options.
  bool use_imu = false;
  bool use_analytical_imu_cost = true;
  bool refine_imu_scale = true;
  bool refine_gravity = true;
  bool refine_imu_velocities = true;
  bool refine_gyro_bias = true;
  bool refine_accel_bias = true;
  bool refine_imu_from_cam_rotation = false;
  bool refine_imu_from_cam_translation = false;
  bool auto_initialize_gravity = true;
  bool auto_initialize_imu_states = true;
  bool imu_warm_start = true;
  bool apply_imu_alignment_to_problem = false;

  double initial_log_scale = 0.0;
  Eigen::Vector3d initial_gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  PoseRecord imu_from_cam;

  bool use_gyro_bias_prior = true;
  Eigen::Vector3d gyro_bias_prior = Eigen::Vector3d::Zero();
  double gyro_bias_prior_stddev = 0.1;
  bool use_accel_bias_prior = true;
  Eigen::Vector3d accel_bias_prior = Eigen::Vector3d::Zero();
  double accel_bias_prior_stddev = 0.5;
  bool apply_bias_prior_to_all_frames = false;

  bool use_imu_from_cam_prior = true;
  double imu_from_cam_rotation_prior_stddev_deg = 0.5;
  double imu_from_cam_translation_prior_stddev = 0.02;

  double reintegrate_angle_norm_thres = 1e-4;
  double reintegrate_vel_norm_thres = 1e-4;

  void Validate() const;
};

struct BundleAdjustmentDiagnostics {
  int num_reprojection_residuals = 0;
  int num_depth_residuals = 0;
  int num_intrinsics_prior_residuals = 0;
  int num_scale_prior_residuals = 0;
  int num_imu_residuals = 0;
  int num_imu_bias_prior_residuals = 0;
  int num_imu_extrinsics_prior_residuals = 0;
  int num_residual_blocks = 0;
  int num_parameter_blocks = 0;
  int num_parameters = 0;
  int num_iterations = 0;
  int termination_type = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
};

struct BundleAdjustmentResult {
  bool success = false;
  std::map<ImageId, Eigen::Vector2d> depth_shift_scales;
  double log_scale = 0.0;
  double scale = 1.0;
  Eigen::Vector3d gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d gravity_in_world = Eigen::Vector3d(0.0, 0.0, -9.81);
  PoseRecord imu_from_cam;
  std::map<ImageId, ImuStateRecord> imu_states;
  BundleAdjustmentDiagnostics diagnostics;
};

BundleAdjustmentResult RunBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const std::vector<DepthConstraintRecord>& depth_constraints,
    const std::vector<DepthScaleRecord>& depth_scales,
    const std::vector<LogFocalPriorRecord>& intrinsics_priors,
    MappingProblem* problem,
    const std::vector<ImuEdgeRecord>& imu_edges = {},
    const std::vector<ImuStateRecord>& imu_states = {});

}  // namespace vidmap
