#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vidmap_native/ceres_loss.h"
#include "vidmap_native/imu_types.h"
#include "vidmap_native/mapping_problem.h"
#include "vidmap_native/solver_backend.h"
#include "vidmap_native/solver_playback.h"

namespace vidmap {

enum class MetricDepthResidualType {
  kLinear,
  kLog,
  kLogLinear,
};

enum class GlobalPositioningOrdering {
  kGrouped,
  kSingleton,
};

enum class GlobalPositioningCenterMode {
  kFrame,
  kImage,
};

struct TemporalAccelerationPrior {
  ImageId prev_image_id = 0;
  ImageId image_id = 0;
  ImageId next_image_id = 0;
  double dt_prev = 1.0;
  double dt_next = 1.0;
  double sqrt_observation_count = 1.0;
};

struct GlobalPositionerOptions {
  bool generate_random_positions = true;
  bool generate_random_points = true;
  bool generate_scales = true;
  bool initialize_warm_start_scales = true;
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;
  // Temporarily apply a separate loss to early observations of each track.
  // Zero disables warm-up. All other solver settings are inherited from GP.
  int sequential_support_warmup_rounds = 0;
  double sequential_support_max_trust_region_radius =
      ceres::Solver::Options().max_trust_region_radius;
  int sequential_support_observations_per_track = 0;
  LossConfig sequential_support_loss;
  std::vector<ImageId> sequential_support_image_timeline;
  int min_num_view_per_track = 3;
  int random_seed = -1;
  double random_init_scale = 100.0;

  LossConfig loss = {LossFunctionType::kHuber, 0.1, 1.0};
  bool apply_uncalibrated_loss_downweight = true;
  double uncalibrated_loss_downweight = 0.5;
  bool use_lc_observations = false;
  bool use_init = false;
  bool use_parameter_block_ordering = true;
  GlobalPositioningOrdering parameter_ordering =
      GlobalPositioningOrdering::kGrouped;
  GlobalPositioningCenterMode center_mode = GlobalPositioningCenterMode::kFrame;

  bool use_metric_depth_constraint = false;
  bool use_log_scale_for_depth_map_scales = false;
  MetricDepthResidualType metric_depth_residual_type =
      MetricDepthResidualType::kLinear;
  bool zero_residual_behind = false;
  double log_linear_threshold = 0.1;
  double scale_prior_stddev = 1.0;
  std::map<ImageId, double> initial_dmap_scales;
  std::map<FrameId, Eigen::Vector3d> initial_frame_centers;

  bool use_temporal_acceleration_prior = false;
  std::vector<TemporalAccelerationPrior> temporal_acceleration_priors;
  double temporal_acceleration_prior_stddev = 1.0;
  double temporal_acceleration_prior_weight = 1.0;
  double temporal_acceleration_prior_loss_dead_zone = 0.0;
  double temporal_acceleration_prior_loss_huber_width = 1.0;

  // Inertial Global Positioning (I-GP) options.
  bool use_imu = true;
  bool replace_temporal_acceleration_with_imu = true;
  // Option GP-B (true, default): unconstrained R^3 gravity warm-start before
  // SphereManifold<3> refinement. Option GP-A (false): direct single-stage
  // nonlinear solve on SphereManifold<3>.
  bool use_linear_gravity_warm_start = true;
  bool apply_imu_scale_to_problem = true;
  PoseRecord imu_from_cam;
  double gravity_magnitude = 9.81;
  Eigen::Vector3d initial_gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  double initial_scale = 1.0;
  double imu_cost_weight = 2.5e-3;
  double reintegration_bias_threshold = 1e-2;
  bool enable_low_acceleration_safeguard = true;
  double low_acceleration_min_singular_value_thres = 1e-2;
  double low_acceleration_accel_bias_prior_stddev = 1e-2;

  LossConfig loss_normal_geometry;
  LossConfig loss_normal_depth;
  LossConfig loss_lc_geometry;
  LossConfig loss_lc_depth;
  LossConfig loss_normal_geometry_inlier;
  LossConfig loss_normal_depth_inlier;
  LossConfig loss_normal_depth_outlier;
  LossConfig loss_normal_geometry_track_anchor;
  LossConfig loss_normal_depth_track_anchor;
  LossConfig loss_scale_prior;

  int num_threads = -1;
  int max_num_iterations = 100;
  double function_tolerance = 1e-5;
  double gradient_tolerance = 1e-10;
  double parameter_tolerance = 1e-8;
  SolverBackendOptions solver_backend;
  SolverPlaybackOptions playback;

  void Validate() const;
};

struct GlobalPositioningDiagnostics {
  int num_bata_residuals = 0;
  int num_metric_depth_residuals = 0;
  int num_scale_prior_residuals = 0;
  int num_temporal_acceleration_residuals = 0;
  int num_imu_residuals = 0;
  int num_imu_accel_bias_prior_residuals = 0;
  bool low_acceleration_safeguard_triggered = false;
  double observability_min_singular_value = 0.0;
  int num_regular_observations_used = 0;
  int num_loop_closure_observations_used = 0;
  int num_bata_scales = 0;
  int num_depth_map_scales = 0;
  int num_camera_centers = 0;
  int num_point3D_parameters = 0;
  int num_residual_blocks = 0;
  int num_parameter_blocks = 0;
  int num_parameters = 0;
  int num_iterations = 0;
  int termination_type = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
};

struct GlobalPositioningResult {
  bool success = false;
  std::map<ImageId, double> depth_map_scales;
  std::map<FrameId, Eigen::Vector3d> initial_frame_centers;
  std::map<Point3DId, Eigen::Vector3d> initial_point3D_xyz;
  std::map<std::string, double> initial_bata_scales;
  std::map<std::string, double> final_bata_scales;
  double log_scale = 0.0;
  double scale = 1.0;
  Eigen::Vector3d gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  Eigen::Vector3d gravity_in_world = Eigen::Vector3d(0.0, 0.0, -9.81);
  std::map<ImageId, ImuStateRecord> imu_states;
  GlobalPositioningDiagnostics diagnostics;
};

GlobalPositioningResult RunGlobalPositioning(
    const GlobalPositionerOptions& options,
    MappingProblem* problem,
    const std::vector<ImuEdgeRecord>& imu_edges = {},
    const std::vector<ImuStateRecord>& imu_states = {});

}  // namespace vidmap
