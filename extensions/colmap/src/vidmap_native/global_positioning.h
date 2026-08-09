#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vidmap_native/ceres_loss.h"
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
  bool filter_depth_outliers = false;
  double filter_depth_outlier_sigma = 3.0;
  std::map<ImageId, double> initial_dmap_scales;
  std::map<FrameId, Eigen::Vector3d> initial_frame_centers;

  bool use_temporal_acceleration_prior = false;
  std::vector<TemporalAccelerationPrior> temporal_acceleration_priors;
  double temporal_acceleration_prior_stddev = 1.0;
  double temporal_acceleration_prior_weight = 1.0;
  double temporal_acceleration_prior_loss_dead_zone = 0.0;
  double temporal_acceleration_prior_loss_huber_width = 1.0;

  LossConfig loss_soft_outlier_fallback = {LossFunctionType::kHuber, 1.0, 1.0};
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
  GlobalPositioningDiagnostics diagnostics;
};

GlobalPositioningResult RunGlobalPositioning(
    const GlobalPositionerOptions& options, MappingProblem* problem);

}  // namespace vidmap
