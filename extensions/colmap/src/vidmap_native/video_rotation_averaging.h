#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include "vidmap_native/imu_types.h"
#include "vidmap_native/mapping_problem.h"

namespace vidmap {

struct VideoRotationAveragingOptions {
  int random_seed = 1;
  int image_order_passes = 1;
  bool filter_unregistered = true;
  bool skip_risky_lc_pairs = false;
  double max_rotation_error_deg = 0.0;
  double video_tracking_huber_scale = 0.1;
  double video_lc_cauchy_scale = 0.05;
  // One thread keeps rotation averaging byte-identical across runs; -1, or any
  // count above one, trades that determinism for Ceres parallelism.
  int num_threads = 1;
  int max_num_iterations = 100;

  // Inertial Rotation Averaging (I-RA) options.
  bool use_imu = false;
  PoseRecord imu_from_cam;
  bool refine_gyro_bias = true;
  bool auto_initialize_gyro_bias = true;
  double visual_rotation_stddev_deg = 0.2;
  double imu_tracking_cauchy_scale_deg = 0.75;
  double reintegrate_angle_norm_thres = 1e-4;
  bool use_dynamic_imu_rotation_threshold = true;
  double imu_dynamic_rotation_threshold_multiplier = 3.5;
  double imu_gyro_bias_stddev_rad_s = 0.002;
  bool invalidate_outlier_pairs = false;
  // Option RP-B: when invalidate_outlier_pairs is true, attempt to salvage
  // rotation-rejected pairs via known-rotation 2-point translation RANSAC on
  // static background matches before invalidating the pair.
  bool salvage_outlier_translations = false;
  // Only salvage with matches that the rejected visual relative pose does not
  // explain (within salvage_epipolar_angle_thres_deg), i.e. require a second
  // motion. Pairs whose matches mostly follow a moving object stay rejected,
  // while pairs with a wrong model but static matches are still salvaged.
  bool salvage_require_second_motion = true;
  double salvage_epipolar_angle_thres_deg = 0.4;
  double salvage_min_inlier_ratio = 0.30;
  int salvage_min_inliers = 50;

  void Validate() const;
};

struct RotationAveragingResult {
  bool success = false;
  std::vector<ImageId> registered_image_ids;
  std::vector<PairId> outlier_pair_ids;
  std::vector<PairId> salvaged_pair_ids;
  Eigen::Vector3d initial_gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d initial_gravity_direction = Eigen::Vector3d(0.0, 0.0, -1.0);
  std::map<ImageId, ImuStateRecord> imu_states;
};

RotationAveragingResult RunVideoRotationAveraging(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImageId>& image_map_order,
    const std::vector<PairId>& pair_map_order,
    MappingProblem* problem,
    const std::vector<ImuEdgeRecord>& imu_edges = {},
    const std::vector<ImuStateRecord>& imu_states = {});

}  // namespace vidmap
