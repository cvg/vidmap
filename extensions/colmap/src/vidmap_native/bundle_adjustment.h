#pragma once

#include <map>
#include <vector>

#include "vidmap_native/ceres_loss.h"
#include "vidmap_native/mapping_problem.h"
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

struct IntrinsicsPriorRecord {
  CameraId camera_id = 0;
  VectorXd values;
  VectorXd stddevs;

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
  SolverPlaybackOptions playback;

  void Validate() const;
};

struct BundleAdjustmentDiagnostics {
  int num_reprojection_residuals = 0;
  int num_depth_residuals = 0;
  int num_intrinsics_prior_residuals = 0;
  int num_scale_prior_residuals = 0;
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
  BundleAdjustmentDiagnostics diagnostics;
};

BundleAdjustmentResult RunBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const std::vector<DepthConstraintRecord>& depth_constraints,
    const std::vector<DepthScaleRecord>& depth_scales,
    const std::vector<IntrinsicsPriorRecord>& intrinsics_priors,
    MappingProblem* problem);

}  // namespace vidmap
