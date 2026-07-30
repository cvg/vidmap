#pragma once

#include <cstddef>
#include <map>

#include "vidmap_native/mapping_problem.h"

namespace vidmap {

struct InlierThresholdOptions {
  double max_epipolar_error_essential = 1.0;
  double max_epipolar_error_fundamental = 4.0;
  double max_epipolar_error_homography = 4.0;
  double min_angle_from_epipole_deg = 3.0;

  void Validate() const;
};

struct ViewGraphCalibrationOptions {
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_calibration_error = 2.0;
  double loss_function_scale = 0.01;
  int num_threads = -1;
  int max_num_iterations = 100;
  double function_tolerance = 1e-5;

  void Validate() const;
};

struct FocalLengthCalibResult {
  bool success = false;
  std::map<CameraId, double> focal_lengths;
  std::map<PairId, double> calibration_errors_sq;
};

void PrepareImageBearings(MappingProblem* problem);
void UpdateImagePairsConfig(MappingProblem* problem);
std::size_t DecomposeRelPose(MappingProblem* problem);
void ImagePairsInlierCount(const InlierThresholdOptions& options,
                           bool clean_inliers,
                           MappingProblem* problem);
std::size_t FilterPairsByInlierNum(int min_inlier_count,
                                   MappingProblem* problem);
std::size_t FilterPairsByInlierRatio(double min_inlier_ratio,
                                     MappingProblem* problem);
FocalLengthCalibResult CalibrateFocalLengths(
    const ViewGraphCalibrationOptions& options, const MappingProblem& problem);
std::size_t ApplyFocalCalibration(const ViewGraphCalibrationOptions& options,
                                  const FocalLengthCalibResult& result,
                                  MappingProblem* problem);

}  // namespace vidmap
