#pragma once

#include <memory>
#include <stdexcept>

#include "vidmap_native/mapping_sidecars.h"
#include <ceres/loss_function.h>

namespace vidmap {

// Per-camera (focal_px, sigma_log_focal) observations.
struct LogFocalPriorRecord {
  CameraId camera_id = 0;
  Eigen::MatrixXd observations;
  std::shared_ptr<ceres::LossFunction> loss;

  void Validate() const {
    if (observations.rows() <= 0 || observations.cols() != 2 ||
        !observations.allFinite() || (observations.array() <= 0.0).any()) {
      throw std::invalid_argument("invalid log-focal prior");
    }
  }
};

// Pairwise relative focal constraint between two cameras:
// residuals[0] = ((log(f2) - log(f1)) - target_log_ratio) * (1 / sigma_log_ratio).
struct LogRelativeFocalPriorRecord {
  CameraId camera_id1 = 0;
  CameraId camera_id2 = 0;
  double target_log_ratio = 0.0;
  double sigma_log_ratio = 1.0;
  std::shared_ptr<ceres::LossFunction> loss;

  void Validate() const {
    if (camera_id1 == camera_id2 || sigma_log_ratio <= 0.0 ||
        !std::isfinite(sigma_log_ratio) || !std::isfinite(target_log_ratio)) {
      throw std::invalid_argument("invalid log-relative-focal prior");
    }
  }
};

}  // namespace vidmap

