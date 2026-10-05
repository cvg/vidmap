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

}  // namespace vidmap
