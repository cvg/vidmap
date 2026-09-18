#pragma once

#include <stdexcept>

#include "vidmap_native/ceres_loss.h"
#include "vidmap_native/mapping_problem.h"

namespace vidmap {

// Per-camera (focal_px, sigma_log_focal) observations.
struct LogFocalPriorRecord {
  CameraId camera_id = 0;
  Eigen::MatrixXd observations;
  LossConfig loss;

  void Validate() const {
    loss.Validate();
    if (observations.rows() <= 0 || observations.cols() != 2 ||
        !observations.allFinite() || (observations.array() <= 0.0).any()) {
      throw std::invalid_argument("invalid log-focal prior");
    }
  }
};

}  // namespace vidmap
