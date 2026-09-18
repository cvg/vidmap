#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <ceres/ceres.h>

namespace vidmap {

class LogMeanFocalPriorCostFunction final : public ceres::CostFunction {
 public:
  LogMeanFocalPriorCostFunction(const int num_camera_params,
                                std::vector<std::size_t> focal_indices,
                                const double focal,
                                const double sigma_log_focal)
      : focal_indices_(std::move(focal_indices)),
        log_focal_(std::log(focal)),
        inverse_sigma_log_focal_(1.0 / sigma_log_focal) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(num_camera_params);
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double* camera = parameters[0];
    double mean_focal = 0.0;
    for (const std::size_t index : focal_indices_) {
      mean_focal += camera[index];
    }
    mean_focal /= static_cast<double>(focal_indices_.size());
    if (!std::isfinite(mean_focal) || mean_focal <= 0.0) return false;
    residuals[0] =
        (std::log(mean_focal) - log_focal_) * inverse_sigma_log_focal_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      std::fill(jacobians[0], jacobians[0] + parameter_block_sizes()[0], 0.0);
      const double derivative =
          inverse_sigma_log_focal_ / (mean_focal * focal_indices_.size());
      for (const std::size_t index : focal_indices_) {
        jacobians[0][index] = derivative;
      }
    }
    return true;
  }

 private:
  std::vector<std::size_t> focal_indices_;
  double log_focal_;
  double inverse_sigma_log_focal_;
};

}  // namespace vidmap
