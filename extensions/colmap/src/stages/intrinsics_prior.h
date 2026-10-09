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

class LogRelativeFocalPriorCostFunction final : public ceres::CostFunction {
 public:
  LogRelativeFocalPriorCostFunction(const int num_camera_params1,
                                    std::vector<std::size_t> focal_indices1,
                                    const int num_camera_params2,
                                    std::vector<std::size_t> focal_indices2,
                                    const double target_log_ratio = 0.0,
                                    const double sigma_log_ratio = 1.0)
      : focal_indices1_(std::move(focal_indices1)),
        focal_indices2_(std::move(focal_indices2)),
        target_log_ratio_(target_log_ratio),
        inverse_sigma_log_ratio_(1.0 / sigma_log_ratio) {
    set_num_residuals(1);
    mutable_parameter_block_sizes()->push_back(num_camera_params1);
    mutable_parameter_block_sizes()->push_back(num_camera_params2);
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const double* cam1 = parameters[0];
    const double* cam2 = parameters[1];
    double mean_focal1 = 0.0;
    for (const std::size_t index : focal_indices1_) {
      mean_focal1 += cam1[index];
    }
    mean_focal1 /= static_cast<double>(focal_indices1_.size());
    if (!std::isfinite(mean_focal1) || mean_focal1 <= 0.0) return false;

    double mean_focal2 = 0.0;
    for (const std::size_t index : focal_indices2_) {
      mean_focal2 += cam2[index];
    }
    mean_focal2 /= static_cast<double>(focal_indices2_.size());
    if (!std::isfinite(mean_focal2) || mean_focal2 <= 0.0) return false;

    residuals[0] =
        ((std::log(mean_focal2) - std::log(mean_focal1)) - target_log_ratio_) *
        inverse_sigma_log_ratio_;

    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        std::fill(jacobians[0], jacobians[0] + parameter_block_sizes()[0], 0.0);
        const double derivative1 =
            -inverse_sigma_log_ratio_ / (mean_focal1 * focal_indices1_.size());
        for (const std::size_t index : focal_indices1_) {
          jacobians[0][index] = derivative1;
        }
      }
      if (jacobians[1] != nullptr) {
        std::fill(jacobians[1], jacobians[1] + parameter_block_sizes()[1], 0.0);
        const double derivative2 =
            inverse_sigma_log_ratio_ / (mean_focal2 * focal_indices2_.size());
        for (const std::size_t index : focal_indices2_) {
          jacobians[1][index] = derivative2;
        }
      }
    }
    return true;
  }

 private:
  std::vector<std::size_t> focal_indices1_;
  std::vector<std::size_t> focal_indices2_;
  double target_log_ratio_;
  double inverse_sigma_log_ratio_;
};

}  // namespace vidmap
