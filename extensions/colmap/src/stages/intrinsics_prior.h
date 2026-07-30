#pragma once

#include <algorithm>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace vidmap {

class IntrinsicsPriorCostFunction : public ceres::CostFunction {
 public:
  IntrinsicsPriorCostFunction(const Eigen::VectorXd& values,
                              const Eigen::VectorXd& stddevs)
      : values_(values), inverse_stddevs_(stddevs.cwiseInverse()) {
    set_num_residuals(values.size());
    mutable_parameter_block_sizes()->push_back(values.size());
  }

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    const int dimension = static_cast<int>(values_.size());
    for (int index = 0; index < dimension; ++index) {
      residuals[index] =
          (parameters[0][index] - values_[index]) * inverse_stddevs_[index];
    }
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      std::fill(jacobians[0], jacobians[0] + dimension * dimension, 0.0);
      for (int index = 0; index < dimension; ++index) {
        jacobians[0][index * dimension + index] = inverse_stddevs_[index];
      }
    }
    return true;
  }

 private:
  const Eigen::VectorXd values_;
  const Eigen::VectorXd inverse_stddevs_;
};

}  // namespace vidmap
