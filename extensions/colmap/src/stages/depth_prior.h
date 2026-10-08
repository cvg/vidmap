#pragma once

#include "colmap/estimators/cost_functions/utils.h"

#include <stdexcept>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace vidmap {

struct ScaledDepthErrorCostFunctor
    : public colmap::
          AutoDiffCostFunctor<ScaledDepthErrorCostFunctor, 1, 7, 3, 2> {
  explicit ScaledDepthErrorCostFunctor(double depth) : depth_(depth) {}

  template <typename T>
  bool operator()(const T* pose,
                  const T* point3D,
                  const T* shift_scale,
                  T* residuals) const {
    residuals[0] = (colmap::EigenQuaternionMap<T>(pose) *
                    colmap::EigenVector3Map<T>(point3D))[2] +
                   pose[6] - shift_scale[0] -
                   T(depth_) * ceres::exp(shift_scale[1]);
    return true;
  }

 private:
  const double depth_;
};

struct LogScaledDepthErrorCostFunctor
    : public colmap::
          AutoDiffCostFunctor<LogScaledDepthErrorCostFunctor, 1, 7, 3, 2> {
  explicit LogScaledDepthErrorCostFunctor(double depth) : depth_(depth) {
    if (depth <= 0.0) {
      throw std::invalid_argument("log-depth constraint must be positive");
    }
  }

  template <typename T>
  bool operator()(const T* pose,
                  const T* point3D,
                  const T* shift_scale,
                  T* residuals) const {
    const T predicted_depth = (colmap::EigenQuaternionMap<T>(pose) *
                               colmap::EigenVector3Map<T>(point3D))[2] +
                              pose[6];
    if (predicted_depth <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }
    residuals[0] =
        ceres::log(predicted_depth) - (ceres::log(T(depth_)) + shift_scale[1]);
    return true;
  }

 private:
  const double depth_;
};

class ScalePriorCostFunction : public ceres::SizedCostFunction<1, 2> {
 public:
  explicit ScalePriorCostFunction(double inverse_stddev,
                                  double target_log_scale = 0.0)
      : inverse_stddev_(inverse_stddev), target_log_scale_(target_log_scale) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const override {
    residuals[0] = inverse_stddev_ * (parameters[0][1] - target_log_scale_);
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = 0.0;
      jacobians[0][1] = inverse_stddev_;
    }
    return true;
  }

 private:
  const double inverse_stddev_;
  const double target_log_scale_;
};

}  // namespace vidmap
