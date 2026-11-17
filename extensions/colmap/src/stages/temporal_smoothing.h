#pragma once

#include "colmap/estimators/cost_functions/utils.h"

#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace vidmap {

// Penalizes changes in camera velocity over three consecutive centers.
struct TemporalAccelerationCostFunctor
    : public colmap::
          AutoDiffCostFunctor<TemporalAccelerationCostFunctor, 3, 3, 3, 3> {
  TemporalAccelerationCostFunctor(double dt_prev,
                                  double dt_next,
                                  double residual_scale)
      : dt_prev_(dt_prev), dt_next_(dt_next), residual_scale_(residual_scale) {
    if (dt_prev <= 0.0 || dt_next <= 0.0 || residual_scale <= 0.0) {
      throw std::invalid_argument(
          "temporal intervals and scale must be positive");
    }
  }

  template <typename T>
  bool operator()(const T* center_prev,
                  const T* center_curr,
                  const T* center_next,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    const Vec3T prev = Eigen::Map<const Vec3T>(center_prev);
    const Vec3T curr = Eigen::Map<const Vec3T>(center_curr);
    const Vec3T next = Eigen::Map<const Vec3T>(center_next);
    const Vec3T acceleration =
        T(2.0) / (T(dt_prev_) + T(dt_next_)) *
        ((next - curr) / T(dt_next_) - (curr - prev) / T(dt_prev_));
    Eigen::Map<Vec3T> residuals_vector(residuals);
    residuals_vector = T(residual_scale_) * acceleration;
    return true;
  }

  const double dt_prev_;
  const double dt_next_;
  const double residual_scale_;
};

// BA stores rig-from-world poses rather than world-space centers.
struct TemporalPoseAccelerationCostFunctor
    : public colmap::AutoDiffCostFunctor<TemporalPoseAccelerationCostFunctor,
                                        3, 7, 7, 7> {
  TemporalPoseAccelerationCostFunctor(double dt_prev, double dt_next,
                                      double residual_scale)
      : acceleration(dt_prev, dt_next, residual_scale) {}

  template <typename T>
  bool operator()(const T* prev, const T* curr, const T* next,
                  T* residuals) const {
    const auto center = [](const T* pose) -> Eigen::Matrix<T, 3, 1> {
      return -(Eigen::Map<const Eigen::Quaternion<T>>(pose).conjugate() *
               Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pose + 4));
    };
    const auto a = center(prev), b = center(curr), c = center(next);
    return acceleration(a.data(), b.data(), c.data(), residuals);
  }
  TemporalAccelerationCostFunctor acceleration;
};

}  // namespace vidmap
