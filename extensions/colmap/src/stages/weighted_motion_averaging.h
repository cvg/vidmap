#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace vidmap {

struct WeightedBATAPairwiseDirectionCostFunctor {
  WeightedBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& pos2_from_pos1_dir,
      const Eigen::Quaterniond& rotation,
      double sigma_x,
      double sigma_y,
      double sigma_z)
      : pos2_from_pos1_dir_(pos2_from_pos1_dir),
        rotation_(rotation),
        inv_sigma_x_(1.0 / sigma_x),
        inv_sigma_y_(1.0 / sigma_y),
        inv_sigma_z_(1.0 / sigma_z) {}

  template <typename T>
  bool operator()(const T* pos1,
                  const T* pos2,
                  const T* scale,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    const Vec3T r_world = pos2_from_pos1_dir_.cast<T>() -
                          scale[0] * (Eigen::Map<const Vec3T>(pos2) -
                                      Eigen::Map<const Vec3T>(pos1));
    const Vec3T r_cam = rotation_.cast<T>() * r_world;
    residuals[0] = T(inv_sigma_x_) * r_cam[0];
    residuals[1] = T(inv_sigma_y_) * r_cam[1];
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& pos2_from_pos1_dir,
                                     const Eigen::Quaterniond& rotation,
                                     double sigma_x,
                                     double sigma_y,
                                     double sigma_z) {
    return new ceres::AutoDiffCostFunction<
        WeightedBATAPairwiseDirectionCostFunctor,
        3,
        3,
        3,
        1>(new WeightedBATAPairwiseDirectionCostFunctor(
        pos2_from_pos1_dir, rotation, sigma_x, sigma_y, sigma_z));
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
  const Eigen::Quaterniond rotation_;
  const double inv_sigma_x_;
  const double inv_sigma_y_;
  const double inv_sigma_z_;
};

// Penalizes changes in camera velocity over three consecutive centers.
struct TemporalAccelerationCostFunctor {
  TemporalAccelerationCostFunctor(double dt_prev,
                                  double dt_next,
                                  double residual_scale)
      : dt_prev_(dt_prev), dt_next_(dt_next), residual_scale_(residual_scale) {}

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

  static ceres::CostFunction* Create(double dt_prev,
                                     double dt_next,
                                     double residual_scale) {
    if (dt_prev <= 0.0 || dt_next <= 0.0 || residual_scale <= 0.0) {
      return nullptr;
    }
    return new ceres::
        AutoDiffCostFunction<TemporalAccelerationCostFunctor, 3, 3, 3, 3>(
            new TemporalAccelerationCostFunctor(
                dt_prev, dt_next, residual_scale));
  }

  const double dt_prev_;
  const double dt_next_;
  const double residual_scale_;
};

}  // namespace vidmap
