#pragma once

#include <algorithm>
#include <stdexcept>

#include "vidmap_native/global_positioning.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace vidmap {

struct MetricDepthError {
  MetricDepthError(const Eigen::Quaterniond& rotation,
                   double depth_prior,
                   double sigma_depth,
                   bool use_log_scale,
                   MetricDepthResidualType residual_type,
                   bool zero_residual_behind,
                   double log_linear_threshold)
      : rotation_(rotation),
        depth_prior_(depth_prior),
        sigma_depth_(sigma_depth),
        use_log_scale_(use_log_scale),
        residual_type_(residual_type),
        zero_residual_behind_(zero_residual_behind),
        log_linear_threshold_(log_linear_threshold) {
    if (sigma_depth <= 1e-9) {
      throw std::invalid_argument("depth standard deviation must be positive");
    }
    if (residual_type == MetricDepthResidualType::kLogLinear &&
        log_linear_threshold <= 0.0) {
      throw std::invalid_argument("log-linear threshold must be positive");
    }
  }

  template <typename T>
  bool operator()(const T* camera_center,
                  const T* point3D,
                  const T* depth_map_scale,
                  T* residuals) const {
    using Vector3 = Eigen::Matrix<T, 3, 1>;
    const Vector3 point_world = Eigen::Map<const Vector3>(point3D);
    const Vector3 center_world = Eigen::Map<const Vector3>(camera_center);
    const T estimated_depth =
        (rotation_.cast<T>() * (point_world - center_world))[2];
    const T scale =
        use_log_scale_ ? ceres::exp(depth_map_scale[0]) : depth_map_scale[0];
    const T scaled_prior = scale * T(depth_prior_);
    const T scaled_stddev = scale * T(sigma_depth_);

    if (zero_residual_behind_ && estimated_depth <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }

    T depth_residual;
    T weight;
    if (residual_type_ != MetricDepthResidualType::kLinear) {
      const T safe_prior = std::max(T(depth_prior_), T(1e-6));
      const T log_stddev = T(sigma_depth_) / safe_prior;
      const T log_weight = T(1.0) / std::max(T(1e-6), log_stddev);
      if (residual_type_ == MetricDepthResidualType::kLogLinear) {
        const T threshold = T(log_linear_threshold_);
        const T safe_scaled_prior = std::max(scaled_prior, T(1e-6));
        if (estimated_depth > threshold) {
          depth_residual = ceres::log(std::max(estimated_depth, T(1e-6)) /
                                      safe_scaled_prior);
        } else {
          depth_residual = ceres::log(threshold / safe_scaled_prior) +
                           (estimated_depth - threshold);
        }
        weight = log_weight;
      } else if (estimated_depth > T(0.0)) {
        depth_residual = ceres::log(std::max(estimated_depth, T(1e-6)) /
                                    std::max(scaled_prior, T(1e-6)));
        weight = log_weight;
      } else {
        depth_residual = estimated_depth - scaled_prior;
        weight = T(1.0) / std::max(T(1e-6), scaled_stddev);
      }
    } else {
      depth_residual = estimated_depth - scaled_prior;
      weight = T(1.0) / std::max(T(1e-6), scaled_stddev);
    }
    residuals[0] = weight * depth_residual;
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& rotation,
                                     double depth_prior,
                                     double sigma_depth,
                                     bool use_log_scale,
                                     MetricDepthResidualType residual_type,
                                     bool zero_residual_behind,
                                     double log_linear_threshold) {
    if (sigma_depth <= 1e-9) return nullptr;
    return new ceres::AutoDiffCostFunction<MetricDepthError, 1, 3, 3, 1>(
        new MetricDepthError(rotation,
                             depth_prior,
                             sigma_depth,
                             use_log_scale,
                             residual_type,
                             zero_residual_behind,
                             log_linear_threshold));
  }

  const Eigen::Quaterniond rotation_;
  const double depth_prior_;
  const double sigma_depth_;
  const bool use_log_scale_;
  const MetricDepthResidualType residual_type_;
  const bool zero_residual_behind_;
  const double log_linear_threshold_;
};

}  // namespace vidmap
