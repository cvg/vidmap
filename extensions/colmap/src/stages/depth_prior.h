#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace vidmap {

struct LogScaledDepthErrorCostFunctor
    : public colmap::
          AutoDiffCostFunctor<LogScaledDepthErrorCostFunctor, 1, 7, 3, 1> {
  explicit LogScaledDepthErrorCostFunctor(
      double depth, const colmap::Rigid3d& sensor_from_rig = colmap::Rigid3d())
      : depth_(depth),
        sensor_z_axis_(sensor_from_rig.rotation().toRotationMatrix().row(2)),
        sensor_z_offset_(sensor_from_rig.translation().z()),
        direct_depth_(sensor_z_axis_ == Eigen::Vector3d::UnitZ() &&
                      sensor_z_offset_ == 0.0) {}

  template <typename T>
  bool operator()(const T* pose,
                  const T* point3D,
                  const T* log_scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> rotated_point =
        colmap::EigenQuaternionMap<T>(pose) *
        colmap::EigenVector3Map<T>(point3D);
    const T predicted_depth =
        direct_depth_
            ? rotated_point[2] + pose[6]
            : sensor_z_axis_.cast<T>().dot(
                  rotated_point + colmap::EigenVector3Map<T>(pose + 4)) +
                  T(sensor_z_offset_);
    if (predicted_depth <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }
    residuals[0] =
        ceres::log(predicted_depth) - (ceres::log(T(depth_)) + log_scale[0]);
    return true;
  }

 private:
  const double depth_;
  const Eigen::Vector3d sensor_z_axis_;
  const double sensor_z_offset_;
  const bool direct_depth_;
};

}  // namespace vidmap
