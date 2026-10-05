#pragma once

#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace vidmap {

struct LogScaledDepthErrorCostFunctor
    : public colmap::
          AutoDiffCostFunctor<LogScaledDepthErrorCostFunctor, 1, 7, 3, 1> {
  explicit LogScaledDepthErrorCostFunctor(double depth) : depth_(depth) {}

  template <typename T>
  bool operator()(const T* pose,
                  const T* point3D,
                  const T* log_scale,
                  T* residuals) const {
    const T predicted_depth = (colmap::EigenQuaternionMap<T>(pose) *
                               colmap::EigenVector3Map<T>(point3D))[2] +
                              pose[6];
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
};

}  // namespace vidmap
