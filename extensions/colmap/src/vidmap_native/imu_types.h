#pragma once

#include "colmap/estimators/imu_preintegration.h"

#include "vidmap_native/ceres_loss.h"
#include "vidmap_native/types.h"
#include <Eigen/Geometry>

namespace vidmap {

struct ImuStateRecord {
  ImageId image_id = 0;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d metric_velocity = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d bias_accel = Eigen::Vector3d::Zero();

  Eigen::Matrix<double, 9, 1> ToVector() const {
    Eigen::Matrix<double, 9, 1> vec;
    vec.segment<3>(0) = velocity;
    vec.segment<3>(3) = bias_gyro;
    vec.segment<3>(6) = bias_accel;
    return vec;
  }

  static ImuStateRecord FromVector(ImageId image_id,
                                   const Eigen::Matrix<double, 9, 1>& vec,
                                   double scale = 1.0) {
    ImuStateRecord record;
    record.image_id = image_id;
    record.velocity = vec.segment<3>(0);
    record.metric_velocity = record.velocity * scale;
    record.bias_gyro = vec.segment<3>(3);
    record.bias_accel = vec.segment<3>(6);
    return record;
  }

  void Validate() const;
};

struct ImuEdgeRecord {
  ImageId image_id1 = 0;
  ImageId image_id2 = 0;
  colmap::PreintegratedImuData data;
  colmap::ImuPreintegrator* integrator = nullptr;
  Eigen::Quaterniond q_iori_1_xyzw = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond q_iori_2_xyzw = Eigen::Quaterniond::Identity();
  LossConfig loss;

  void Validate() const;
};

}  // namespace vidmap
