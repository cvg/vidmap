#pragma once

#include "colmap/scene/reconstruction.h"

#include <cmath>
#include <stdexcept>

namespace vidmap {

inline colmap::Rigid3d FixedSensorFromRig(const colmap::Image& image) {
  if (image.IsRefInFrame()) return colmap::Rigid3d();
  const auto& rig = *image.FramePtr()->RigPtr();
  const auto sensor_id = image.CameraPtr()->SensorId();
  if (!rig.HasSensorFromRig(sensor_id))
    throw std::invalid_argument("solver requires known fixed rig extrinsics");
  const auto& pose = rig.SensorFromRig(sensor_id);
  if (!pose.rotation().coeffs().allFinite() ||
      std::abs(pose.rotation().norm() - 1.0) > 1e-8 ||
      !pose.translation().allFinite())
    throw std::invalid_argument(
        "solver requires finite rigid sensor transforms");
  return pose;
}

// Positive offset added to X - C_rig in the fixed-rig BATA convention.
inline Eigen::Vector3d RigTranslationInWorld(const colmap::Image& image) {
  if (image.IsRefInFrame()) return Eigen::Vector3d::Zero();
  return image.CamFromWorld().rotation().inverse() *
         FixedSensorFromRig(image).translation();
}

inline bool ValidateFixedRigScene(
    const colmap::Reconstruction& reconstruction) {
  bool is_rig = false;
  for (const auto& [id, image] : reconstruction.Images()) {
    FixedSensorFromRig(image);
    is_rig |= !image.IsRefInFrame();
  }
  return is_rig;
}

}  // namespace vidmap
