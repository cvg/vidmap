#pragma once

#include <unordered_map>
#include <vector>

#include "vidmap_native/imu_types.h"
#include "vidmap_native/video_rotation_averaging.h"

namespace vidmap {

// Closed-form SO(3) preintegrated rotation uncertainty propagation model for
// dynamic pair rotation outlier thresholding:
//   sigma_total,3D(dt) = sqrt(3 * (sigma_vis^2 + sigma_g^2 * dt
//                                  + sigma_bg0^2 * dt^2
//                                  + (1/3) * sigma_bg_rw^2 * dt^3))
//   theta_max(dt) = min(theta_cap, k_sigma * sigma_total,3D(dt))
class ImuDynamicRotationThresholdModel {
 public:
  ImuDynamicRotationThresholdModel(const VideoRotationAveragingOptions& options,
                                   const std::vector<ImuEdgeRecord>& imu_edges,
                                   double max_rotation_error_deg);

  double MaxRotationErrorRad(ImageId image_id1, ImageId image_id2) const;

 private:
  double theta_cap_rad_ = 0.0;
  bool enabled_ = false;
  double k_sigma_ = 3.5;
  double sigma_vis_rad_ = 0.0;
  double sigma_bg0_rad_s_ = 0.0;
  double sigma_g_sq_ = 0.0;
  double sigma_bg_rw_sq_ = 0.0;
  std::unordered_map<ImageId, int> image_comp_;
  std::unordered_map<ImageId, double> image_time_s_;
};

}  // namespace vidmap
