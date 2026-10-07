#include "imu_uncertainty.h"

#include "colmap/math/math.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <utility>

namespace vidmap {
namespace {

double MedianOfVector(std::vector<double>* values) {
  if (values->empty()) return 0.0;
  std::sort(values->begin(), values->end());
  const std::size_t mid = values->size() / 2;
  if (values->size() % 2 == 1) {
    return (*values)[mid];
  }
  return 0.5 * ((*values)[mid - 1] + (*values)[mid]);
}

}  // namespace

ImuDynamicRotationThresholdModel::ImuDynamicRotationThresholdModel(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImuEdgeRecord>& imu_edges,
    const double max_rotation_error_deg)
    : theta_cap_rad_(colmap::DegToRad(max_rotation_error_deg)),
      enabled_(options.use_imu && options.use_dynamic_imu_rotation_threshold &&
               !imu_edges.empty() && max_rotation_error_deg > 0.0),
      k_sigma_(options.imu_dynamic_rotation_threshold_multiplier),
      sigma_vis_rad_(colmap::DegToRad(options.visual_rotation_stddev_deg)),
      sigma_bg0_rad_s_(options.imu_gyro_bias_stddev_rad_s) {
  if (!enabled_) return;

  std::vector<double> sigma_g_sq_samples;
  std::vector<double> sigma_bg_rw_sq_samples;
  sigma_g_sq_samples.reserve(imu_edges.size());
  sigma_bg_rw_sq_samples.reserve(imu_edges.size());

  std::map<ImageId, std::vector<std::pair<ImageId, double>>> adj;
  for (const ImuEdgeRecord& edge : imu_edges) {
    const double dt = edge.data.delta_t;
    if (!std::isfinite(dt) || dt <= 1e-6) continue;
    adj[edge.image_id1].emplace_back(edge.image_id2, dt);
    adj[edge.image_id2].emplace_back(edge.image_id1, -dt);

    const double trace_cov_dR = edge.data.covariance.block<3, 3>(0, 0).trace();
    const double trace_cov_bg = edge.data.covariance.block<3, 3>(9, 9).trace();
    if (std::isfinite(trace_cov_dR) && trace_cov_dR >= 0.0 &&
        std::isfinite(trace_cov_bg) && trace_cov_bg >= 0.0) {
      const double sigma_bg_rw_sq = std::max(0.0, trace_cov_bg / (3.0 * dt));
      const double sigma_g_sq = std::max(
          0.0,
          trace_cov_dR / (3.0 * dt) - (1.0 / 3.0) * sigma_bg_rw_sq * dt * dt);
      sigma_g_sq_samples.push_back(sigma_g_sq);
      sigma_bg_rw_sq_samples.push_back(sigma_bg_rw_sq);
    }
  }

  sigma_g_sq_ = MedianOfVector(&sigma_g_sq_samples);
  sigma_bg_rw_sq_ = MedianOfVector(&sigma_bg_rw_sq_samples);

  int comp_id = 0;
  for (const auto& [root_id, _] : adj) {
    if (image_comp_.count(root_id) != 0) continue;
    ++comp_id;
    image_comp_[root_id] = comp_id;
    image_time_s_[root_id] = 0.0;
    std::queue<ImageId> q;
    q.push(root_id);
    while (!q.empty()) {
      const ImageId u = q.front();
      q.pop();
      const double t_u = image_time_s_[u];
      for (const auto& [v, signed_dt] : adj[u]) {
        if (image_comp_.count(v) == 0) {
          image_comp_[v] = comp_id;
          image_time_s_[v] = t_u + signed_dt;
          q.push(v);
        }
      }
    }
  }
}

double ImuDynamicRotationThresholdModel::MaxRotationErrorRad(
    const ImageId image_id1, const ImageId image_id2) const {
  if (!enabled_) return theta_cap_rad_;
  const auto c1_it = image_comp_.find(image_id1);
  const auto c2_it = image_comp_.find(image_id2);
  if (c1_it == image_comp_.end() || c2_it == image_comp_.end() ||
      c1_it->second != c2_it->second) {
    return theta_cap_rad_;
  }
  const double dt =
      std::abs(image_time_s_.at(image_id2) - image_time_s_.at(image_id1));
  const double var_1d = sigma_vis_rad_ * sigma_vis_rad_ + sigma_g_sq_ * dt +
                        (sigma_bg0_rad_s_ * sigma_bg0_rad_s_) * (dt * dt) +
                        (1.0 / 3.0) * sigma_bg_rw_sq_ * (dt * dt * dt);
  const double sigma_total_3d_rad = std::sqrt(3.0 * std::max(0.0, var_1d));
  return std::min(theta_cap_rad_, k_sigma_ * sigma_total_3d_rad);
}

}  // namespace vidmap
