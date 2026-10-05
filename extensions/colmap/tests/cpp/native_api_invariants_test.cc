#include <cmath>
#include <stdexcept>
#include <string>

#include "stages/intrinsics_prior.h"
#include "vidmap_native/bundle_adjustment.h"
#include "vidmap_native/global_positioning.h"
#include "vidmap_native/tracks.h"
#include "vidmap_native/types.h"
#include "vidmap_native/video_rotation_averaging.h"
#include "vidmap_native/view_graph.h"

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callable>
void CheckInvalidArgument(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void TestStableIdentifiers() {
  Check(vidmap::CanonicalPairId(1, 2) == vidmap::CanonicalPairId(2, 1),
        "canonical pair IDs must be orientation independent");
  Check(vidmap::EncodeObservationKey(7, 11) ==
            (static_cast<vidmap::Point3DId>(7) << 32 | 11),
        "track observation encoding changed");
}

void TestSolverDefaults() {
  const vidmap::GlobalPositionerOptions options;
  Check(
      options.parameter_ordering == vidmap::GlobalPositioningOrdering::kGrouped,
      "global positioning ordering default changed");
  Check(options.center_mode == vidmap::GlobalPositioningCenterMode::kFrame,
        "global positioning center default changed");
  options.Validate();
}

void TestOptionValidation() {
  vidmap::TrackEstablishmentOptions track_options;
  track_options.required_tracks_per_view = -1;
  CheckInvalidArgument([&] { track_options.Validate(); },
                       "negative track quota was accepted");

  vidmap::InlierThresholdOptions inlier_options;
  inlier_options.min_angle_from_epipole_deg = 181.0;
  CheckInvalidArgument([&] { inlier_options.Validate(); },
                       "invalid epipole angle was accepted");

  vidmap::VideoRotationAveragingOptions rotation_options;
  rotation_options.num_threads = 0;
  CheckInvalidArgument([&] { rotation_options.Validate(); },
                       "zero rotation-averaging threads were accepted");

  vidmap::BundleAdjustmentOptions bundle_options;
  bundle_options.max_num_iterations = 0;
  CheckInvalidArgument([&] { bundle_options.Validate(); },
                       "zero bundle-adjustment iterations were accepted");

  vidmap::BundleAdjustmentOptions imu_ba_options;
  imu_ba_options.use_imu = true;
  imu_ba_options.Validate();
  imu_ba_options.initial_gravity_direction.setZero();
  CheckInvalidArgument([&] { imu_ba_options.Validate(); },
                       "zero initial gravity direction was accepted");

  vidmap::ImuStateRecord state_record;
  state_record.velocity.x() = std::numeric_limits<double>::quiet_NaN();
  CheckInvalidArgument([&] { state_record.Validate(); },
                       "NaN velocity in ImuStateRecord was accepted");
  state_record.image_id = 5;
  state_record.velocity = Eigen::Vector3d(1.0, 2.0, 3.0);
  state_record.bias_gyro = Eigen::Vector3d(0.01, -0.02, 0.03);
  state_record.bias_accel = Eigen::Vector3d(-0.1, 0.2, -0.3);
  state_record.Validate();
  const auto roundtrip =
      vidmap::ImuStateRecord::FromVector(5, state_record.ToVector(), 2.5);
  Check((roundtrip.velocity - state_record.velocity).norm() < 1e-12,
        "ImuStateRecord velocity roundtrip failed");
  Check(
      (roundtrip.metric_velocity - 2.5 * state_record.velocity).norm() < 1e-12,
      "ImuStateRecord metric_velocity scaling failed");

  vidmap::ImuEdgeRecord edge_record;
  edge_record.image_id1 = 1;
  edge_record.image_id2 = 1;
  edge_record.data.delta_t = 0.1;
  edge_record.data.sqrt_info.setIdentity();
  CheckInvalidArgument([&] { edge_record.Validate(); },
                       "self-loop ImuEdgeRecord was accepted");
  edge_record.image_id2 = 2;
  edge_record.Validate();
}

void TestFrozenLogFocalJacobian() {
  // The production cost for scalar VGC, SIMPLE_PINHOLE and PINHOLE. GeoCalib's
  // first-order conversion uses source-pixel standard deviation, not variance.
  for (const int dimension : {1, 3, 4}) {
    const int focal_count = dimension == 4 ? 2 : 1;
    std::vector<std::size_t> indices;
    for (int i = 0; i < focal_count; ++i) indices.push_back(i);
    const double target = 500.0;
    const double sigma_f = 10.0;
    const double sigma_log = sigma_f / target;
    vidmap::LogMeanFocalPriorCostFunction cost(
        dimension, indices, target, sigma_log);
    std::vector<double> params(dimension, 600.0), jacobian(dimension);
    const double* blocks[] = {params.data()};
    double* jacobians[] = {jacobian.data()};
    double residual;
    Check(cost.Evaluate(blocks, &residual, jacobians),
          "log cost evaluation failed");
    Check(std::abs(residual - std::log(600.0 / target) / sigma_log) < 1e-12,
          "first-order log conversion changed");
    for (int i = 0; i < dimension; ++i) {
      const double expected =
          i < focal_count ? 1.0 / (sigma_log * 600.0 * focal_count) : 0.0;
      Check(std::abs(jacobian[i] - expected) < 1e-15,
            "log focal analytic Jacobian changed");
      const double step = 1e-3;
      double plus, minus;
      params[i] += step;
      cost.Evaluate(blocks, &plus, nullptr);
      params[i] -= 2 * step;
      cost.Evaluate(blocks, &minus, nullptr);
      params[i] += step;
      Check(std::abs(jacobian[i] - (plus - minus) / (2 * step)) < 1e-9,
            "log focal finite-difference Jacobian mismatch");
    }
    for (int i = 0; i < focal_count; ++i) params[i] = target;
    cost.Evaluate(blocks, &residual, jacobians);
    Check(residual == 0.0, "original target must have zero residual");
    Check(std::abs(jacobian[0] - 1.0 / (sigma_f * focal_count)) < 1e-15,
          "first-order derivative at original focal must match pixel "
          "uncertainty");
  }
}

}  // namespace

int main() {
  TestStableIdentifiers();
  TestSolverDefaults();
  TestOptionValidation();
  TestFrozenLogFocalJacobian();
  return 0;
}
