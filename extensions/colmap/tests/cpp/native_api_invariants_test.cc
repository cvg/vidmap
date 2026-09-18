#include <cmath>
#include <stdexcept>
#include "stages/intrinsics_prior.h"
#include <string>

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
    vidmap::LogMeanFocalPriorCostFunction cost(dimension, indices, target, sigma_log);
    std::vector<double> params(dimension, 600.0), jacobian(dimension);
    const double* blocks[] = {params.data()};
    double* jacobians[] = {jacobian.data()};
    double residual;
    Check(cost.Evaluate(blocks, &residual, jacobians), "log cost evaluation failed");
    Check(std::abs(residual - std::log(600.0 / target) / sigma_log) < 1e-12,
          "first-order log conversion changed");
    for (int i = 0; i < dimension; ++i) {
      const double expected = i < focal_count ? 1.0 / (sigma_log * 600.0 * focal_count) : 0.0;
      Check(std::abs(jacobian[i] - expected) < 1e-15, "log focal analytic Jacobian changed");
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
          "first-order derivative at original focal must match pixel uncertainty");
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
