#include <cmath>
#include <stdexcept>
#include <string>

#include "stages/intrinsics_prior.h"
#include "vidmap_native/tracks.h"
#include "vidmap_native/types.h"
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

void TestOptionValidation() {
  vidmap::TrackEstablishmentOptions track_options;
  track_options.min_num_views_per_track = 0;
  CheckInvalidArgument([&] { track_options.Validate(); },
                       "zero track minimum was accepted");

  vidmap::InlierThresholdOptions inlier_options;
  inlier_options.min_angle_from_epipole_deg = 181.0;
  CheckInvalidArgument([&] { inlier_options.Validate(); },
                       "invalid epipole angle was accepted");
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
  TestOptionValidation();
  TestFrozenLogFocalJacobian();
  return 0;
}
