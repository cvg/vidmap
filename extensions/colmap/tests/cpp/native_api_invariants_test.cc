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

void TestRelativeLogFocalJacobian() {
  const double target_ratio = std::log(1.2);
  const double sigma_log = 0.05;
  for (const int dim1 : {1, 3, 4}) {
    for (const int dim2 : {1, 3, 4}) {
      const int focal_count1 = dim1 == 4 ? 2 : 1;
      const int focal_count2 = dim2 == 4 ? 2 : 1;
      std::vector<std::size_t> idxs1;
      for (int i = 0; i < focal_count1; ++i) idxs1.push_back(i);
      std::vector<std::size_t> idxs2;
      for (int i = 0; i < focal_count2; ++i) idxs2.push_back(i);

      vidmap::LogRelativeFocalPriorCostFunction cost(
          dim1, idxs1, dim2, idxs2, target_ratio, sigma_log);

      std::vector<double> p1(dim1, 500.0);
      std::vector<double> p2(dim2, 600.0);
      std::vector<double> j1(dim1), j2(dim2);
      const double* blocks[] = {p1.data(), p2.data()};
      double* jacobians[] = {j1.data(), j2.data()};
      double residual;
      Check(cost.Evaluate(blocks, &residual, jacobians),
            "relative focal cost evaluation failed");
      const double expected_res =
          ((std::log(600.0) - std::log(500.0)) - target_ratio) / sigma_log;
      Check(std::abs(residual - expected_res) < 1e-12,
            "relative focal residual mismatch");

      // Check jacobians with finite differences
      const double step = 1e-4;
      for (int i = 0; i < dim1; ++i) {
        double plus, minus;
        p1[i] += step;
        cost.Evaluate(blocks, &plus, nullptr);
        p1[i] -= 2 * step;
        cost.Evaluate(blocks, &minus, nullptr);
        p1[i] += step;
        Check(std::abs(j1[i] - (plus - minus) / (2 * step)) < 1e-8,
              "relative focal p1 finite-difference mismatch");
      }
      for (int i = 0; i < dim2; ++i) {
        double plus, minus;
        p2[i] += step;
        cost.Evaluate(blocks, &plus, nullptr);
        p2[i] -= 2 * step;
        cost.Evaluate(blocks, &minus, nullptr);
        p2[i] += step;
        Check(std::abs(j2[i] - (plus - minus) / (2 * step)) < 1e-8,
              "relative focal p2 finite-difference mismatch");
      }
    }
  }
}

}  // namespace

int main() {
  TestOptionValidation();
  TestFrozenLogFocalJacobian();
  TestRelativeLogFocalJacobian();
  return 0;
}
