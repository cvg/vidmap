#include "vidmap_native/types.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace vidmap {
namespace {

template <typename Derived>
void RequireFinite(const Eigen::MatrixBase<Derived>& values,
                   const std::string& name) {
  if (!values.allFinite()) {
    throw std::invalid_argument(name + " must contain only finite values");
  }
}

void RequireOptionalLength(Eigen::Index size,
                           Eigen::Index expected,
                           const std::string& name) {
  if (size != 0 && size != expected) {
    throw std::invalid_argument(name + " must be empty or feature-aligned");
  }
}

void ValidateObservations(const MatrixX2u& observations,
                          const std::string& name) {
  for (Eigen::Index row = 0; row < observations.rows(); ++row) {
    if (observations(row, 0) == std::numeric_limits<ImageId>::max()) {
      throw std::invalid_argument(name + " contains an invalid image ID");
    }
  }
}

}  // namespace

void ImageData::Validate(std::size_t num_features) const {
  RequireFinite(bearings, "bearings");
  RequireFinite(depth_values, "depth values");
  RequireFinite(depth_stddevs, "depth standard deviations");

  RequireOptionalLength(bearings.rows(), num_features, "bearings");
  RequireOptionalLength(depth_values.size(), num_features, "depth values");
  RequireOptionalLength(
      depth_stddevs.size(), num_features, "depth standard deviations");
  RequireOptionalLength(depth_validity.size(), num_features, "depth validity");
  RequireOptionalLength(
      is_depth_outlier.size(), num_features, "depth-outlier mask");
}

void PairData::Validate() const {
  if (geometry.F) {
    RequireFinite(*geometry.F, "fundamental matrix");
  }
  if (geometry.H) {
    RequireFinite(*geometry.H, "homography matrix");
  }
  RequireOptionalLength(
      are_loop_closure.size(), all_matches.rows(), "loop-closure mask");
  for (Eigen::Index index = 0; index < inlier_indices.size(); ++index) {
    if (inlier_indices[index] < 0 ||
        inlier_indices[index] >= all_matches.rows()) {
      throw std::invalid_argument("inlier index is outside all_matches");
    }
  }
}

void TrackData::Validate() const {
  ValidateObservations(loop_closure_observations, "loop-closure observations");
  if (loop_closure_anchors.rows() != 0 &&
      loop_closure_anchors.rows() != loop_closure_observations.rows()) {
    throw std::invalid_argument(
        "loop-closure anchors must be empty or observation-aligned");
  }
  ValidateObservations(loop_closure_anchors, "loop-closure anchors");
}

}  // namespace vidmap
