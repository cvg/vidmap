#include "vidmap_native/types.h"

#include "colmap/scene/camera.h"
#include "colmap/util/types.h"

#include <cmath>
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

void CameraRecord::Validate() const {
  if (camera_id == std::numeric_limits<CameraId>::max()) {
    throw std::invalid_argument("camera_id must be valid");
  }
  if (width == 0 || height == 0) {
    throw std::invalid_argument("camera dimensions must be positive");
  }
  RequireFinite(params, "camera params");

  colmap::Camera camera;
  camera.camera_id = camera_id;
  camera.model_id = static_cast<colmap::CameraModelId>(model_id);
  camera.width = width;
  camera.height = height;
  camera.params.assign(params.data(), params.data() + params.size());
  if (!camera.VerifyParams()) {
    throw std::invalid_argument("camera parameters do not match model_id");
  }
}

void PoseRecord::Validate() const {
  RequireFinite(rotation_xyzw, "pose rotation");
  RequireFinite(translation, "pose translation");
  if (has_pose && rotation_xyzw.norm() <= 1e-12) {
    throw std::invalid_argument("posed rotation quaternion must be non-zero");
  }
}

void ImageRecord::Validate() const {
  if (image_id == std::numeric_limits<ImageId>::max() ||
      camera_id == std::numeric_limits<CameraId>::max() ||
      frame_id == std::numeric_limits<FrameId>::max()) {
    throw std::invalid_argument("image, camera, and frame IDs must be valid");
  }
  if (name.empty()) {
    throw std::invalid_argument("image name must not be empty");
  }
  pose.Validate();
  RequireFinite(keypoints, "keypoints");
  RequireFinite(bearings, "bearings");
  RequireFinite(depth_values, "depth values");
  RequireFinite(depth_stddevs, "depth standard deviations");
  RequireFinite(angular_stddevs, "angular standard deviations");

  const Eigen::Index num_features = keypoints.rows();
  RequireOptionalLength(bearings.rows(), num_features, "bearings");
  RequireOptionalLength(depth_values.size(), num_features, "depth values");
  RequireOptionalLength(
      depth_stddevs.size(), num_features, "depth standard deviations");
  RequireOptionalLength(depth_validity.size(), num_features, "depth validity");
  RequireOptionalLength(
      angular_stddevs.rows(), num_features, "angular standard deviations");
  RequireOptionalLength(is_inlier.size(), num_features, "inlier mask");
  RequireOptionalLength(
      is_track_anchor.size(), num_features, "track-anchor mask");
  RequireOptionalLength(
      is_depth_outlier.size(), num_features, "depth-outlier mask");
}

std::size_t ImageRecord::NumFeatures() const {
  return static_cast<std::size_t>(keypoints.rows());
}

void TwoViewGeometryRecord::Validate() const {
  if (has_essential) {
    RequireFinite(essential, "essential matrix");
  }
  if (has_fundamental) {
    RequireFinite(fundamental, "fundamental matrix");
  }
  if (has_homography) {
    RequireFinite(homography, "homography matrix");
  }
  cam2_from_cam1.Validate();
}

void PairRecord::Validate() const {
  if (image_id1 == std::numeric_limits<ImageId>::max() ||
      image_id2 == std::numeric_limits<ImageId>::max() ||
      image_id1 == image_id2) {
    throw std::invalid_argument("pair image IDs must be distinct and valid");
  }
  if (pair_id != CanonicalPairId(image_id1, image_id2)) {
    throw std::invalid_argument(
        "pair_id does not match the canonical image pair");
  }
  geometry.Validate();
  RequireOptionalLength(
      are_loop_closure.size(), all_matches.rows(), "loop-closure mask");
  for (Eigen::Index index = 0; index < inlier_indices.size(); ++index) {
    if (inlier_indices[index] < 0 ||
        inlier_indices[index] >= all_matches.rows()) {
      throw std::invalid_argument("inlier index is outside all_matches");
    }
  }
}

void TrackRecord::Validate() const {
  if (point3D_id == std::numeric_limits<Point3DId>::max()) {
    throw std::invalid_argument("point3D_id must be valid");
  }
  RequireFinite(xyz, "track xyz");
  if (!std::isfinite(error)) {
    throw std::invalid_argument("track error must be finite");
  }
  ValidateObservations(observations, "track observations");
  ValidateObservations(loop_closure_observations, "loop-closure observations");
  if (loop_closure_anchors.rows() != 0 &&
      loop_closure_anchors.rows() != loop_closure_observations.rows()) {
    throw std::invalid_argument(
        "loop-closure anchors must be empty or observation-aligned");
  }
  ValidateObservations(loop_closure_anchors, "loop-closure anchors");
}

PairId CanonicalPairId(ImageId image_id1, ImageId image_id2) {
  return colmap::ImagePairToPairId(image_id1, image_id2);
}

}  // namespace vidmap
