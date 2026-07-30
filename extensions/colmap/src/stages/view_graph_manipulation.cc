#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/two_view_geometry.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vidmap_native/conversion.h"
#include "vidmap_native/view_graph.h"

namespace vidmap {
namespace {

constexpr double kEpsilon = 1e-12;

colmap::TwoViewGeometry GeometryWithoutStoredInliers(const PairRecord& pair) {
  colmap::TwoViewGeometry geometry = ToColmapGeometry(pair);
  // PairRecord keeps all matches separate from the selected inlier rows.
  geometry.inlier_matches.clear();
  return geometry;
}

}  // namespace

void PrepareImageBearings(MappingProblem* problem) {
  problem->Validate();
  for (const ImageId image_id : problem->ImageIds()) {
    ImageRecord image = problem->Image(image_id);
    const colmap::Camera camera =
        ToColmapCamera(problem->Camera(image.camera_id));
    image.bearings.resize(image.keypoints.rows(), 3);
    for (Eigen::Index row = 0; row < image.keypoints.rows(); ++row) {
      const std::optional<Eigen::Vector2d> camera_point =
          camera.CamFromImg(image.keypoints.row(row));
      if (!camera_point.has_value()) {
        throw std::runtime_error("CamFromImg failed for feature " +
                                 std::to_string(row) + " of image " +
                                 std::to_string(image_id));
      }
      image.bearings.row(row) = camera_point->homogeneous().normalized();
    }
    problem->UpdateImage(image);
  }
}

void UpdateImagePairsConfig(MappingProblem* problem) {
  problem->Validate();
  std::unordered_map<CameraId, std::pair<int, int>> camera_counts;
  for (const PairId pair_id : problem->PairIds()) {
    const PairRecord& pair = problem->Pair(pair_id);
    if (!pair.is_valid) continue;
    const CameraRecord& camera1 =
        problem->Camera(problem->Image(pair.image_id1).camera_id);
    const CameraRecord& camera2 =
        problem->Camera(problem->Image(pair.image_id2).camera_id);
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length) {
      continue;
    }
    if (pair.geometry.configuration == colmap::TwoViewGeometry::CALIBRATED) {
      ++camera_counts[camera1.camera_id].first;
      ++camera_counts[camera2.camera_id].first;
      ++camera_counts[camera1.camera_id].second;
      ++camera_counts[camera2.camera_id].second;
    } else if (pair.geometry.configuration ==
               colmap::TwoViewGeometry::UNCALIBRATED) {
      ++camera_counts[camera1.camera_id].first;
      ++camera_counts[camera2.camera_id].first;
    }
  }

  std::unordered_map<CameraId, bool> camera_validity;
  for (const auto& [camera_id, counts] : camera_counts) {
    camera_validity[camera_id] =
        counts.first > 0 &&
        static_cast<double>(counts.second) / counts.first > 0.5;
  }

  for (const PairId pair_id : problem->PairIds()) {
    PairRecord pair = problem->Pair(pair_id);
    if (!pair.is_valid ||
        pair.geometry.configuration != colmap::TwoViewGeometry::UNCALIBRATED ||
        !pair.geometry.cam2_from_cam1.has_pose) {
      continue;
    }
    const CameraRecord& camera1 =
        problem->Camera(problem->Image(pair.image_id1).camera_id);
    const CameraRecord& camera2 =
        problem->Camera(problem->Image(pair.image_id2).camera_id);
    if (!camera_validity[camera1.camera_id] ||
        !camera_validity[camera2.camera_id]) {
      continue;
    }
    pair.geometry.configuration = colmap::TwoViewGeometry::CALIBRATED;
    pair.geometry.fundamental = colmap::FundamentalFromEssentialMatrix(
        ToColmapCamera(camera2).CalibrationMatrix(),
        colmap::EssentialMatrixFromPose(
            ToColmapPose(pair.geometry.cam2_from_cam1)),
        ToColmapCamera(camera1).CalibrationMatrix());
    pair.geometry.has_fundamental = true;
    problem->UpdatePair(pair);
  }
}

std::size_t DecomposeRelPose(MappingProblem* problem) {
  problem->Validate();
  std::size_t pure_rotation_count = 0;
  for (const PairId pair_id : problem->PairIds()) {
    PairRecord pair = problem->Pair(pair_id);
    if (!pair.is_valid) continue;
    const ImageRecord& image1 = problem->Image(pair.image_id1);
    const ImageRecord& image2 = problem->Image(pair.image_id2);
    const CameraRecord& camera1 = problem->Camera(image1.camera_id);
    const CameraRecord& camera2 = problem->Camera(image2.camera_id);
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length) {
      continue;
    }

    colmap::TwoViewGeometry geometry = GeometryWithoutStoredInliers(pair);
    const int original_configuration = geometry.config;
    colmap::EstimateTwoViewGeometryPose(ToColmapCamera(camera1),
                                        ToColmapPoints(image1.keypoints),
                                        ToColmapCamera(camera2),
                                        ToColmapPoints(image2.keypoints),
                                        &geometry);
    if (original_configuration == colmap::TwoViewGeometry::PLANAR) {
      geometry.config = colmap::TwoViewGeometry::CALIBRATED;
    } else if (geometry.cam2_from_cam1 &&
               geometry.cam2_from_cam1->translation().norm() > kEpsilon) {
      geometry.cam2_from_cam1->translation().normalize();
    }
    UpdateGeometryRecord(geometry, &pair.geometry);
    problem->UpdatePair(pair);
    if (geometry.config != colmap::TwoViewGeometry::CALIBRATED &&
        geometry.config != colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC) {
      ++pure_rotation_count;
    }
  }
  return pure_rotation_count;
}

std::size_t FilterPairsByInlierNum(int min_inlier_count,
                                   MappingProblem* problem) {
  if (min_inlier_count < 0) {
    throw std::invalid_argument("min_inlier_count must be non-negative");
  }
  std::size_t filtered = 0;
  for (const PairId pair_id : problem->PairIds()) {
    PairRecord pair = problem->Pair(pair_id);
    if (pair.is_valid && pair.inlier_indices.size() < min_inlier_count) {
      pair.is_valid = false;
      problem->UpdatePair(pair);
      ++filtered;
    }
  }
  return filtered;
}

std::size_t FilterPairsByInlierRatio(double min_inlier_ratio,
                                     MappingProblem* problem) {
  if (min_inlier_ratio < 0.0 || min_inlier_ratio > 1.0) {
    throw std::invalid_argument("min_inlier_ratio must be in [0, 1]");
  }
  std::size_t filtered = 0;
  for (const PairId pair_id : problem->PairIds()) {
    PairRecord pair = problem->Pair(pair_id);
    if (!pair.is_valid || pair.all_matches.rows() == 0) continue;
    const double ratio = static_cast<double>(pair.inlier_indices.size()) /
                         pair.all_matches.rows();
    if (ratio < min_inlier_ratio) {
      pair.is_valid = false;
      problem->UpdatePair(pair);
      ++filtered;
    }
  }
  return filtered;
}

}  // namespace vidmap
