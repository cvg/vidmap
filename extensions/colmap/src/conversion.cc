#include "vidmap_native/conversion.h"

#include "colmap/feature/types.h"

#include <optional>

namespace vidmap {

colmap::Camera ToColmapCamera(const CameraRecord& record) {
  record.Validate();
  colmap::Camera camera;
  camera.camera_id = record.camera_id;
  camera.model_id = static_cast<colmap::CameraModelId>(record.model_id);
  camera.width = record.width;
  camera.height = record.height;
  camera.params.assign(record.params.data(),
                       record.params.data() + record.params.size());
  camera.has_prior_focal_length = record.has_prior_focal_length;
  return camera;
}

colmap::Rigid3d ToColmapPose(const PoseRecord& record) {
  record.Validate();
  Eigen::Quaterniond rotation;
  rotation.coeffs() = record.rotation_xyzw;
  return colmap::Rigid3d(rotation, record.translation);
}

PoseRecord FromColmapPose(const colmap::Rigid3d& pose) {
  PoseRecord record;
  record.has_pose = true;
  record.rotation_xyzw = pose.rotation().coeffs();
  record.translation = pose.translation();
  return record;
}

std::vector<Eigen::Vector2d> ToColmapPoints(const MatrixX2d& points) {
  std::vector<Eigen::Vector2d> output;
  output.reserve(points.rows());
  for (Eigen::Index row = 0; row < points.rows(); ++row) {
    output.push_back(points.row(row));
  }
  return output;
}

colmap::TwoViewGeometry ToColmapGeometry(const PairRecord& pair) {
  colmap::TwoViewGeometry geometry;
  geometry.config = pair.geometry.configuration;
  if (pair.geometry.has_essential) {
    geometry.E = pair.geometry.essential;
  }
  if (pair.geometry.has_fundamental) {
    geometry.F = pair.geometry.fundamental;
  }
  if (pair.geometry.has_homography) {
    geometry.H = pair.geometry.homography;
  }
  if (pair.geometry.cam2_from_cam1.has_pose) {
    geometry.cam2_from_cam1 = ToColmapPose(pair.geometry.cam2_from_cam1);
  }
  geometry.inlier_matches.reserve(pair.inlier_indices.size());
  for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
    const Eigen::Index row = pair.inlier_indices[index];
    geometry.inlier_matches.push_back(
        {pair.all_matches(row, 0), pair.all_matches(row, 1)});
  }
  return geometry;
}

void UpdateGeometryRecord(const colmap::TwoViewGeometry& geometry,
                          TwoViewGeometryRecord* record) {
  record->configuration = geometry.config;
  record->has_essential = geometry.E.has_value();
  record->has_fundamental = geometry.F.has_value();
  record->has_homography = geometry.H.has_value();
  if (geometry.E) {
    record->essential = *geometry.E;
  }
  if (geometry.F) {
    record->fundamental = *geometry.F;
  }
  if (geometry.H) {
    record->homography = *geometry.H;
  }
  if (geometry.cam2_from_cam1) {
    record->cam2_from_cam1 = FromColmapPose(*geometry.cam2_from_cam1);
  } else {
    record->cam2_from_cam1 = PoseRecord();
  }
}

}  // namespace vidmap
