#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/homography_matrix.h"
#include "colmap/math/math.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "vidmap_native/view_graph.h"

namespace vidmap {
namespace {

constexpr double kEpsilon = 1e-12;

bool CheckCheirality(const colmap::Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_z,
                     double max_z) {
  const Eigen::Vector3d rotated_x1 = pose.rotation() * x1;
  const double a = -rotated_x1.dot(x2);
  const double b1 = -rotated_x1.dot(pose.translation());
  const double b2 = x2.dot(pose.translation());
  const double lambda1 = b1 - a * b2;
  const double lambda2 = -a * b1 + b2;
  min_z *= 1.0 - a * a;
  max_z *= 1.0 - a * a;
  return lambda1 > min_z && lambda2 > min_z && lambda1 < max_z &&
         lambda2 < max_z;
}

double OrientationSign(const Eigen::Matrix3d& fundamental,
                       const Eigen::Vector3d& epipole,
                       const Eigen::Vector2d& point1,
                       const Eigen::Vector2d& point2) {
  const double sign1 = fundamental(0, 0) * point2[0] +
                       fundamental(1, 0) * point2[1] + fundamental(2, 0);
  const double sign2 = epipole(1) - epipole(2) * point1[1];
  return sign1 * sign2;
}

double BearingSampsonError(const Eigen::Matrix3d& essential,
                           const Eigen::Vector3d& point1,
                           const Eigen::Vector3d& point2) {
  const Eigen::Vector3d essential_point1 =
      essential * point1 / (kEpsilon + point1[2]);
  const Eigen::Vector3d essential_t_point2 =
      essential.transpose() * point2 / (kEpsilon + point2[2]);
  const double constraint = essential_point1.dot(point2);
  return constraint * constraint /
         (essential_point1.head<2>().squaredNorm() +
          essential_t_point2.head<2>().squaredNorm());
}

void ScoreEssentialPair(const InlierThresholdOptions& options,
                        const colmap::Reconstruction& reconstruction,
                        const colmap::PoseGraph& graph,
                        const MappingSidecars& sidecars,
                        PairId pair_id,
                        PairData* pair) {
  if (!pair->has_relative_pose) {
    throw std::invalid_argument(
        "calibrated pair requires cam2_from_cam1 for inlier scoring");
  }
  const colmap::Rigid3d cam2_from_cam1 =
      graph.Edges().at(pair_id).cam2_from_cam1;
  const Eigen::Matrix3d essential =
      colmap::EssentialMatrixFromPose(cam2_from_cam1);
  Eigen::Vector3d epipole12 = cam2_from_cam1.translation();
  Eigen::Vector3d epipole21 = colmap::Inverse(cam2_from_cam1).translation();
  if (epipole12.norm() > kEpsilon) epipole12.normalize();
  if (epipole21.norm() > kEpsilon) epipole21.normalize();
  if (epipole12[2] < 0.0) epipole12 = -epipole12;
  if (epipole21[2] < 0.0) epipole21 = -epipole21;

  const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
  const auto& image1 = reconstruction.Image(id1);
  const auto& image2 = reconstruction.Image(id2);
  const auto& data1 = sidecars.images.at(id1);
  const auto& data2 = sidecars.images.at(id2);
  const colmap::Camera camera1 = *image1.CameraPtr();
  const colmap::Camera camera2 = *image2.CameraPtr();
  if (data1.bearings.rows() != image1.NumPoints2D() ||
      data2.bearings.rows() != image2.NumPoints2D()) {
    throw std::invalid_argument(
        "essential inlier scoring requires feature-aligned bearings");
  }

  const double threshold =
      options.max_epipolar_error_essential * 0.5 *
      (1.0 / camera1.MeanFocalLength() + 1.0 / camera2.MeanFocalLength());
  const double threshold_sq = threshold * threshold;
  const double epipole_threshold =
      std::cos(colmap::DegToRad(options.min_angle_from_epipole_deg)) + 1e-6;

  std::vector<int> inliers;
  inliers.reserve(pair->all_matches.rows());
  for (Eigen::Index row = 0; row < pair->all_matches.rows(); ++row) {
    const Eigen::Vector3d point1 =
        data1.bearings.row(pair->all_matches(row, 0));
    const Eigen::Vector3d point2 =
        data2.bearings.row(pair->all_matches(row, 1));
    if (BearingSampsonError(essential, point1, point2) >= threshold_sq) {
      continue;
    }
    if (!CheckCheirality(cam2_from_cam1, point1, point2, 1e-2, 100.0)) {
      continue;
    }
    if (point1.dot(epipole21) >= epipole_threshold ||
        point2.dot(epipole12) >= epipole_threshold) {
      continue;
    }
    inliers.push_back(static_cast<int>(row));
  }
  pair->inlier_indices =
      Eigen::Map<const VectorXi>(inliers.data(), inliers.size());
}

void ScoreFundamentalPair(const InlierThresholdOptions& options,
                          const colmap::Reconstruction& reconstruction,
                          PairId pair_id,
                          PairData* pair) {
  if (!pair->geometry.F) {
    throw std::invalid_argument(
        "uncalibrated pair requires a fundamental matrix");
  }
  const Eigen::Matrix3d& fundamental = *pair->geometry.F;
  Eigen::Vector3d epipole = fundamental.row(0).cross(fundamental.row(2));
  if ((epipole.array().abs() <= kEpsilon).all()) {
    epipole = fundamental.row(1).cross(fundamental.row(2));
  }

  const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
  const auto& image1 = reconstruction.Image(id1);
  const auto& image2 = reconstruction.Image(id2);
  const double threshold_sq = options.max_epipolar_error_fundamental *
                              options.max_epipolar_error_fundamental;
  std::vector<double> signs;
  std::vector<int> provisional;
  int positive_count = 0;
  int negative_count = 0;
  for (Eigen::Index row = 0; row < pair->all_matches.rows(); ++row) {
    const Eigen::Vector2d point1 = image1.Point2D(pair->all_matches(row, 0)).xy;
    const Eigen::Vector2d point2 = image2.Point2D(pair->all_matches(row, 1)).xy;
    const double error = colmap::ComputeSquaredSampsonError(
        point1.homogeneous(), point2.homogeneous(), fundamental);
    if (error >= threshold_sq) continue;
    signs.push_back(OrientationSign(fundamental, epipole, point1, point2));
    if (signs.back() > 0.0) {
      ++positive_count;
    } else {
      ++negative_count;
    }
    provisional.push_back(static_cast<int>(row));
  }

  std::vector<int> inliers;
  if (positive_count != negative_count) {
    const bool use_positive = positive_count > negative_count;
    for (std::size_t index = 0; index < provisional.size(); ++index) {
      if ((signs[index] > 0.0) == use_positive) {
        inliers.push_back(provisional[index]);
      }
    }
  }
  pair->inlier_indices =
      Eigen::Map<const VectorXi>(inliers.data(), inliers.size());
}

void ScoreHomographyPair(const InlierThresholdOptions& options,
                         const colmap::Reconstruction& reconstruction,
                         PairId pair_id,
                         PairData* pair) {
  if (!pair->geometry.H) {
    throw std::invalid_argument("planar pair requires a homography matrix");
  }
  const auto [id1, id2] = colmap::PairIdToImagePair(pair_id);
  const auto& image1 = reconstruction.Image(id1);
  const auto& image2 = reconstruction.Image(id2);
  const double threshold_sq = options.max_epipolar_error_homography *
                              options.max_epipolar_error_homography;
  std::vector<int> inliers;
  for (Eigen::Index row = 0; row < pair->all_matches.rows(); ++row) {
    const Eigen::Vector2d point1 = image1.Point2D(pair->all_matches(row, 0)).xy;
    const Eigen::Vector2d point2 = image2.Point2D(pair->all_matches(row, 1)).xy;
    if (colmap::ComputeSquaredHomographyError(
            point1, point2, *pair->geometry.H) < threshold_sq) {
      inliers.push_back(static_cast<int>(row));
    }
  }
  pair->inlier_indices =
      Eigen::Map<const VectorXi>(inliers.data(), inliers.size());
}

}  // namespace

void InlierThresholdOptions::Validate() const {
  if (max_epipolar_error_essential < 0.0 ||
      max_epipolar_error_fundamental < 0.0 ||
      max_epipolar_error_homography < 0.0 || min_angle_from_epipole_deg < 0.0 ||
      min_angle_from_epipole_deg > 180.0) {
    throw std::invalid_argument("invalid inlier threshold options");
  }
}

void ImagePairsInlierCount(const InlierThresholdOptions& options,
                           const colmap::Reconstruction& reconstruction,
                           const colmap::PoseGraph& graph,
                           MappingSidecars& sidecars) {
  options.Validate();
  sidecars.Validate(reconstruction);
  for (auto& [pair_id, pair] : sidecars.pairs) {
    pair.inlier_indices.resize(0);
    if (!graph.IsValid(pair_id)) {
      continue;
    }
    switch (pair.geometry.config) {
      case colmap::TwoViewGeometry::CALIBRATED:
        ScoreEssentialPair(
            options, reconstruction, graph, sidecars, pair_id, &pair);
        break;
      case colmap::TwoViewGeometry::UNCALIBRATED:
        ScoreFundamentalPair(options, reconstruction, pair_id, &pair);
        break;
      case colmap::TwoViewGeometry::PLANAR:
      case colmap::TwoViewGeometry::PANORAMIC:
      case colmap::TwoViewGeometry::PLANAR_OR_PANORAMIC:
        ScoreHomographyPair(options, reconstruction, pair_id, &pair);
        break;
      default:
        break;
    }
  }
}

}  // namespace vidmap
