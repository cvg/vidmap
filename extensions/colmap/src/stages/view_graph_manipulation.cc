#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/math.h"
#include "colmap/optim/loransac.h"
#include "colmap/optim/ransac.h"
#include "colmap/optim/support_measurement.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vidmap_native/conversion.h"
#include "vidmap_native/view_graph.h"
#include <Eigen/Eigenvalues>

namespace vidmap {
namespace {

constexpr double kEpsilon = 1e-12;

colmap::TwoViewGeometry GeometryWithoutStoredInliers(const PairRecord& pair) {
  colmap::TwoViewGeometry geometry = ToColmapGeometry(pair);
  // PairRecord keeps all matches separate from the selected inlier rows.
  geometry.inlier_matches.clear();
  return geometry;
}

// Minimal 2-point estimator for unit relative translation t_21 given known
// relative rotation R_21. Each correspondence provides the rotated unit
// bearing X_t = b1_rot = R_21 * b1 in camera 2's frame and the unit bearing
// Y_t = b2 in camera 2's frame.
class KnownRotationTranslationEstimator {
 public:
  using X_t = Eigen::Vector3d;
  using Y_t = Eigen::Vector3d;
  using M_t = Eigen::Vector3d;

  static const int kMinNumSamples = 2;

  static bool HasPositiveDepth(const Eigen::Vector3d& b1_rot,
                               const Eigen::Vector3d& b2,
                               const Eigen::Vector3d& t_cand) {
    const double dot_b = b1_rot.dot(b2);
    const double t_b1 = b1_rot.dot(t_cand);
    const double t_b2 = b2.dot(t_cand);
    // Positive depth in both cameras:
    // lambda_1 ~ dot_b * t_b2 - t_b1 > 0, lambda_2 ~ t_b2 - dot_b * t_b1 > 0
    return (dot_b * t_b2 - t_b1 > 0.0) && (t_b2 - dot_b * t_b1 > 0.0);
  }

  void Estimate(const std::vector<X_t>& b1_rot,
                const std::vector<Y_t>& b2,
                std::vector<M_t>* models) const {
    THROW_CHECK_EQ(b1_rot.size(), kMinNumSamples);
    THROW_CHECK_EQ(b2.size(), kMinNumSamples);
    THROW_CHECK_NOTNULL(models);
    models->clear();

    const Eigen::Vector3d n0 = b2[0].cross(b1_rot[0]);
    const Eigen::Vector3d n1 = b2[1].cross(b1_rot[1]);
    Eigen::Vector3d t_base = n0.cross(n1);
    const double t_norm = t_base.norm();
    if (t_norm < 1e-8) {
      return;
    }
    t_base /= t_norm;

    for (const double sign : {1.0, -1.0}) {
      const Eigen::Vector3d t_cand = sign * t_base;
      if (HasPositiveDepth(b1_rot[0], b2[0], t_cand) &&
          HasPositiveDepth(b1_rot[1], b2[1], t_cand)) {
        models->push_back(t_cand);
      }
    }
  }

  void Residuals(const std::vector<X_t>& b1_rot,
                 const std::vector<Y_t>& b2,
                 const M_t& t_cand,
                 std::vector<double>* residuals) const {
    THROW_CHECK_EQ(b1_rot.size(), b2.size());
    THROW_CHECK_NOTNULL(residuals);
    const std::size_t num_samples = b1_rot.size();
    residuals->resize(num_samples);
    for (std::size_t k = 0; k < num_samples; ++k) {
      const double ep_norm_sq = t_cand.cross(b1_rot[k]).squaredNorm();
      if (ep_norm_sq < 1e-8 || !HasPositiveDepth(b1_rot[k], b2[k], t_cand)) {
        (*residuals)[k] = std::numeric_limits<double>::max();
        continue;
      }
      const double normal_dot_t = b2[k].cross(b1_rot[k]).dot(t_cand);
      (*residuals)[k] = (normal_dot_t * normal_dot_t) / ep_norm_sq;
    }
  }
};

// Non-minimal least-squares estimator for unit relative translation t_21 given
// known relative rotation R_21, minimizing the sum of squared projections onto
// the unit epipolar plane normals of the inliers.
class KnownRotationTranslationLocalEstimator
    : public KnownRotationTranslationEstimator {
 public:
  static const int kMinNumSamples = 2;

  void Estimate(const std::vector<X_t>& b1_rot,
                const std::vector<Y_t>& b2,
                std::vector<M_t>* models) const {
    THROW_CHECK_EQ(b1_rot.size(), b2.size());
    THROW_CHECK_NOTNULL(models);
    models->clear();
    if (b1_rot.size() < kMinNumSamples) {
      return;
    }

    Eigen::Matrix3d normal_cov = Eigen::Matrix3d::Zero();
    int num_valid = 0;
    for (std::size_t i = 0; i < b1_rot.size(); ++i) {
      const Eigen::Vector3d n = b2[i].cross(b1_rot[i]);
      const double n_norm = n.norm();
      if (n_norm > 1e-6) {
        const Eigen::Vector3d n_unit = n / n_norm;
        normal_cov.noalias() += n_unit * n_unit.transpose();
        ++num_valid;
      }
    }
    if (num_valid < kMinNumSamples) {
      return;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(normal_cov);
    if (eig.info() != Eigen::Success) {
      return;
    }
    const Eigen::Vector3d t_base = eig.eigenvectors().col(0);

    int pos_count = 0;
    int neg_count = 0;
    for (std::size_t i = 0; i < b1_rot.size(); ++i) {
      if (HasPositiveDepth(b1_rot[i], b2[i], t_base)) {
        ++pos_count;
      } else if (HasPositiveDepth(b1_rot[i], b2[i], -t_base)) {
        ++neg_count;
      }
    }
    if (pos_count >= neg_count && pos_count > 0) {
      models->push_back(t_base);
    } else if (neg_count > 0) {
      models->push_back(-t_base);
    }
  }
};

void CheckExcludedMatches(const std::vector<bool>& excluded_matches,
                          const Eigen::Index num_matches) {
  if (!excluded_matches.empty() &&
      static_cast<Eigen::Index>(excluded_matches.size()) != num_matches) {
    throw std::invalid_argument(
        "excluded_matches must be empty or have one entry per match");
  }
}

}  // namespace

std::vector<bool> FindMatchesExplainedByRelativePose(
    const ImageRecord& image1,
    const ImageRecord& image2,
    const PairRecord& pair,
    const double max_epipolar_angle_deg) {
  const Eigen::Index num_matches = pair.all_matches.rows();
  std::vector<bool> explained(num_matches, false);
  if (!pair.geometry.cam2_from_cam1.has_pose || image1.bearings.rows() == 0 ||
      image2.bearings.rows() == 0 || max_epipolar_angle_deg <= 0.0) {
    return explained;
  }
  const colmap::Rigid3d cam2_from_cam1 =
      ToColmapPose(pair.geometry.cam2_from_cam1);
  const Eigen::Matrix3d R_21 = cam2_from_cam1.rotation().toRotationMatrix();
  const double t_norm = cam2_from_cam1.translation().norm();
  const Eigen::Vector3d t_21 =
      t_norm > kEpsilon ? Eigen::Vector3d(cam2_from_cam1.translation() / t_norm)
                        : Eigen::Vector3d::Zero();
  const double sin_thres = std::sin(colmap::DegToRad(max_epipolar_angle_deg));
  const double max_residual = sin_thres * sin_thres;
  for (Eigen::Index k = 0; k < num_matches; ++k) {
    const std::uint32_t idx1 = pair.all_matches(k, 0);
    const std::uint32_t idx2 = pair.all_matches(k, 1);
    if (idx1 >= static_cast<std::uint32_t>(image1.bearings.rows()) ||
        idx2 >= static_cast<std::uint32_t>(image2.bearings.rows())) {
      continue;
    }
    const Eigen::Vector3d b1_rot = R_21 * image1.bearings.row(idx1).transpose();
    const Eigen::Vector3d b2 = image2.bearings.row(idx2).transpose();
    double residual = std::numeric_limits<double>::max();
    const double ep_norm_sq = t_21.cross(b1_rot).squaredNorm();
    if (ep_norm_sq >= 1e-8) {
      // Epipolar angle, as in the salvage estimator (no cheirality test, so
      // that any match the rejected motion can account for is excluded).
      const double normal_dot_t = b2.cross(b1_rot).dot(t_21);
      residual = (normal_dot_t * normal_dot_t) / ep_norm_sq;
    } else if (t_norm <= kEpsilon) {
      // Rotation-only (panoramic) model: angle between b2 and R_21 * b1.
      residual = b2.cross(b1_rot).squaredNorm();
    }
    explained[k] = residual <= max_residual;
  }
  return explained;
}

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

bool TrySalvagePairTranslationWithKnownRotation(
    const ImageRecord& image1,
    const ImageRecord& image2,
    const double max_epipolar_angle_deg,
    const int min_inliers,
    const double min_inlier_ratio,
    PairRecord* pair,
    const std::vector<bool>& excluded_matches) {
  if (pair == nullptr) {
    throw std::invalid_argument("pair must not be null");
  }
  const Eigen::Index num_matches = pair->all_matches.rows();
  CheckExcludedMatches(excluded_matches, num_matches);
  if (!image1.pose.has_pose || !image2.pose.has_pose ||
      image1.bearings.rows() == 0 || image2.bearings.rows() == 0 ||
      max_epipolar_angle_deg <= 0.0 || num_matches < std::max(2, min_inliers)) {
    return false;
  }

  const Eigen::Quaterniond q_21 =
      (ToColmapPose(image2.pose).rotation() *
       ToColmapPose(image1.pose).rotation().inverse())
          .normalized();
  const Eigen::Matrix3d R_21 = q_21.toRotationMatrix();

  std::vector<Eigen::Vector3d> b1_rot;
  std::vector<Eigen::Vector3d> b2_vec;
  std::vector<int> valid_match_indices;
  b1_rot.reserve(num_matches);
  b2_vec.reserve(num_matches);
  valid_match_indices.reserve(num_matches);

  for (Eigen::Index k = 0; k < num_matches; ++k) {
    if (!excluded_matches.empty() && excluded_matches[k]) {
      continue;
    }
    const std::uint32_t idx1 = pair->all_matches(k, 0);
    const std::uint32_t idx2 = pair->all_matches(k, 1);
    if (idx1 >= static_cast<std::uint32_t>(image1.bearings.rows()) ||
        idx2 >= static_cast<std::uint32_t>(image2.bearings.rows())) {
      continue;
    }
    const Eigen::Vector3d b1_r = R_21 * image1.bearings.row(idx1).transpose();
    const Eigen::Vector3d b2 = image2.bearings.row(idx2).transpose();
    if (b2.cross(b1_r).squaredNorm() > 1e-8) {
      b1_rot.push_back(b1_r);
      b2_vec.push_back(b2);
      valid_match_indices.push_back(static_cast<int>(k));
    }
  }

  const int required_inliers =
      std::max(std::max(2, min_inliers),
               static_cast<int>(std::ceil(min_inlier_ratio *
                                          static_cast<double>(num_matches))));
  if (static_cast<int>(valid_match_indices.size()) < required_inliers) {
    return false;
  }

  colmap::RANSACOptions ransac_options;
  ransac_options.max_error = std::sin(colmap::DegToRad(max_epipolar_angle_deg));
  ransac_options.min_inlier_ratio = std::clamp(min_inlier_ratio, 0.01, 1.0);
  ransac_options.confidence = 0.9999;
  ransac_options.min_num_trials = 100;
  ransac_options.max_num_trials = 1000;

  colmap::LORANSAC<KnownRotationTranslationEstimator,
                   KnownRotationTranslationLocalEstimator,
                   colmap::MEstimatorSupportMeasurer>
      ransac(ransac_options);
  const auto report = ransac.Estimate(b1_rot, b2_vec);
  if (!report.success ||
      static_cast<int>(report.support.num_inliers) < required_inliers) {
    return false;
  }

  const double max_residual =
      ransac_options.max_error * ransac_options.max_error;
  Eigen::Vector3d best_t = report.model;
  std::vector<char> inlier_mask = report.inlier_mask;
  std::vector<double> residuals;
  std::vector<Eigen::Vector3d> inlier_b1_rot;
  std::vector<Eigen::Vector3d> inlier_b2;
  std::vector<Eigen::Vector3d> local_models;

  // Iteratively refine the translation direction via least-squares on the
  // inlier set to remove any residual minimal-sample bias.
  for (int iter = 0; iter < 2; ++iter) {
    inlier_b1_rot.clear();
    inlier_b2.clear();
    for (std::size_t i = 0; i < b1_rot.size(); ++i) {
      if (inlier_mask[i]) {
        inlier_b1_rot.push_back(b1_rot[i]);
        inlier_b2.push_back(b2_vec[i]);
      }
    }
    ransac.local_estimator.Estimate(inlier_b1_rot, inlier_b2, &local_models);
    if (local_models.empty()) {
      break;
    }
    const Eigen::Vector3d& refined_t = local_models.front();
    ransac.estimator.Residuals(b1_rot, b2_vec, refined_t, &residuals);
    int refined_inliers = 0;
    for (const double r : residuals) {
      if (r <= max_residual) {
        ++refined_inliers;
      }
    }
    if (refined_inliers < required_inliers) {
      break;
    }
    best_t = refined_t;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
      inlier_mask[i] = residuals[i] <= max_residual;
    }
  }

  std::vector<int> best_inliers;
  best_inliers.reserve(inlier_mask.size());
  for (std::size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      best_inliers.push_back(valid_match_indices[i]);
    }
  }
  if (static_cast<int>(best_inliers.size()) < required_inliers) {
    return false;
  }

  pair->is_valid = true;
  pair->inlier_indices =
      Eigen::Map<const VectorXi>(best_inliers.data(), best_inliers.size());
  pair->geometry.cam2_from_cam1 = FromColmapPose(colmap::Rigid3d(q_21, best_t));
  return true;
}

}  // namespace vidmap
