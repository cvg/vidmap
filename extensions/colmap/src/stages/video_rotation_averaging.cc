// Video-aware rotation averaging over VidMap-owned value records.
#include "vidmap_native/video_rotation_averaging.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/imu_preintegration.h"
#include "colmap/estimators/imu_preintegration_cost.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/connected_components.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/math/spanning_tree.h"
#include "colmap/util/threading.h"

#if __has_include("colmap/util/hash_containers.h")
#include "colmap/util/hash_containers.h"
#define VIDMAP_COLMAP_HAS_FLAT_HASH_SET 1
#else
#define VIDMAP_COLMAP_HAS_FLAT_HASH_SET 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vidmap_native/conversion.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vidmap {
namespace {

constexpr float kLCPenalty = 1e9f;
constexpr float kImuSpanningTreeWeight = 1e6f;

#if VIDMAP_COLMAP_HAS_FLAT_HASH_SET
using ConnectedComponentFrameSet = colmap::FlatHashSet<FrameId>;
#else
using ConnectedComponentFrameSet = std::unordered_set<FrameId>;
#endif

struct RelativeRotationError {
  explicit RelativeRotationError(const Eigen::Vector3d& rel_rot_aa,
                                 const double weight = 1.0)
      : rel_rot_aa_(rel_rot_aa), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const r1_aa,
                  const T* const r2_aa,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3> R1, R2, R_rel;
    ceres::AngleAxisToRotationMatrix(r1_aa, R1.data());
    ceres::AngleAxisToRotationMatrix(r2_aa, R2.data());
    Eigen::Matrix<T, 3, 1> rel_aa_t = rel_rot_aa_.cast<T>();
    ceres::AngleAxisToRotationMatrix(rel_aa_t.data(), R_rel.data());
    Eigen::Matrix<T, 3, 3> R_err = R2.transpose() * R_rel * R1;
    ceres::RotationMatrixToAngleAxis(R_err.data(), residuals);
    if (weight_ != 1.0) {
      residuals[0] *= T(weight_);
      residuals[1] *= T(weight_);
      residuals[2] *= T(weight_);
    }
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& rel_rot_aa,
                                     const double weight = 1.0) {
    return new ceres::AutoDiffCostFunction<RelativeRotationError, 3, 3, 3>(
        new RelativeRotationError(rel_rot_aa, weight));
  }

  const Eigen::Vector3d rel_rot_aa_;
  const double weight_;
};

template <typename Id>
std::vector<Id> HashMapOrderPasses(std::vector<Id> ids, int num_passes) {
  num_passes = std::max(1, num_passes);
  for (int pass = 0; pass < num_passes; ++pass) {
    std::unordered_map<Id, char> values;
    values.reserve(ids.size());
    for (const Id id : ids) values.emplace(id, 0);
    ids.clear();
    ids.reserve(values.size());
    for (const auto& [id, unused] : values) ids.push_back(id);
  }
  return ids;
}

template <typename Id>
std::vector<Id> SortedHashMapOrderPasses(std::vector<Id> ids, int num_passes) {
  std::sort(ids.begin(), ids.end());
  return HashMapOrderPasses(std::move(ids), num_passes);
}

void ValidateImageMapOrder(const MappingProblem& problem,
                           const std::vector<ImageId>& image_map_order) {
  if (image_map_order.size() != problem.NumImages()) {
    throw std::invalid_argument(
        "image map order must contain every mapping image");
  }
  std::unordered_set<ImageId> image_ids;
  image_ids.reserve(image_map_order.size());
  std::unordered_set<FrameId> frame_ids;
  frame_ids.reserve(image_map_order.size());
  for (const ImageId image_id : image_map_order) {
    const ImageRecord& image = problem.Image(image_id);
    if (!image_ids.insert(image_id).second) {
      throw std::invalid_argument("duplicate image in image map order");
    }
    if (!frame_ids.insert(image.frame_id).second) {
      throw std::invalid_argument(
          "video rotation averaging requires one image per frame");
    }
  }
}

void ValidatePairMapOrder(const MappingProblem& problem,
                          const std::vector<PairId>& pair_map_order) {
  if (pair_map_order.size() != problem.NumPairs()) {
    throw std::invalid_argument("pair map order must contain every pair");
  }
  std::unordered_set<PairId> pair_ids;
  pair_ids.reserve(pair_map_order.size());
  for (const PairId pair_id : pair_map_order) {
    problem.Pair(pair_id);
    if (!pair_ids.insert(pair_id).second) {
      throw std::invalid_argument("duplicate pair in pair map order");
    }
  }
}

void ValidateImuInputs(const MappingProblem& problem,
                       const std::vector<ImuEdgeRecord>& imu_edges,
                       const std::vector<ImuStateRecord>& imu_states) {
  for (const ImuEdgeRecord& edge : imu_edges) {
    edge.Validate();
    problem.Image(edge.image_id1);
    problem.Image(edge.image_id2);
  }
  std::unordered_set<ImageId> seen_states;
  for (const ImuStateRecord& state : imu_states) {
    state.Validate();
    problem.Image(state.image_id);
    if (!seen_states.insert(state.image_id).second) {
      throw std::invalid_argument("duplicate RA IMU state record");
    }
  }
}

bool IsPoseGraphPair(const PairRecord& pair) {
  return pair.is_valid && pair.geometry.cam2_from_cam1.has_pose;
}

bool IsTrackingPair(const PairRecord& pair) {
  if (pair.inlier_indices.size() == 0) return false;
  std::size_t loop_closure_count = 0;
  for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
    const int row = pair.inlier_indices[index];
    if (row >= 0 && row < pair.are_loop_closure.size() &&
        pair.are_loop_closure[row] != 0) {
      ++loop_closure_count;
    }
  }
  return static_cast<std::size_t>(pair.inlier_indices.size()) -
             loop_closure_count >=
         loop_closure_count;
}

std::unordered_set<ImageId> ComputeLargestConnectedComponentImageIds(
    const MappingProblem& problem,
    const std::vector<ImageId>& image_map_order,
    const std::vector<PairId>& pair_map_order,
    bool filter_unregistered,
    const std::unordered_set<PairId>& excluded_pair_ids = {},
    const std::vector<ImuEdgeRecord>& imu_edges = {},
    const std::unordered_set<ImageId>* allowed_imu_images = nullptr) {
  ConnectedComponentFrameSet nodes;
  std::vector<std::pair<FrameId, FrameId>> edges;
  for (const PairId pair_id : pair_map_order) {
    if (excluded_pair_ids.count(pair_id) != 0) continue;
    const PairRecord& pair = problem.Pair(pair_id);
    if (!IsPoseGraphPair(pair)) continue;
    const ImageRecord& image1 = problem.Image(pair.image_id1);
    const ImageRecord& image2 = problem.Image(pair.image_id2);
    if (filter_unregistered &&
        (!image1.pose.has_pose || !image2.pose.has_pose)) {
      continue;
    }
    nodes.insert(image1.frame_id);
    nodes.insert(image2.frame_id);
    edges.emplace_back(image1.frame_id, image2.frame_id);
  }
  for (const ImuEdgeRecord& imu_edge : imu_edges) {
    if (allowed_imu_images != nullptr &&
        (allowed_imu_images->count(imu_edge.image_id1) == 0 ||
         allowed_imu_images->count(imu_edge.image_id2) == 0)) {
      continue;
    }
    const ImageRecord& image1 = problem.Image(imu_edge.image_id1);
    const ImageRecord& image2 = problem.Image(imu_edge.image_id2);
    if (filter_unregistered &&
        (!image1.pose.has_pose || !image2.pose.has_pose)) {
      continue;
    }
    nodes.insert(image1.frame_id);
    nodes.insert(image2.frame_id);
    edges.emplace_back(image1.frame_id, image2.frame_id);
  }
  if (nodes.empty()) return {};

  const std::vector<FrameId> largest_component =
      colmap::FindLargestConnectedComponent(nodes, edges);
  const std::unordered_set<FrameId> active_frames(largest_component.begin(),
                                                  largest_component.end());
  std::unordered_set<ImageId> active_images;
  active_images.reserve(active_frames.size());
  for (const ImageId image_id : image_map_order) {
    if (active_frames.count(problem.Image(image_id).frame_id) != 0) {
      active_images.insert(image_id);
    }
  }
  return active_images;
}

std::vector<ImageId> OrderedActiveImages(
    const std::unordered_set<ImageId>& active_images, int image_order_passes) {
  std::vector<ImageId> ordered(active_images.begin(), active_images.end());
  return SortedHashMapOrderPasses(std::move(ordered), image_order_passes);
}

Eigen::Quaterniond ComputeBiasCorrectedImuCameraRelativeRotation(
    const ImuEdgeRecord& edge,
    const Eigen::Quaterniond& q_IC,
    const Eigen::Vector3d& bg) {
  const Eigen::Quaterniond q_CI = q_IC.conjugate();
  const Eigen::Vector3d dbg = bg - edge.data.biases.head<3>();
  const Eigen::Vector3d omega_bias = edge.data.dR_dbg * dbg;
  Eigen::Quaterniond dq_bias = Eigen::Quaterniond::Identity();
  const double angle = omega_bias.norm();
  if (angle > 1e-12) {
    dq_bias = Eigen::Quaterniond(Eigen::AngleAxisd(angle, omega_bias / angle));
  }
  const Eigen::Quaterniond delta_R_corr =
      (edge.data.delta_R * dq_bias).normalized();
  return (edge.q_iori_2_xyzw * q_CI * delta_R_corr * q_IC *
          edge.q_iori_1_xyzw.conjugate())
      .normalized();
}

// Step 1 of I-RA: Closed-form coordinate-wise median seed for the global
// gyroscope bias across consecutive pairs that have both an IMU edge and a
// valid visual relative rotation.
bool EstimateInitialGyroBiasMedian(
    const MappingProblem& problem,
    const std::vector<PairId>& pair_map_order,
    const std::unordered_set<ImageId>& active_images,
    const std::unordered_set<PairId>& excluded_pair_ids,
    const std::vector<ImuEdgeRecord>& imu_edges,
    const Eigen::Quaterniond& q_IC,
    Eigen::Vector3d* bg_out) {
  std::unordered_map<PairId, const PairRecord*> valid_pairs;
  valid_pairs.reserve(pair_map_order.size());
  for (const PairId pair_id : pair_map_order) {
    if (excluded_pair_ids.count(pair_id) != 0) continue;
    const PairRecord& pair = problem.Pair(pair_id);
    if (!IsPoseGraphPair(pair)) continue;
    if (active_images.count(pair.image_id1) == 0 ||
        active_images.count(pair.image_id2) == 0) {
      continue;
    }
    valid_pairs.emplace(CanonicalPairId(pair.image_id1, pair.image_id2), &pair);
  }

  const Eigen::Quaterniond q_CI = q_IC.conjugate();
  std::vector<Eigen::Vector3d> samples;
  samples.reserve(imu_edges.size());
  for (const ImuEdgeRecord& edge : imu_edges) {
    if (active_images.count(edge.image_id1) == 0 ||
        active_images.count(edge.image_id2) == 0 || edge.data.delta_t <= 1e-4) {
      continue;
    }
    const auto pair_it =
        valid_pairs.find(CanonicalPairId(edge.image_id1, edge.image_id2));
    if (pair_it == valid_pairs.end()) continue;
    const PairRecord& pair = *pair_it->second;

    Eigen::Quaterniond q_c2_from_c1 =
        ToColmapPose(pair.geometry.cam2_from_cam1).rotation().normalized();
    if (pair.image_id1 == edge.image_id2 && pair.image_id2 == edge.image_id1) {
      q_c2_from_c1 = q_c2_from_c1.conjugate();
    }

    const Eigen::Quaterniond delta_R_vis =
        (q_IC * edge.q_iori_2_xyzw.conjugate() * q_c2_from_c1 *
         edge.q_iori_1_xyzw * q_CI)
            .normalized();
    const Eigen::Quaterniond q_diff =
        (edge.data.delta_R.conjugate() * delta_R_vis).normalized();
    const Eigen::Vector3d theta_err =
        colmap::RotationMatrixToAngleAxis(q_diff.toRotationMatrix());
    const Eigen::Vector3d dbg =
        edge.data.dR_dbg.colPivHouseholderQr().solve(theta_err);
    const Eigen::Vector3d bg_sample = edge.data.biases.head<3>() + dbg;
    if (bg_sample.allFinite() && bg_sample.norm() < 2.0) {
      samples.push_back(bg_sample);
    }
  }

  if (samples.empty()) {
    return false;
  }

  Eigen::Vector3d median_bg = Eigen::Vector3d::Zero();
  std::vector<double> coords(samples.size());
  for (int axis = 0; axis < 3; ++axis) {
    for (std::size_t i = 0; i < samples.size(); ++i) {
      coords[i] = samples[i](axis);
    }
    std::sort(coords.begin(), coords.end());
    const std::size_t mid = coords.size() / 2;
    if (coords.size() % 2 == 1) {
      median_bg(axis) = coords[mid];
    } else {
      median_bg(axis) = 0.5 * (coords[mid - 1] + coords[mid]);
    }
  }
  *bg_out = median_bg;
  return true;
}

void ApplyUniformGyroBiasAndReintegrate(
    const Eigen::Vector3d& bg,
    std::map<ImageId, Eigen::Matrix<double, 9, 1>>* imu_state_params,
    std::vector<ImuEdgeRecord>* mutable_imu_edges) {
  for (auto& [image_id, state] : *imu_state_params) {
    state.segment<3>(3) = bg;
  }
  for (ImuEdgeRecord& edge : *mutable_imu_edges) {
    if (edge.integrator != nullptr) {
      Eigen::Vector6d biases = edge.data.biases;
      biases.head<3>() = bg;
      edge.integrator->Reintegrate(biases);
      edge.integrator->Update(&edge.data);
    }
  }
}

void InitializeFromMaximumSpanningTree(
    const VideoRotationAveragingOptions& options,
    const std::vector<PairId>& pair_map_order,
    const std::unordered_set<ImageId>& active_images,
    MappingProblem* problem,
    const std::unordered_set<PairId>& excluded_pair_ids = {},
    const std::vector<ImuEdgeRecord>& imu_edges = {},
    const std::map<ImageId, Eigen::Matrix<double, 9, 1>>& imu_state_params = {},
    const Eigen::Quaterniond& q_IC = Eigen::Quaterniond::Identity()) {
  const std::vector<ImageId> ordered_images =
      OrderedActiveImages(active_images, options.image_order_passes);
  std::unordered_map<ImageId, int> image_to_index;
  image_to_index.reserve(ordered_images.size());
  for (std::size_t index = 0; index < ordered_images.size(); ++index) {
    image_to_index.emplace(ordered_images[index], static_cast<int>(index));
  }

  std::vector<std::pair<int, int>> edges;
  std::vector<float> weights;
  for (const PairId pair_id : pair_map_order) {
    if (excluded_pair_ids.count(pair_id) != 0) continue;
    const PairRecord& pair = problem->Pair(pair_id);
    if (!IsPoseGraphPair(pair)) continue;
    const auto image1_it = image_to_index.find(pair.image_id1);
    const auto image2_it = image_to_index.find(pair.image_id2);
    if (image1_it == image_to_index.end() ||
        image2_it == image_to_index.end()) {
      continue;
    }
    edges.emplace_back(image1_it->second, image2_it->second);
    float weight = static_cast<float>(pair.inlier_indices.size());
    if (!IsTrackingPair(pair)) {
      weight -= kLCPenalty;
    }
    weights.push_back(weight);
  }

  std::unordered_map<PairId, std::pair<ImageId, Eigen::Quaterniond>>
      imu_rel_rotations;
  for (const ImuEdgeRecord& imu_edge : imu_edges) {
    const auto image1_it = image_to_index.find(imu_edge.image_id1);
    const auto image2_it = image_to_index.find(imu_edge.image_id2);
    if (image1_it == image_to_index.end() ||
        image2_it == image_to_index.end()) {
      continue;
    }
    edges.emplace_back(image1_it->second, image2_it->second);
    weights.push_back(kImuSpanningTreeWeight);
    const Eigen::Vector3d bg =
        imu_state_params.count(imu_edge.image_id1) != 0
            ? imu_state_params.at(imu_edge.image_id1).segment<3>(3).eval()
            : imu_edge.data.biases.head<3>().eval();
    imu_rel_rotations[CanonicalPairId(imu_edge.image_id1, imu_edge.image_id2)] =
        std::make_pair(
            imu_edge.image_id1,
            ComputeBiasCorrectedImuCameraRelativeRotation(imu_edge, q_IC, bg));
  }

  const colmap::SpanningTree tree =
      colmap::ComputeMaximumSpanningTree(ordered_images.size(), edges, weights);
  if (!tree.IsValid()) {
    throw std::runtime_error("failed to build rotation spanning tree");
  }

  std::vector<std::vector<int>> children(ordered_images.size());
  for (std::size_t child = 0; child < tree.parents.size(); ++child) {
    if (static_cast<int>(child) == tree.root || tree.parents[child] < 0) {
      continue;
    }
    children[tree.parents[child]].push_back(static_cast<int>(child));
  }

  std::vector<colmap::Rigid3d> cam_from_world(ordered_images.size());
  const ImageRecord& root_image = problem->Image(ordered_images[tree.root]);
  if (root_image.pose.has_pose) {
    cam_from_world[tree.root] = ToColmapPose(root_image.pose);
  }
  std::queue<int> queue;
  queue.push(tree.root);
  while (!queue.empty()) {
    const int parent_index = queue.front();
    queue.pop();
    for (const int child_index : children[parent_index]) {
      queue.push(child_index);
      const ImageId child_id = ordered_images[child_index];
      const ImageId parent_id = ordered_images[parent_index];
      const PairId canonical_pair_id = CanonicalPairId(child_id, parent_id);
      const auto imu_it = imu_rel_rotations.find(canonical_pair_id);
      if (imu_it != imu_rel_rotations.end()) {
        const ImageId id1 = imu_it->second.first;
        const Eigen::Quaterniond& q_c2_from_c1 = imu_it->second.second;
        if (parent_id == id1) {
          cam_from_world[child_index].rotation() =
              (q_c2_from_c1 * cam_from_world[parent_index].rotation())
                  .normalized();
        } else {
          cam_from_world[child_index].rotation() =
              (q_c2_from_c1.conjugate() *
               cam_from_world[parent_index].rotation())
                  .normalized();
        }
        continue;
      }

      const PairRecord& pair = problem->Pair(canonical_pair_id);
      const colmap::Rigid3d relative_pose =
          ToColmapPose(pair.geometry.cam2_from_cam1);
      if (pair.image_id1 == child_id && pair.image_id2 == parent_id) {
        cam_from_world[child_index].rotation() =
            (colmap::Inverse(relative_pose) * cam_from_world[parent_index])
                .rotation();
      } else if (pair.image_id2 == child_id && pair.image_id1 == parent_id) {
        cam_from_world[child_index].rotation() =
            (relative_pose * cam_from_world[parent_index]).rotation();
      } else {
        throw std::logic_error("pair orientation does not match pair images");
      }
    }
  }

  for (std::size_t index = 0; index < ordered_images.size(); ++index) {
    ImageRecord image = problem->Image(ordered_images[index]);
    const Eigen::Vector3d translation = image.pose.translation;
    image.pose = FromColmapPose(
        colmap::Rigid3d(cam_from_world[index].rotation(), translation));
    problem->UpdateImage(image);
  }
}

std::unordered_set<PairId> FindRotationOutlierPairs(
    const MappingProblem& problem,
    const std::vector<PairId>& pair_map_order,
    const std::unordered_set<ImageId>& active_images,
    double max_rotation_error_deg) {
  std::unordered_set<PairId> outlier_pairs;
  if (max_rotation_error_deg <= 0.0) return outlier_pairs;
  const double max_rotation_error = colmap::DegToRad(max_rotation_error_deg);
  for (const PairId pair_id : pair_map_order) {
    const PairRecord& pair = problem.Pair(pair_id);
    if (!IsPoseGraphPair(pair)) continue;
    if (active_images.count(pair.image_id1) == 0 ||
        active_images.count(pair.image_id2) == 0) {
      continue;
    }
    const ImageRecord& image1 = problem.Image(pair.image_id1);
    const ImageRecord& image2 = problem.Image(pair.image_id2);
    if (!image1.pose.has_pose || !image2.pose.has_pose) continue;
    const Eigen::Quaterniond estimated_relative_rotation =
        ToColmapPose(image2.pose).rotation() *
        ToColmapPose(image1.pose).rotation().inverse();
    if (estimated_relative_rotation.angularDistance(
            ToColmapPose(pair.geometry.cam2_from_cam1).rotation()) >
        max_rotation_error) {
      outlier_pairs.insert(pair_id);
    }
  }
  return outlier_pairs;
}

// Option RP-B: Attempt to salvage a rotation-rejected pair by fixing its
// relative rotation to R_21 = R_cw_2 * R_cw_1^T and running 2-point epipolar
// translation RANSAC over the raw matches.
bool TrySalvagePairTranslationWithKnownRotation(
    const VideoRotationAveragingOptions& options,
    const ImageRecord& image1,
    const ImageRecord& image2,
    PairRecord* pair) {
  const Eigen::Index num_matches = pair->all_matches.rows();
  if (!image1.pose.has_pose || !image2.pose.has_pose ||
      image1.bearings.rows() == 0 || image2.bearings.rows() == 0 ||
      num_matches < options.salvage_min_inliers) {
    return false;
  }

  const Eigen::Quaterniond q_21 =
      (ToColmapPose(image2.pose).rotation() *
       ToColmapPose(image1.pose).rotation().inverse())
          .normalized();
  const Eigen::Matrix3d R_21 = q_21.toRotationMatrix();

  std::vector<Eigen::Vector3d> b1_rot(num_matches);
  std::vector<Eigen::Vector3d> b2_vec(num_matches);
  std::vector<Eigen::Vector3d> normals(num_matches);
  std::vector<int> valid_indices;
  valid_indices.reserve(num_matches);

  for (Eigen::Index k = 0; k < num_matches; ++k) {
    const std::uint32_t idx1 = pair->all_matches(k, 0);
    const std::uint32_t idx2 = pair->all_matches(k, 1);
    if (idx1 >= static_cast<std::uint32_t>(image1.bearings.rows()) ||
        idx2 >= static_cast<std::uint32_t>(image2.bearings.rows())) {
      continue;
    }
    b1_rot[k] = R_21 * image1.bearings.row(idx1).transpose();
    b2_vec[k] = image2.bearings.row(idx2).transpose();
    normals[k] = b2_vec[k].cross(b1_rot[k]);
    if (normals[k].norm() > 1e-4) {
      valid_indices.push_back(static_cast<int>(k));
    }
  }
  if (static_cast<int>(valid_indices.size()) <
      std::max(2, options.salvage_min_inliers)) {
    return false;
  }

  const double sin_thres =
      std::sin(colmap::DegToRad(options.salvage_epipolar_angle_thres_deg));
  const double sin_thres_sq = sin_thres * sin_thres;
  std::vector<int> best_inliers;
  double best_score = -1.0;
  Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
  constexpr int kNumRansacTrials = 100;

  for (int trial = 0; trial < kNumRansacTrials; ++trial) {
    const int i1 = colmap::RandomUniformInteger(
        0, static_cast<int>(valid_indices.size()) - 1);
    int i2 = colmap::RandomUniformInteger(
        0, static_cast<int>(valid_indices.size()) - 2);
    if (i2 >= i1) ++i2;

    Eigen::Vector3d t_base =
        normals[valid_indices[i1]].cross(normals[valid_indices[i2]]);
    const double t_norm = t_base.norm();
    if (t_norm < 1e-8) continue;
    t_base /= t_norm;

    for (const double sign : {1.0, -1.0}) {
      const Eigen::Vector3d t_cand = sign * t_base;
      std::vector<int> inliers;
      inliers.reserve(valid_indices.size());
      double score = 0.0;
      for (const int k : valid_indices) {
        const double ep_norm = t_cand.cross(b1_rot[k]).norm();
        if (ep_norm < 1e-4) continue;
        const double sin_err = std::abs(normals[k].dot(t_cand)) / ep_norm;
        if (sin_err >= sin_thres) continue;
        const double dot_b = b1_rot[k].dot(b2_vec[k]);
        const double t_b1 = b1_rot[k].dot(t_cand);
        const double t_b2 = b2_vec[k].dot(t_cand);
        // Positive depth in both cameras:
        // lambda_1 ~ dot_b * t_b2 - t_b1 > 0, lambda_2 ~ t_b2 - dot_b * t_b1 >
        // 0
        if (dot_b * t_b2 - t_b1 <= 0.0 || t_b2 - dot_b * t_b1 <= 0.0) {
          continue;
        }
        inliers.push_back(k);
        score += 1.0 - (sin_err * sin_err) / sin_thres_sq;
      }
      if (score > best_score) {
        best_score = score;
        best_inliers = std::move(inliers);
        best_t = t_cand;
      }
    }
  }

  const int required_inliers =
      std::max(options.salvage_min_inliers,
               static_cast<int>(std::ceil(options.salvage_min_inlier_ratio *
                                          static_cast<double>(num_matches))));
  if (static_cast<int>(best_inliers.size()) < required_inliers) {
    return false;
  }

  pair->is_valid = true;
  pair->inlier_indices.resize(static_cast<Eigen::Index>(best_inliers.size()));
  for (std::size_t idx = 0; idx < best_inliers.size(); ++idx) {
    pair->inlier_indices(static_cast<Eigen::Index>(idx)) = best_inliers[idx];
  }
  pair->geometry.cam2_from_cam1 = FromColmapPose(colmap::Rigid3d(q_21, best_t));
  return true;
}

bool SolveRotationAveragingCeresPass(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImageId>& parameter_image_order,
    const std::vector<PairId>& pair_map_order,
    const std::unordered_set<PairId>& excluded_pair_ids,
    const bool has_imu,
    const bool optimize_gyro_bias,
    const Eigen::Quaterniond& q_IC,
    const Eigen::Vector3d& effective_gyro_bias_prior,
    std::vector<ImuEdgeRecord>* mutable_imu_edges,
    std::map<ImageId, Eigen::Matrix<double, 9, 1>>* imu_state_params,
    MappingProblem* problem) {
  const ImageId fixed_image_id = parameter_image_order.front();

  std::unordered_map<ImageId, int> image_to_parameter_index;
  image_to_parameter_index.reserve(parameter_image_order.size());
  Eigen::VectorXd rotations(3 * parameter_image_order.size());
  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    const ImageId image_id = parameter_image_order[index];
    image_to_parameter_index.emplace(image_id, 3 * static_cast<int>(index));
    const Eigen::AngleAxisd angle_axis(
        ToColmapPose(problem->Image(image_id).pose).rotation());
    rotations.segment<3>(3 * index) = angle_axis.angle() * angle_axis.axis();
  }

  if (options.random_seed >= 0) {
    colmap::SetPRNGSeed(static_cast<unsigned>(options.random_seed));
  }
  ceres::Problem ceres_problem;
  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    double* parameter = rotations.data() + 3 * index;
    ceres_problem.AddParameterBlock(parameter, 3);
    if (parameter_image_order[index] == fixed_image_id) {
      ceres_problem.SetParameterBlockConstant(parameter);
    }
  }

  const double vis_weight =
      has_imu ? (1.0 / colmap::DegToRad(options.visual_rotation_stddev_deg))
              : 1.0;
  const double imu_tracking_cauchy_scale =
      vis_weight * colmap::DegToRad(options.imu_tracking_cauchy_scale_deg);
  const double lc_cauchy_scale = vis_weight * options.video_lc_cauchy_scale;

  for (const PairId pair_id : pair_map_order) {
    if (excluded_pair_ids.count(pair_id) != 0) continue;
    const PairRecord& pair = problem->Pair(pair_id);
    if (!IsPoseGraphPair(pair)) continue;
    const auto image1_it = image_to_parameter_index.find(pair.image_id1);
    const auto image2_it = image_to_parameter_index.find(pair.image_id2);
    if (image1_it == image_to_parameter_index.end() ||
        image2_it == image_to_parameter_index.end()) {
      continue;
    }
    const bool is_tracking = IsTrackingPair(pair);
    if (options.skip_risky_lc_pairs && !is_tracking) continue;
    ceres::LossFunction* loss = nullptr;
    if (has_imu) {
      loss = new ceres::CauchyLoss(is_tracking ? imu_tracking_cauchy_scale
                                               : lc_cauchy_scale);
    } else {
      loss = is_tracking
                 ? static_cast<ceres::LossFunction*>(
                       new ceres::HuberLoss(options.video_tracking_huber_scale))
                 : static_cast<ceres::LossFunction*>(
                       new ceres::CauchyLoss(options.video_lc_cauchy_scale));
    }
    const Eigen::Vector3d relative_angle_axis =
        colmap::RotationMatrixToAngleAxis(
            ToColmapPose(pair.geometry.cam2_from_cam1)
                .rotation()
                .toRotationMatrix());
    ceres_problem.AddResidualBlock(
        RelativeRotationError::Create(relative_angle_axis, vis_weight),
        loss,
        rotations.data() + image1_it->second,
        rotations.data() + image2_it->second);
  }

  colmap::ImuReintegrationOptions reint_options;
  reint_options.reintegrate_angle_norm_thres =
      options.reintegrate_angle_norm_thres;
  colmap::ImuReintegrationCallback reint_callback(reint_options);
  bool has_reint = false;

  if (has_imu) {
    for (ImuEdgeRecord& edge : *mutable_imu_edges) {
      const auto image1_it = image_to_parameter_index.find(edge.image_id1);
      const auto image2_it = image_to_parameter_index.find(edge.image_id2);
      if (image1_it == image_to_parameter_index.end() ||
          image2_it == image_to_parameter_index.end()) {
        continue;
      }
      const Eigen::Matrix<double, 6, 6> sqrt_info_6x6 =
          colmap::ExtractRotationGyroBiasSqrtInformation(edge.data);
      ceres::CostFunction* cost =
          colmap::InertialRotationCostFunctor::Create(&edge.data,
                                                      q_IC,
                                                      sqrt_info_6x6,
                                                      edge.q_iori_1_xyzw,
                                                      edge.q_iori_2_xyzw);
      double* state1_ptr = imu_state_params->at(edge.image_id1).data();
      double* state2_ptr = imu_state_params->at(edge.image_id2).data();
      ceres_problem.AddResidualBlock(cost,
                                     nullptr,
                                     rotations.data() + image1_it->second,
                                     state1_ptr,
                                     rotations.data() + image2_it->second,
                                     state2_ptr);
      if (optimize_gyro_bias && edge.integrator != nullptr) {
        reint_callback.AddEdge(edge.integrator, &edge.data, state1_ptr);
        has_reint = true;
      }
    }

    bool first_imu_state = true;
    for (const ImageId image_id : parameter_image_order) {
      auto state_it = imu_state_params->find(image_id);
      if (state_it == imu_state_params->end()) continue;
      double* state_ptr = state_it->second.data();
      if (!ceres_problem.HasParameterBlock(state_ptr)) continue;

      if (!optimize_gyro_bias) {
        ceres_problem.SetParameterBlockConstant(state_ptr);
      } else {
        colmap::SetManifold(&ceres_problem,
                            state_ptr,
                            colmap::CreateImuStateGyroOnlyManifold());
        if (options.use_gyro_bias_prior &&
            (first_imu_state || options.apply_bias_prior_to_all_frames)) {
          ceres_problem.AddResidualBlock(
              colmap::BiasPriorCostFunctor<9>::CreateGyro(
                  effective_gyro_bias_prior, options.gyro_bias_prior_stddev),
              nullptr,
              state_ptr);
        }
      }
      first_imu_state = false;
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.num_threads =
      options.num_threads > 0
          ? options.num_threads
          : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  if (has_reint) {
    solver_options.callbacks.push_back(&reint_callback);
    solver_options.update_state_every_iteration = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &ceres_problem, &summary);
  if (!summary.IsSolutionUsable()) return false;

  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    ImageRecord image = problem->Image(parameter_image_order[index]);
    const Eigen::Matrix3d rotation =
        colmap::AngleAxisToRotationMatrix(rotations.segment<3>(3 * index));
    const Eigen::Vector3d translation = image.pose.translation;
    image.pose = FromColmapPose(
        colmap::Rigid3d(Eigen::Quaterniond(rotation), translation));
    problem->UpdateImage(image);
  }
  return true;
}

Eigen::Vector3d ComputeTelescopingInitialGravityDirection(
    const MappingProblem& problem,
    const std::unordered_set<ImageId>& active_images,
    const std::vector<ImuEdgeRecord>& imu_edges,
    const std::map<ImageId, Eigen::Matrix<double, 9, 1>>& imu_state_params,
    const Eigen::Quaterniond& q_IC) {
  const Eigen::Quaterniond q_CI = q_IC.conjugate();
  Eigen::Vector3d dv_telescoping = Eigen::Vector3d::Zero();
  for (const ImuEdgeRecord& edge : imu_edges) {
    if (active_images.count(edge.image_id1) == 0 ||
        active_images.count(edge.image_id2) == 0) {
      continue;
    }
    const ImageRecord& img1 = problem.Image(edge.image_id1);
    if (!img1.pose.has_pose) continue;
    const Eigen::Quaterniond q_cw_1 =
        ToColmapPose(img1.pose).rotation().normalized();
    const Eigen::Quaterniond q_wb_1 =
        (q_cw_1.conjugate() * edge.q_iori_1_xyzw * q_CI).normalized();
    const Eigen::Vector3d bg =
        imu_state_params.count(edge.image_id1) != 0
            ? imu_state_params.at(edge.image_id1).segment<3>(3).eval()
            : edge.data.biases.head<3>().eval();
    const Eigen::Vector3d dbg = bg - edge.data.biases.head<3>();
    const Eigen::Vector3d dv_corr = edge.data.delta_v + edge.data.dv_dbg * dbg;
    dv_telescoping += q_wb_1 * dv_corr;
  }
  if (dv_telescoping.norm() > 1e-6) {
    return (-dv_telescoping).normalized();
  }
  return Eigen::Vector3d(0.0, 0.0, -1.0);
}

}  // namespace

void VideoRotationAveragingOptions::Validate() const {
  imu_from_cam.Validate();
  if (random_seed < -1 || image_order_passes < 0 ||
      !std::isfinite(max_rotation_error_deg) || max_rotation_error_deg < 0.0 ||
      !std::isfinite(video_tracking_huber_scale) ||
      video_tracking_huber_scale <= 0.0 ||
      !std::isfinite(video_lc_cauchy_scale) || video_lc_cauchy_scale <= 0.0 ||
      num_threads == 0 || num_threads < -1 || max_num_iterations <= 0) {
    throw std::invalid_argument("invalid rotation averaging options");
  }
  if (use_imu) {
    if (!gyro_bias_prior.allFinite() ||
        !std::isfinite(gyro_bias_prior_stddev) ||
        gyro_bias_prior_stddev <= 0.0 ||
        !std::isfinite(visual_rotation_stddev_deg) ||
        visual_rotation_stddev_deg <= 0.0 ||
        !std::isfinite(imu_tracking_cauchy_scale_deg) ||
        imu_tracking_cauchy_scale_deg <= 0.0 ||
        !std::isfinite(reintegrate_angle_norm_thres) ||
        reintegrate_angle_norm_thres < 0.0) {
      throw std::invalid_argument("invalid I-RA options");
    }
  }
  if (salvage_outlier_translations) {
    if (!std::isfinite(salvage_epipolar_angle_thres_deg) ||
        salvage_epipolar_angle_thres_deg <= 0.0 ||
        !std::isfinite(salvage_min_inlier_ratio) ||
        salvage_min_inlier_ratio <= 0.0 || salvage_min_inlier_ratio > 1.0 ||
        salvage_min_inliers < 2) {
      throw std::invalid_argument(
          "invalid salvage_outlier_translations options");
    }
  }
}

RotationAveragingResult RunVideoRotationAveraging(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImageId>& image_map_order,
    const std::vector<PairId>& pair_map_order,
    MappingProblem* problem,
    const std::vector<ImuEdgeRecord>& imu_edges,
    const std::vector<ImuStateRecord>& imu_states) {
  options.Validate();
  problem->Validate();
  ValidateImageMapOrder(*problem, image_map_order);
  ValidatePairMapOrder(*problem, pair_map_order);

  const bool has_imu = options.use_imu && !imu_edges.empty();
  if (options.use_imu) {
    ValidateImuInputs(*problem, imu_edges, imu_states);
  }

  RotationAveragingResult result;
  std::unordered_set<ImageId> active_images =
      ComputeLargestConnectedComponentImageIds(
          *problem,
          image_map_order,
          pair_map_order,
          options.filter_unregistered,
          /*excluded_pair_ids=*/{},
          has_imu ? imu_edges : std::vector<ImuEdgeRecord>{});
  if (active_images.empty()) return result;
  const std::unordered_set<ImageId> initial_active_images = active_images;

  Eigen::Quaterniond q_IC = Eigen::Quaterniond::Identity();
  if (options.imu_from_cam.has_pose) {
    q_IC = ToColmapPose(options.imu_from_cam).rotation().normalized();
  }

  std::vector<ImuEdgeRecord> mutable_imu_edges = imu_edges;
  std::map<ImageId, Eigen::Matrix<double, 9, 1>> imu_state_params;
  Eigen::Vector3d bg_init = options.gyro_bias_prior;
  Eigen::Vector3d effective_gyro_bias_prior = options.gyro_bias_prior;

  if (has_imu) {
    for (const ImuStateRecord& state : imu_states) {
      imu_state_params.emplace(state.image_id, state.ToVector());
    }
    for (const ImuEdgeRecord& edge : mutable_imu_edges) {
      for (const ImageId image_id : {edge.image_id1, edge.image_id2}) {
        if (imu_state_params.count(image_id) == 0) {
          Eigen::Matrix<double, 9, 1> state =
              Eigen::Matrix<double, 9, 1>::Zero();
          state.segment<3>(3) = bg_init;
          imu_state_params.emplace(image_id, state);
        }
      }
    }

    if (options.auto_initialize_gyro_bias && imu_states.empty() &&
        EstimateInitialGyroBiasMedian(*problem,
                                      pair_map_order,
                                      active_images,
                                      /*excluded_pair_ids=*/{},
                                      mutable_imu_edges,
                                      q_IC,
                                      &bg_init)) {
      if (options.gyro_bias_prior.isZero(1e-12)) {
        effective_gyro_bias_prior = bg_init;
      }
      ApplyUniformGyroBiasAndReintegrate(
          bg_init, &imu_state_params, &mutable_imu_edges);
    }
    result.initial_gyro_bias = bg_init;
  }

  InitializeFromMaximumSpanningTree(
      options,
      pair_map_order,
      active_images,
      problem,
      /*excluded_pair_ids=*/{},
      has_imu ? mutable_imu_edges : std::vector<ImuEdgeRecord>{},
      imu_state_params,
      q_IC);
  const std::vector<ImageId> parameter_image_order =
      OrderedActiveImages(active_images, options.image_order_passes);

  if (!has_imu) {
    if (!SolveRotationAveragingCeresPass(options,
                                         parameter_image_order,
                                         pair_map_order,
                                         /*excluded_pair_ids=*/{},
                                         /*has_imu=*/false,
                                         /*optimize_gyro_bias=*/false,
                                         q_IC,
                                         effective_gyro_bias_prior,
                                         &mutable_imu_edges,
                                         &imu_state_params,
                                         problem)) {
      return result;
    }
  } else {
    // Pass 1: Optimize global rotations with redescending CauchyLoss on visual
    // edges and gyro bias held at the median seed bg_init so contiguous
    // moving-object outliers cannot twist the gyro bias.
    if (!SolveRotationAveragingCeresPass(options,
                                         parameter_image_order,
                                         pair_map_order,
                                         /*excluded_pair_ids=*/{},
                                         /*has_imu=*/true,
                                         /*optimize_gyro_bias=*/false,
                                         q_IC,
                                         effective_gyro_bias_prior,
                                         &mutable_imu_edges,
                                         &imu_state_params,
                                         problem)) {
      return result;
    }

    const double internal_outlier_thres_deg =
        options.max_rotation_error_deg > 0.0 ? options.max_rotation_error_deg
                                             : 3.0;
    const std::unordered_set<PairId> pass1_outliers =
        FindRotationOutlierPairs(*problem,
                                 pair_map_order,
                                 initial_active_images,
                                 internal_outlier_thres_deg);

    if (!pass1_outliers.empty() && options.auto_initialize_gyro_bias &&
        imu_states.empty() &&
        EstimateInitialGyroBiasMedian(*problem,
                                      pair_map_order,
                                      active_images,
                                      pass1_outliers,
                                      mutable_imu_edges,
                                      q_IC,
                                      &bg_init)) {
      if (options.gyro_bias_prior.isZero(1e-12)) {
        effective_gyro_bias_prior = bg_init;
      }
      ApplyUniformGyroBiasAndReintegrate(
          bg_init, &imu_state_params, &mutable_imu_edges);
      result.initial_gyro_bias = bg_init;
      InitializeFromMaximumSpanningTree(options,
                                        pair_map_order,
                                        active_images,
                                        problem,
                                        pass1_outliers,
                                        mutable_imu_edges,
                                        imu_state_params,
                                        q_IC);
    }

    // Pass 2: Jointly optimize global rotations and per-frame gyroscope biases
    // over the surviving inlier visual edges and 6D IMU preintegration factors.
    if (options.refine_gyro_bias || !pass1_outliers.empty()) {
      if (!SolveRotationAveragingCeresPass(options,
                                           parameter_image_order,
                                           pair_map_order,
                                           pass1_outliers,
                                           /*has_imu=*/true,
                                           options.refine_gyro_bias,
                                           q_IC,
                                           effective_gyro_bias_prior,
                                           &mutable_imu_edges,
                                           &imu_state_params,
                                           problem)) {
        return result;
      }
    }
  }

  if (options.max_rotation_error_deg > 0.0) {
    const std::unordered_set<PairId> outlier_pairs =
        FindRotationOutlierPairs(*problem,
                                 pair_map_order,
                                 initial_active_images,
                                 options.max_rotation_error_deg);
    result.outlier_pair_ids.assign(outlier_pairs.begin(), outlier_pairs.end());
    std::sort(result.outlier_pair_ids.begin(), result.outlier_pair_ids.end());

    std::unordered_set<PairId> excluded_pairs = outlier_pairs;
    if (options.invalidate_outlier_pairs ||
        options.salvage_outlier_translations) {
      if (options.random_seed >= 0 && options.salvage_outlier_translations) {
        colmap::SetPRNGSeed(static_cast<unsigned>(options.random_seed));
      }
      for (const PairId pair_id : result.outlier_pair_ids) {
        PairRecord pair = problem->Pair(pair_id);
        const ImageRecord& image1 = problem->Image(pair.image_id1);
        const ImageRecord& image2 = problem->Image(pair.image_id2);
        if (options.salvage_outlier_translations &&
            TrySalvagePairTranslationWithKnownRotation(
                options, image1, image2, &pair)) {
          problem->UpdatePair(pair);
          result.salvaged_pair_ids.push_back(pair_id);
          excluded_pairs.erase(pair_id);
        } else if (options.invalidate_outlier_pairs) {
          pair.is_valid = false;
          problem->UpdatePair(pair);
        }
      }
    }

    // Exclude every edge outside the initial largest component so a discarded
    // component cannot re-enter after outlier filtering.
    for (const PairId pair_id : pair_map_order) {
      const PairRecord& pair = problem->Pair(pair_id);
      if (initial_active_images.count(pair.image_id1) == 0 ||
          initial_active_images.count(pair.image_id2) == 0) {
        excluded_pairs.insert(pair_id);
      }
    }
    active_images = ComputeLargestConnectedComponentImageIds(
        *problem,
        image_map_order,
        pair_map_order,
        true,
        excluded_pairs,
        has_imu ? mutable_imu_edges : std::vector<ImuEdgeRecord>{},
        has_imu ? &initial_active_images : nullptr);
    if (active_images.empty()) return result;
  }

  if (has_imu) {
    for (ImuEdgeRecord& edge : mutable_imu_edges) {
      if (edge.integrator != nullptr &&
          imu_state_params.count(edge.image_id1) != 0) {
        Eigen::Vector6d biases = edge.data.biases;
        biases.head<3>() = imu_state_params.at(edge.image_id1).segment<3>(3);
        edge.integrator->Reintegrate(biases);
        edge.integrator->Update(&edge.data);
      }
    }
    result.initial_gravity_direction =
        ComputeTelescopingInitialGravityDirection(
            *problem, active_images, mutable_imu_edges, imu_state_params, q_IC);
    for (const auto& [image_id, state_vec] : imu_state_params) {
      result.imu_states.emplace(
          image_id, ImuStateRecord::FromVector(image_id, state_vec));
    }
  }

  for (const ImageId image_id : active_images) {
    result.registered_image_ids.push_back(image_id);
  }
  result.success = true;
  return result;
}

}  // namespace vidmap

#undef VIDMAP_COLMAP_HAS_FLAT_HASH_SET
