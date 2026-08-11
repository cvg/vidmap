// Video-aware rotation averaging over VidMap-owned value records.
#include "vidmap_native/video_rotation_averaging.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/connected_components.h"
#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/math/spanning_tree.h"

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
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vidmap_native/conversion.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vidmap {
namespace {

constexpr float kLCPenalty = 1e9f;

#if VIDMAP_COLMAP_HAS_FLAT_HASH_SET
using ConnectedComponentFrameSet = colmap::FlatHashSet<FrameId>;
#else
using ConnectedComponentFrameSet = std::unordered_set<FrameId>;
#endif

struct RelativeRotationError {
  explicit RelativeRotationError(const Eigen::Vector3d& rel_rot_aa)
      : rel_rot_aa_(rel_rot_aa) {}

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
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& rel_rot_aa) {
    return new ceres::AutoDiffCostFunction<RelativeRotationError, 3, 3, 3>(
        new RelativeRotationError(rel_rot_aa));
  }

  const Eigen::Vector3d rel_rot_aa_;
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
    const std::unordered_set<PairId>& excluded_pair_ids = {}) {
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

void InitializeFromMaximumSpanningTree(
    const VideoRotationAveragingOptions& options,
    const std::vector<PairId>& pair_map_order,
    const std::unordered_set<ImageId>& active_images,
    MappingProblem* problem) {
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
      const PairRecord& pair =
          problem->Pair(CanonicalPairId(child_id, parent_id));
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

}  // namespace

void VideoRotationAveragingOptions::Validate() const {
  if (random_seed < -1 || image_order_passes < 0 ||
      !std::isfinite(max_rotation_error_deg) || max_rotation_error_deg < 0.0 ||
      !std::isfinite(video_tracking_huber_scale) ||
      video_tracking_huber_scale <= 0.0 ||
      !std::isfinite(video_lc_cauchy_scale) || video_lc_cauchy_scale <= 0.0 ||
      num_threads == 0 || num_threads < -1 || max_num_iterations <= 0) {
    throw std::invalid_argument("invalid rotation averaging options");
  }
}

RotationAveragingResult RunVideoRotationAveraging(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImageId>& image_map_order,
    const std::vector<PairId>& pair_map_order,
    MappingProblem* problem) {
  options.Validate();
  problem->Validate();
  ValidateImageMapOrder(*problem, image_map_order);
  ValidatePairMapOrder(*problem, pair_map_order);

  RotationAveragingResult result;
  std::unordered_set<ImageId> active_images =
      ComputeLargestConnectedComponentImageIds(*problem,
                                               image_map_order,
                                               pair_map_order,
                                               options.filter_unregistered);
  if (active_images.empty()) return result;
  const std::unordered_set<ImageId> initial_active_images = active_images;

  InitializeFromMaximumSpanningTree(
      options, pair_map_order, active_images, problem);
  const std::vector<ImageId> parameter_image_order =
      OrderedActiveImages(active_images, options.image_order_passes);
  const ImageId fixed_image_id = parameter_image_order.front();

  std::unordered_map<ImageId, int> image_to_parameter_index;
  image_to_parameter_index.reserve(parameter_image_order.size());
  Eigen::VectorXd rotations(3 * parameter_image_order.size());
  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    const ImageId image_id = parameter_image_order[index];
    image_to_parameter_index.emplace(image_id, 3 * index);
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

  for (const PairId pair_id : pair_map_order) {
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
    ceres::LossFunction* loss =
        is_tracking ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(
                          options.video_tracking_huber_scale))
                    : static_cast<ceres::LossFunction*>(
                          new ceres::CauchyLoss(options.video_lc_cauchy_scale));
    const Eigen::Vector3d relative_angle_axis =
        colmap::RotationMatrixToAngleAxis(
            ToColmapPose(pair.geometry.cam2_from_cam1)
                .rotation()
                .toRotationMatrix());
    ceres_problem.AddResidualBlock(
        RelativeRotationError::Create(relative_angle_axis),
        loss,
        rotations.data() + image1_it->second,
        rotations.data() + image2_it->second);
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.num_threads =
      options.num_threads > 0
          ? options.num_threads
          : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &ceres_problem, &summary);
  if (!summary.IsSolutionUsable()) return result;

  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    ImageRecord image = problem->Image(parameter_image_order[index]);
    const Eigen::Matrix3d rotation =
        colmap::AngleAxisToRotationMatrix(rotations.segment<3>(3 * index));
    const Eigen::Vector3d translation = image.pose.translation;
    image.pose = FromColmapPose(
        colmap::Rigid3d(Eigen::Quaterniond(rotation), translation));
    problem->UpdateImage(image);
  }

  if (options.max_rotation_error_deg > 0.0) {
    const std::unordered_set<PairId> outlier_pairs =
        FindRotationOutlierPairs(*problem,
                                 pair_map_order,
                                 initial_active_images,
                                 options.max_rotation_error_deg);

    // Exclude every edge outside the initial largest component so a discarded
    // component cannot re-enter after outlier filtering.
    std::unordered_set<PairId> excluded_pairs = outlier_pairs;
    for (const PairId pair_id : pair_map_order) {
      const PairRecord& pair = problem->Pair(pair_id);
      if (initial_active_images.count(pair.image_id1) == 0 ||
          initial_active_images.count(pair.image_id2) == 0) {
        excluded_pairs.insert(pair_id);
      }
    }
    active_images = ComputeLargestConnectedComponentImageIds(
        *problem, image_map_order, pair_map_order, true, excluded_pairs);
    if (active_images.empty()) return result;
  }
  for (const ImageId image_id : active_images) {
    result.registered_image_ids.push_back(image_id);
  }
  result.success = true;
  return result;
}

}  // namespace vidmap

#undef VIDMAP_COLMAP_HAS_FLAT_HASH_SET
