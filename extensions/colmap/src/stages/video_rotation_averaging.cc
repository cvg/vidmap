// Video-aware rotation averaging over the canonical COLMAP scene.
#include "vidmap_native/video_rotation_averaging.h"

#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/geometry/pose.h"
#include "colmap/math/spanning_tree.h"
#include "colmap/util/threading.h"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace vidmap {
namespace {

constexpr float kLCPenalty = 1e9f;

struct RelativeRotationError
    : public colmap::AutoDiffCostFunctor<RelativeRotationError, 3, 3, 3> {
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

  const Eigen::Vector3d rel_rot_aa_;
};

void ValidateTrivialFrames(const colmap::Reconstruction& reconstruction) {
  std::unordered_set<FrameId> frame_ids;
  frame_ids.reserve(reconstruction.NumImages());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.IsRefInFrame() || !frame_ids.insert(image.FrameId()).second) {
      throw std::invalid_argument(
          "video rotation averaging requires one reference camera per frame");
    }
  }
}

bool IsTrackingPair(const PairData& pair) {
  if (pair.inlier_indices.size() == 0) return false;
  std::size_t loop_closure_count = 0;
  for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
    const int row = pair.inlier_indices[index];
    if (row < pair.are_loop_closure.size() && pair.are_loop_closure[row] != 0) {
      ++loop_closure_count;
    }
  }
  return static_cast<std::size_t>(pair.inlier_indices.size()) -
             loop_closure_count >=
         loop_closure_count;
}

void InitializeFromMaximumSpanningTree(
    const std::vector<ImageId>& ordered_images,
    colmap::Reconstruction& reconstruction,
    const colmap::PoseGraph& graph,
    const MappingSidecars& sidecars) {
  std::unordered_map<ImageId, int> image_to_index;
  image_to_index.reserve(ordered_images.size());
  for (std::size_t index = 0; index < ordered_images.size(); ++index) {
    image_to_index.emplace(ordered_images[index], static_cast<int>(index));
  }

  std::vector<std::pair<int, int>> edges;
  std::vector<float> weights;
  for (const auto& [pair_id, edge] : graph.ValidEdges()) {
    const PairData& pair = sidecars.pairs.at(pair_id);
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    edges.emplace_back(image_to_index.at(image_id1),
                       image_to_index.at(image_id2));
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
  const auto& root_image = reconstruction.Image(ordered_images[tree.root]);
  if (root_image.HasPose()) {
    cam_from_world[tree.root] = root_image.CamFromWorld();
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
      const auto relative_pose =
          graph.GetEdge(parent_id, child_id).cam2_from_cam1;
      cam_from_world[child_index].rotation() =
          (relative_pose * cam_from_world[parent_index]).rotation();
    }
  }

  for (std::size_t index = 0; index < ordered_images.size(); ++index) {
    auto& image = reconstruction.Image(ordered_images[index]);
    const Eigen::Vector3d translation =
        image.HasPose() ? Eigen::Vector3d(image.CamFromWorld().translation())
                        : Eigen::Vector3d::Zero();
    image.FramePtr()->SetRigFromWorld(
        colmap::Rigid3d(cam_from_world[index].rotation(), translation));
  }
}

}  // namespace

void VideoRotationAveragingOptions::Validate() const {
  if (!std::isfinite(max_rotation_error_deg) || max_rotation_error_deg < 0.0 ||
      !std::isfinite(video_tracking_huber_scale) ||
      video_tracking_huber_scale <= 0.0 ||
      !std::isfinite(video_lc_cauchy_scale) || video_lc_cauchy_scale <= 0.0 ||
      num_threads == 0 || num_threads < -1 || max_num_iterations <= 0) {
    throw std::invalid_argument("invalid rotation averaging options");
  }
}

RotationAveragingResult RunVideoRotationAveraging(
    const VideoRotationAveragingOptions& options,
    colmap::Reconstruction& reconstruction,
    const colmap::PoseGraph& graph,
    const MappingSidecars& sidecars) {
  options.Validate();
  sidecars.Validate(reconstruction);
  ValidateTrivialFrames(reconstruction);

  RotationAveragingResult result;
  // Component filtering is local to this pass; preserve canonical edge
  // validity.
  colmap::PoseGraph eligible_graph;
  for (const auto& [pair_id, edge] : graph.ValidEdges()) {
    if (sidecars.pairs.at(pair_id).has_relative_pose) {
      eligible_graph.Edges().emplace(pair_id, edge);
    }
  }
  auto components = eligible_graph.ConnectedImageIdsForFrameComponents(
      reconstruction, options.filter_unregistered);
  if (components.empty()) return result;
  auto active_images = std::move(components.front());
  eligible_graph.InvalidatePairsOutsideActiveImageIds(active_images);

  const std::vector<ImageId> parameter_image_order(active_images.begin(),
                                                   active_images.end());
  InitializeFromMaximumSpanningTree(
      parameter_image_order, reconstruction, eligible_graph, sidecars);
  const ImageId fixed_image_id = parameter_image_order.front();

  std::unordered_map<ImageId, int> image_to_parameter_index;
  image_to_parameter_index.reserve(parameter_image_order.size());
  Eigen::VectorXd rotations(3 * parameter_image_order.size());
  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    const ImageId image_id = parameter_image_order[index];
    image_to_parameter_index.emplace(image_id, 3 * index);
    const Eigen::AngleAxisd angle_axis(
        reconstruction.Image(image_id).CamFromWorld().rotation());
    rotations.segment<3>(3 * index) = angle_axis.angle() * angle_axis.axis();
  }

  ceres::HuberLoss tracking_loss(options.video_tracking_huber_scale);
  ceres::CauchyLoss loop_closure_loss(options.video_lc_cauchy_scale);
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);
  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    double* parameter = rotations.data() + 3 * index;
    ceres_problem.AddParameterBlock(parameter, 3);
    if (parameter_image_order[index] == fixed_image_id) {
      ceres_problem.SetParameterBlockConstant(parameter);
    }
  }

  for (const auto& [pair_id, edge] : eligible_graph.ValidEdges()) {
    const PairData& pair = sidecars.pairs.at(pair_id);
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    const bool is_tracking = IsTrackingPair(pair);
    if (options.skip_risky_lc_pairs && !is_tracking) continue;
    ceres::LossFunction* loss =
        is_tracking ? static_cast<ceres::LossFunction*>(&tracking_loss)
                    : static_cast<ceres::LossFunction*>(&loop_closure_loss);
    const Eigen::Vector3d relative_angle_axis =
        colmap::RotationMatrixToAngleAxis(
            edge.cam2_from_cam1.rotation().toRotationMatrix());
    ceres_problem.AddResidualBlock(
        RelativeRotationError::Create(relative_angle_axis),
        loss,
        rotations.data() + image_to_parameter_index.at(image_id1),
        rotations.data() + image_to_parameter_index.at(image_id2));
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(options.num_threads);
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &ceres_problem, &summary);
  if (!summary.IsSolutionUsable()) return result;

  for (std::size_t index = 0; index < parameter_image_order.size(); ++index) {
    auto& image = reconstruction.Image(parameter_image_order[index]);
    const Eigen::Matrix3d rotation =
        colmap::AngleAxisToRotationMatrix(rotations.segment<3>(3 * index));
    image.FramePtr()->SetRigFromWorld(colmap::Rigid3d(
        Eigen::Quaterniond(rotation), image.CamFromWorld().translation()));
  }

  if (options.max_rotation_error_deg > 0.0) {
    colmap::FilterEdgesByRelativeRotation(
        eligible_graph, reconstruction, options.max_rotation_error_deg);
    components = eligible_graph.ConnectedImageIdsForFrameComponents(
        reconstruction, /*filter_unregistered=*/true);
    if (components.empty()) return result;
    active_images = std::move(components.front());
  }
  result.registered_image_ids.assign(active_images.begin(),
                                     active_images.end());
  result.success = true;
  return result;
}

}  // namespace vidmap
