// Global positioning over VidMap-owned value records using COLMAP and Ceres
// primitives.
#include "vidmap_native/global_positioning.h"

#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/random.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "metric_depth.h"
#include "solver_playback.h"
#include "vidmap_native/conversion.h"
#include "weighted_motion_averaging.h"
#include <ceres/ceres.h>

namespace vidmap {
namespace {

std::string GpObservationKey(Point3DId point3D_id,
                             ImageId image_id,
                             std::uint32_t point2D_idx,
                             bool is_loop_closure) {
  return std::to_string(point3D_id) + ":" + std::to_string(image_id) + ":" +
         std::to_string(point2D_idx) + ":" + (is_loop_closure ? "1" : "0");
}

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(colmap::RandomUniformReal(low, high),
                         colmap::RandomUniformReal(low, high),
                         colmap::RandomUniformReal(low, high));
}

bool IsLossOverride(const LossConfig& loss) {
  return loss.type != LossFunctionType::kTrivial || loss.scale != 1.0 ||
         loss.weight != 1.0;
}

std::shared_ptr<ceres::LossFunction> SharedLoss(const LossConfig& config) {
  return std::shared_ptr<ceres::LossFunction>(config.Create().release());
}

class DeadZoneHuberLoss final : public ceres::LossFunction {
 public:
  DeadZoneHuberLoss(double dead_zone, double huber_width)
      : dead_zone_(dead_zone), huber_width_(huber_width) {}

  void Evaluate(double squared_norm, double rho[3]) const override {
    const double residual_norm = std::sqrt(std::max(0.0, squared_norm));
    if (residual_norm <= dead_zone_) {
      rho[0] = 0.0;
      rho[1] = 0.0;
      rho[2] = 0.0;
      return;
    }
    const double shifted_norm = residual_norm - dead_zone_;
    if (shifted_norm <= huber_width_) {
      rho[0] = shifted_norm * shifted_norm;
      rho[1] = shifted_norm / residual_norm;
      rho[2] =
          dead_zone_ / (2.0 * residual_norm * residual_norm * residual_norm);
      return;
    }
    rho[0] = 2.0 * huber_width_ * shifted_norm - huber_width_ * huber_width_;
    rho[1] = huber_width_ / residual_norm;
    rho[2] =
        -huber_width_ / (2.0 * residual_norm * residual_norm * residual_norm);
  }

 private:
  const double dead_zone_;
  const double huber_width_;
};

Eigen::Quaterniond ImageRotation(const ImageRecord& image) {
  return ToColmapPose(image.pose).rotation();
}

Eigen::Vector3d ImageCenter(const ImageRecord& image) {
  return ToColmapPose(image.pose).TgtOriginInSrc();
}

class GlobalPositioner {
 public:
  GlobalPositioner(const GlobalPositionerOptions& options,
                   MappingProblem* mapping_problem)
      : options_(options), mapping_problem_(mapping_problem) {}

  GlobalPositioningResult Solve() {
    options_.Validate();
    mapping_problem_->Validate();
    ValidateProblem();
    if (mapping_problem_->NumImages() == 0 ||
        mapping_problem_->NumTracks() == 0) {
      return result_;
    }
    if (options_.random_seed >= 0) {
      colmap::SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
    }

    SetupProblem();
    InitializeRandomPositions();
    AddPointToCameraConstraints();
    if (options_.use_parameter_block_ordering) {
      AddCamerasAndPointsToParameterGroups();
    }
    ParameterizeVariables();

    ceres::Solver::Summary summary;
    try {
      SolveWithPlayback(
          options_.playback,
          solver_options_,
          problem_.get(),
          [this](const char* phase, const int iteration) {
            WritePlaybackCapture(phase, iteration);
          },
          &summary);
    } catch (...) {
      ConvertBackResults();
      throw;
    }
    PopulateResult(summary);
    ConvertBackResults();
    return result_;
  }

 private:
  struct Observation {
    ImageId image_id;
    std::uint32_t point2D_idx;
  };

  struct PlaybackObservation {
    Observation observation;
    Observation anchor;
    ceres::ResidualBlockId residual_block_id;
    ceres::LossFunction* loss_function;
  };

  struct PlaybackEdge {
    ImageId image_id1;
    ImageId image_id2;
    std::vector<std::size_t> observation_indices;
    std::size_t support_count;
  };

  void ValidateProblem() {
    image_ids_.clear();
    image_ids_.reserve(mapping_problem_->NumImages());
    std::unordered_set<FrameId> frame_ids;
    frame_ids.reserve(mapping_problem_->NumImages());
    for (const ImageId image_id : mapping_problem_->ImageIds()) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      image_ids_.insert(image_id);
      if (!frame_ids.insert(image.frame_id).second) {
        throw std::invalid_argument(
            "global positioning requires one image per frame");
      }
    }
  }

  void SetupProblem() {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    problem_ = std::make_unique<ceres::Problem>(problem_options);

    point_xyz_.clear();
    std::size_t total_observations = 0;
    for (const Point3DId point3D_id : mapping_problem_->Point3DIds()) {
      const TrackRecord& track = mapping_problem_->Track(point3D_id);
      point_xyz_.emplace(point3D_id, track.xyz);
      total_observations += track.observations.rows();
      total_observations += track.loop_closure_observations.rows();
    }
    scales_.clear();
    scales_.reserve(total_observations);
    scale_indices_.clear();
    image_centers_.clear();
    image_centers_.reserve(mapping_problem_->NumImages());
    frame_centers_.clear();
    frame_centers_.reserve(mapping_problem_->NumImages());
    dmap_scales_.clear();
    dmap_scale_observation_counts_.clear();
    depth_outliers_.clear();
    per_image_scale_prior_losses_.clear();
    temporal_acceleration_losses_.clear();
    result_ = GlobalPositioningResult();

    loss_ = SharedLoss(options_.loss);
    calibrated_loss_ = loss_;
    if (options_.apply_uncalibrated_loss_downweight) {
      uncalibrated_loss_ = std::make_shared<ceres::ScaledLoss>(
          loss_.get(),
          options_.uncalibrated_loss_downweight,
          ceres::DO_NOT_TAKE_OWNERSHIP);
    } else {
      uncalibrated_loss_ = loss_;
    }
    loss_normal_geometry_ = SharedLoss(options_.loss_normal_geometry);
    loss_normal_depth_ = SharedLoss(options_.loss_normal_depth);
    loss_lc_geometry_ = IsLossOverride(options_.loss_lc_geometry)
                            ? SharedLoss(options_.loss_lc_geometry)
                            : nullptr;
    loss_lc_depth_ = SharedLoss(options_.loss_lc_depth);
    loss_normal_geometry_inlier_ =
        SharedLoss(options_.loss_normal_geometry_inlier);
    loss_normal_depth_inlier_ = SharedLoss(options_.loss_normal_depth_inlier);
    loss_normal_depth_outlier_ = SharedLoss(options_.loss_normal_depth_outlier);
    loss_normal_geometry_track_anchor_ =
        SharedLoss(options_.loss_normal_geometry_track_anchor);
    loss_normal_depth_track_anchor_ =
        SharedLoss(options_.loss_normal_depth_track_anchor);
    loss_scale_prior_ = SharedLoss(options_.loss_scale_prior);
    loss_soft_outlier_fallback_.reset();

    solver_options_.num_threads = options_.num_threads;
    solver_options_.max_num_iterations = options_.max_num_iterations;
    solver_options_.function_tolerance = options_.function_tolerance;
    solver_options_.gradient_tolerance = options_.gradient_tolerance;
    solver_options_.parameter_tolerance = options_.parameter_tolerance;
    solver_options_.minimizer_progress_to_stdout = false;
  }

  void InitializeRandomPositions() {
    std::unordered_set<FrameId> constrained_frames;
    constrained_frames.reserve(mapping_problem_->NumImages());
    for (const PairId pair_id : mapping_problem_->PairIds()) {
      const PairRecord& pair = mapping_problem_->Pair(pair_id);
      if (!pair.is_valid || !pair.geometry.cam2_from_cam1.has_pose) continue;
      const ImageRecord& image1 = mapping_problem_->Image(pair.image_id1);
      const ImageRecord& image2 = mapping_problem_->Image(pair.image_id2);
      if (!image1.pose.has_pose || !image2.pose.has_pose) continue;
      constrained_frames.insert(image1.frame_id);
      constrained_frames.insert(image2.frame_id);
    }
    for (const Point3DId point3D_id : mapping_problem_->Point3DIds()) {
      const TrackRecord& track = mapping_problem_->Track(point3D_id);
      if (track.observations.rows() < options_.min_num_view_per_track) {
        continue;
      }
      AddConstrainedFrames(track.observations, &constrained_frames);
      if (options_.use_lc_observations) {
        AddConstrainedFrames(track.loop_closure_observations,
                             &constrained_frames);
      }
    }

    std::vector<ImageId> ordered_image_ids = mapping_problem_->ImageIds();
    if (options_.center_mode == GlobalPositioningCenterMode::kFrame) {
      std::sort(ordered_image_ids.begin(),
                ordered_image_ids.end(),
                [this](const ImageId lhs, const ImageId rhs) {
                  return mapping_problem_->Image(lhs).frame_id <
                         mapping_problem_->Image(rhs).frame_id;
                });
    }
    for (const ImageId image_id : ordered_image_ids) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      if (constrained_frames.count(image.frame_id) == 0) continue;
      Eigen::Vector3d center;
      if (options_.generate_random_positions && options_.optimize_positions &&
          !options_.use_init) {
        center = options_.random_init_scale * RandVector3d(-1.0, 1.0);
      } else {
        center = ImageCenter(image);
      }
      const auto initial_center =
          options_.initial_frame_centers.find(image.frame_id);
      if (initial_center != options_.initial_frame_centers.end()) {
        center = initial_center->second;
      }
      if (options_.center_mode == GlobalPositioningCenterMode::kImage) {
        image_centers_.emplace(image_id, center);
      } else {
        frame_centers_.emplace(image.frame_id, center);
      }
      result_.initial_frame_centers.emplace(image.frame_id, center);
    }
  }

  Eigen::Vector3d& CenterForImage(const ImageRecord& image) {
    if (options_.center_mode == GlobalPositioningCenterMode::kImage) {
      return image_centers_.at(image.image_id);
    }
    return frame_centers_.at(image.frame_id);
  }

  const Eigen::Vector3d& CenterForImage(const ImageRecord& image) const {
    if (options_.center_mode == GlobalPositioningCenterMode::kImage) {
      return image_centers_.at(image.image_id);
    }
    return frame_centers_.at(image.frame_id);
  }

  void AddConstrainedFrames(
      const MatrixX2u& observations,
      std::unordered_set<FrameId>* constrained_frames) const {
    for (Eigen::Index row = 0; row < observations.rows(); ++row) {
      const ImageId image_id = observations(row, 0);
      if (image_ids_.count(image_id) == 0) continue;
      const ImageRecord& image = mapping_problem_->Image(image_id);
      if (image.pose.has_pose) constrained_frames->insert(image.frame_id);
    }
  }

  void AddPointToCameraConstraints() {
    if (options_.use_metric_depth_constraint &&
        !options_.initial_dmap_scales.empty()) {
      for (const auto& [image_id, linear_scale] :
           options_.initial_dmap_scales) {
        dmap_scales_[image_id] = options_.use_log_scale_for_depth_map_scales
                                     ? std::log(std::max(linear_scale, 1e-9))
                                     : linear_scale;
        dmap_scale_observation_counts_[image_id] = 0;
      }
    }
    if (options_.use_metric_depth_constraint &&
        options_.filter_depth_outliers) {
      FindDepthOutliers();
    }

    for (const Point3DId point3D_id : mapping_problem_->Point3DIds()) {
      const TrackRecord& track = mapping_problem_->Track(point3D_id);
      if (track.observations.rows() < options_.min_num_view_per_track) {
        continue;
      }
      AddPoint3DToProblem(point3D_id);
    }
    AddTemporalAccelerationConstraints();
    AddDepthMapScalePriors();
  }

  void AddTemporalAccelerationConstraints() {
    if (!options_.use_temporal_acceleration_prior) return;

    const double residual_scale =
        1.0 / options_.temporal_acceleration_prior_stddev;
    for (const TemporalAccelerationPrior& prior :
         options_.temporal_acceleration_priors) {
      if (image_ids_.count(prior.prev_image_id) == 0 ||
          image_ids_.count(prior.image_id) == 0 ||
          image_ids_.count(prior.next_image_id) == 0) {
        throw std::invalid_argument(
            "temporal acceleration prior references an unknown image");
      }
      const ImageRecord& prev_image =
          mapping_problem_->Image(prior.prev_image_id);
      const ImageRecord& image = mapping_problem_->Image(prior.image_id);
      const ImageRecord& next_image =
          mapping_problem_->Image(prior.next_image_id);
      if (!prev_image.pose.has_pose || !image.pose.has_pose ||
          !next_image.pose.has_pose) {
        throw std::invalid_argument(
            "temporal acceleration prior requires posed images");
      }
      const bool centers_exist =
          options_.center_mode == GlobalPositioningCenterMode::kImage
              ? image_centers_.count(prior.prev_image_id) != 0 &&
                    image_centers_.count(prior.image_id) != 0 &&
                    image_centers_.count(prior.next_image_id) != 0
              : frame_centers_.count(prev_image.frame_id) != 0 &&
                    frame_centers_.count(image.frame_id) != 0 &&
                    frame_centers_.count(next_image.frame_id) != 0;
      if (!centers_exist) {
        throw std::invalid_argument(
            "temporal acceleration prior requires active camera centers");
      }

      ceres::CostFunction* cost = TemporalAccelerationCostFunctor::Create(
          prior.dt_prev, prior.dt_next, residual_scale);
      if (cost == nullptr) {
        throw std::invalid_argument(
            "temporal acceleration prior has invalid timing");
      }
      const double observation_count_weight =
          prior.sqrt_observation_count * prior.sqrt_observation_count;
      temporal_acceleration_losses_.push_back(
          std::make_unique<ceres::ScaledLoss>(
              new DeadZoneHuberLoss(
                  options_.temporal_acceleration_prior_loss_dead_zone,
                  options_.temporal_acceleration_prior_loss_huber_width),
              options_.temporal_acceleration_prior_weight *
                  observation_count_weight,
              ceres::TAKE_OWNERSHIP));
      problem_->AddResidualBlock(cost,
                                 temporal_acceleration_losses_.back().get(),
                                 CenterForImage(prev_image).data(),
                                 CenterForImage(image).data(),
                                 CenterForImage(next_image).data());
      ++result_.diagnostics.num_temporal_acceleration_residuals;
    }
  }

  void AddPoint3DToProblem(Point3DId point3D_id) {
    const bool random_initialization = options_.optimize_points &&
                                       options_.generate_random_points &&
                                       !options_.use_init;
    Eigen::Vector3d& xyz = point_xyz_.at(point3D_id);
    if (random_initialization) {
      xyz = options_.random_init_scale * RandVector3d(-1.0, 1.0);
    }
    result_.initial_point3D_xyz.emplace(point3D_id, xyz);

    const TrackRecord& track = mapping_problem_->Track(point3D_id);
    for (Eigen::Index row = 0; row < track.observations.rows(); ++row) {
      AddObservation(point3D_id,
                     {track.observations(row, 0), track.observations(row, 1)},
                     random_initialization,
                     false,
                     std::nullopt);
    }
    if (options_.use_lc_observations) {
      for (Eigen::Index row = 0; row < track.loop_closure_observations.rows();
           ++row) {
        std::optional<Observation> anchor;
        if (track.loop_closure_anchors.rows() != 0) {
          anchor = Observation{track.loop_closure_anchors(row, 0),
                               track.loop_closure_anchors(row, 1)};
        }
        AddObservation(point3D_id,
                       {track.loop_closure_observations(row, 0),
                        track.loop_closure_observations(row, 1)},
                       random_initialization,
                       true,
                       anchor);
      }
    }
  }

  void AddObservation(Point3DId point3D_id,
                      const Observation& observation,
                      bool random_initialization,
                      bool is_loop_closure,
                      const std::optional<Observation>& loop_closure_anchor) {
    if (image_ids_.count(observation.image_id) == 0) return;
    const ImageRecord& image = mapping_problem_->Image(observation.image_id);
    if (!image.pose.has_pose) return;
    if ((options_.center_mode == GlobalPositioningCenterMode::kImage &&
         image_centers_.count(observation.image_id) == 0) ||
        (options_.center_mode == GlobalPositioningCenterMode::kFrame &&
         frame_centers_.count(image.frame_id) == 0)) {
      return;
    }
    Eigen::Vector3d& center = CenterForImage(image);

    const Eigen::Vector3d bearing = Bearing(image, observation.point2D_idx);
    if (bearing.array().isNaN().any()) return;
    const Eigen::Vector3d point_from_camera_direction =
        ImageRotation(image).inverse() * bearing;

    scales_.emplace_back(1.0);
    double& scale = scales_.back();
    Eigen::Vector3d& point_xyz = point_xyz_.at(point3D_id);
    if (!options_.generate_scales &&
        (random_initialization || options_.initialize_warm_start_scales)) {
      const Eigen::Vector3d delta = point_xyz - center;
      scale = std::max(
          1e-5, point_from_camera_direction.dot(delta) / delta.squaredNorm());
    }
    const std::string scale_key = GpObservationKey(point3D_id,
                                                   observation.image_id,
                                                   observation.point2D_idx,
                                                   is_loop_closure);
    scale_indices_[scale_key] = scales_.size() - 1;
    result_.initial_bata_scales.emplace(scale_key, scale);

    const CameraRecord& camera = mapping_problem_->Camera(image.camera_id);
    ceres::LossFunction* geometry_loss = camera.has_prior_focal_length
                                             ? calibrated_loss_.get()
                                             : uncalibrated_loss_.get();
    const bool is_track_anchor =
        MaskValue(image.is_track_anchor, observation.point2D_idx);
    const bool is_inlier = MaskValue(image.is_inlier, observation.point2D_idx);
    if (is_loop_closure && loss_lc_geometry_) {
      geometry_loss = loss_lc_geometry_.get();
    } else if (options_.use_metric_depth_constraint) {
      geometry_loss = is_track_anchor ? loss_normal_geometry_track_anchor_.get()
                      : is_inlier     ? loss_normal_geometry_inlier_.get()
                                      : loss_normal_geometry_.get();
    }

    ceres::CostFunction* cost = nullptr;
    if (observation.point2D_idx <
        static_cast<std::uint32_t>(image.angular_stddevs.rows())) {
      const double sigma_x =
          std::max(1e-9, image.angular_stddevs(observation.point2D_idx, 0));
      const double sigma_y =
          std::max(1e-9, image.angular_stddevs(observation.point2D_idx, 1));
      const double sigma_z = 0.5 * (sigma_x + sigma_y);
      cost = WeightedBATAPairwiseDirectionCostFunctor::Create(
          point_from_camera_direction,
          ImageRotation(image),
          sigma_x,
          sigma_y,
          sigma_z);
    }
    if (cost == nullptr) {
      cost = colmap::BATAPairwiseDirectionCostFunctor::Create(
          point_from_camera_direction);
    }
    const ceres::ResidualBlockId residual_block_id = problem_->AddResidualBlock(
        cost, geometry_loss, center.data(), point_xyz.data(), &scale);
    if (is_loop_closure && options_.playback.IsEnabled()) {
      if (!loop_closure_anchor.has_value()) {
        throw std::invalid_argument(
            "playback requires exact loop-closure provenance");
      }
      if (loop_closure_anchor->image_id == observation.image_id) {
        throw std::invalid_argument(
            "loop-closure playback endpoints must use different images");
      }
      playback_observations_.push_back({observation,
                                        *loop_closure_anchor,
                                        residual_block_id,
                                        geometry_loss});
    }
    problem_->SetParameterLowerBound(&scale, 0, 1e-5);
    ++result_.diagnostics.num_bata_residuals;
    if (is_loop_closure) {
      ++result_.diagnostics.num_loop_closure_observations_used;
    } else {
      ++result_.diagnostics.num_regular_observations_used;
    }

    if (options_.use_metric_depth_constraint) {
      AddMetricDepthResidual(point3D_id, observation, is_loop_closure, image);
    }
  }

  void PreparePlaybackTopology() {
    if (playback_topology_ready_) return;

    playback_image_ids_ = options_.playback.image_ids;
    if (playback_image_ids_.empty()) {
      for (const ImageId image_id : mapping_problem_->ImageIds()) {
        const ImageRecord& image = mapping_problem_->Image(image_id);
        const bool has_center =
            options_.center_mode == GlobalPositioningCenterMode::kImage
                ? image_centers_.count(image_id) != 0
                : frame_centers_.count(image.frame_id) != 0;
        if (image.pose.has_pose && has_center) {
          playback_image_ids_.push_back(image_id);
        }
      }
    }
    for (const ImageId image_id : playback_image_ids_) {
      const ImageRecord& image = mapping_problem_->Image(image_id);
      const bool has_center =
          options_.center_mode == GlobalPositioningCenterMode::kImage
              ? image_centers_.count(image_id) != 0
              : frame_centers_.count(image.frame_id) != 0;
      if (!image.pose.has_pose || !has_center) {
        throw std::invalid_argument(
            "playback image is not active in global positioning");
      }
    }

    playback_point3D_ids_ = options_.playback.point3D_ids;
    if (playback_point3D_ids_.empty()) {
      for (const auto& [point3D_id, xyz] : point_xyz_) {
        playback_point3D_ids_.push_back(point3D_id);
      }
      playback_point3D_ids_ =
          SelectPlaybackPoints(std::move(playback_point3D_ids_));
    }
    for (const Point3DId point3D_id : playback_point3D_ids_) {
      if (point_xyz_.count(point3D_id) == 0) {
        throw std::invalid_argument(
            "playback point is not active in global positioning");
      }
    }

    const std::set<ImageId> playback_images(playback_image_ids_.begin(),
                                            playback_image_ids_.end());
    using ObservationKey = std::pair<ImageId, std::uint32_t>;
    using MatchKey = std::pair<ObservationKey, ObservationKey>;
    using ImagePair = std::pair<ImageId, ImageId>;
    std::map<ImagePair, std::vector<std::size_t>> edge_observations;
    std::map<ImagePair, std::set<MatchKey>> edge_matches;
    for (std::size_t index = 0; index < playback_observations_.size();
         ++index) {
      const PlaybackObservation& observation = playback_observations_[index];
      if (playback_images.count(observation.observation.image_id) == 0 ||
          playback_images.count(observation.anchor.image_id) == 0) {
        continue;
      }
      const ImagePair image_pair = std::minmax(observation.observation.image_id,
                                               observation.anchor.image_id);
      edge_observations[image_pair].push_back(index);
      const ObservationKey endpoint{observation.observation.image_id,
                                    observation.observation.point2D_idx};
      const ObservationKey anchor{observation.anchor.image_id,
                                  observation.anchor.point2D_idx};
      edge_matches[image_pair].insert(std::minmax(endpoint, anchor));
    }
    for (auto& [image_pair, observation_indices] : edge_observations) {
      playback_edges_.push_back({image_pair.first,
                                 image_pair.second,
                                 std::move(observation_indices),
                                 edge_matches.at(image_pair).size()});
    }
    std::sort(playback_edges_.begin(),
              playback_edges_.end(),
              [](const PlaybackEdge& lhs, const PlaybackEdge& rhs) {
                return std::tuple(-static_cast<std::int64_t>(lhs.support_count),
                                  lhs.image_id1,
                                  lhs.image_id2) <
                       std::tuple(-static_cast<std::int64_t>(rhs.support_count),
                                  rhs.image_id1,
                                  rhs.image_id2);
              });
    playback_topology_ready_ = true;
  }

  void WritePlaybackCapture(const char* phase, const int iteration) {
    PreparePlaybackTopology();
    SolverPlaybackCapture capture;
    capture.phase = phase;
    capture.iteration = iteration;
    capture.image_ids = playback_image_ids_;
    capture.centers.resize(playback_image_ids_.size(), 3);
    for (std::size_t index = 0; index < playback_image_ids_.size(); ++index) {
      capture.centers.row(index) =
          CenterForImage(mapping_problem_->Image(playback_image_ids_[index]))
              .transpose();
    }
    capture.point3D_ids = playback_point3D_ids_;
    capture.points_xyz.resize(playback_point3D_ids_.size(), 3);
    for (std::size_t index = 0; index < playback_point3D_ids_.size(); ++index) {
      capture.points_xyz.row(index) =
          point_xyz_.at(playback_point3D_ids_[index]).transpose();
    }

    capture.loop_closure_pairs.resize(playback_edges_.size(), 2);
    capture.loop_closure_support_counts.reserve(playback_edges_.size());
    capture.loop_closure_raw_scores.resize(playback_edges_.size());
    std::unordered_map<std::size_t, double> observation_scores;
    for (std::size_t edge_index = 0; edge_index < playback_edges_.size();
         ++edge_index) {
      const PlaybackEdge& edge = playback_edges_[edge_index];
      capture.loop_closure_pairs(edge_index, 0) = edge.image_id1;
      capture.loop_closure_pairs(edge_index, 1) = edge.image_id2;
      capture.loop_closure_support_counts.push_back(edge.support_count);
      double edge_score = 0.0;
      for (const std::size_t observation_index : edge.observation_indices) {
        const auto [score_it, inserted] =
            observation_scores.try_emplace(observation_index, 0.0);
        if (inserted) {
          const PlaybackObservation& observation =
              playback_observations_.at(observation_index);
          double cost = 0.0;
          if (problem_->EvaluateResidualBlock(observation.residual_block_id,
                                              false,
                                              &cost,
                                              nullptr,
                                              nullptr)) {
            const double squared_norm = 2.0 * cost;
            double rho[3] = {squared_norm, 1.0, 0.0};
            if (observation.loss_function != nullptr) {
              observation.loss_function->Evaluate(squared_norm, rho);
            }
            score_it->second =
                std::max(rho[1], 0.0) * std::sqrt(std::max(rho[0], 0.0));
            if (!std::isfinite(score_it->second)) {
              score_it->second = 0.0;
            }
          }
        }
        edge_score += score_it->second;
      }
      capture.loop_closure_raw_scores[edge_index] = edge_score;
    }
    options_.playback.callback(capture);
  }

  Eigen::Vector3d Bearing(const ImageRecord& image,
                          std::uint32_t point2D_idx) const {
    if (point2D_idx < static_cast<std::uint32_t>(image.bearings.rows())) {
      return image.bearings.row(point2D_idx);
    }
    if (point2D_idx >= static_cast<std::uint32_t>(image.keypoints.rows())) {
      return Eigen::Vector3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
    }
    const colmap::Camera camera =
        ToColmapCamera(mapping_problem_->Camera(image.camera_id));
    const std::optional<Eigen::Vector2d> camera_point =
        camera.CamFromImg(image.keypoints.row(point2D_idx));
    if (!camera_point) {
      return Eigen::Vector3d::Constant(
          std::numeric_limits<double>::quiet_NaN());
    }
    return camera_point->homogeneous().normalized();
  }

  static bool MaskValue(const VectorXb& mask, std::uint32_t point2D_idx) {
    return point2D_idx < static_cast<std::uint32_t>(mask.size()) &&
           mask[point2D_idx] != 0;
  }

  void AddMetricDepthResidual(Point3DId point3D_id,
                              const Observation& observation,
                              bool is_loop_closure,
                              const ImageRecord& image) {
    if (!MaskValue(image.depth_validity, observation.point2D_idx)) return;
    if (observation.point2D_idx >=
            static_cast<std::uint32_t>(image.depth_values.size()) ||
        observation.point2D_idx >=
            static_cast<std::uint32_t>(image.depth_stddevs.size())) {
      throw std::invalid_argument("incomplete metric-depth sidecar");
    }
    const double depth_prior = image.depth_values[observation.point2D_idx];
    const double depth_stddev = image.depth_stddevs[observation.point2D_idx];
    if (depth_stddev <= 1e-9) return;

    auto [scale_it, inserted] = dmap_scales_.try_emplace(
        observation.image_id,
        options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0);
    if (inserted) dmap_scale_observation_counts_[observation.image_id] = 0;
    ++dmap_scale_observation_counts_[observation.image_id];

    ceres::CostFunction* cost =
        MetricDepthError::Create(ImageRotation(image),
                                 depth_prior,
                                 depth_stddev,
                                 options_.use_log_scale_for_depth_map_scales,
                                 options_.metric_depth_residual_type,
                                 options_.zero_residual_behind,
                                 options_.log_linear_threshold);
    if (cost == nullptr) return;

    const std::pair<ImageId, std::uint32_t> observation_key{
        observation.image_id, observation.point2D_idx};
    ceres::LossFunction* depth_loss = nullptr;
    if (depth_outliers_.count(observation_key) != 0) {
      if (is_loop_closure) {
        delete cost;
        return;
      }
      if (!loss_soft_outlier_fallback_) {
        loss_soft_outlier_fallback_ =
            SharedLoss(options_.loss_soft_outlier_fallback);
      }
      depth_loss = loss_soft_outlier_fallback_.get();
    } else if (is_loop_closure) {
      depth_loss = loss_lc_depth_.get();
    } else if (MaskValue(image.is_track_anchor, observation.point2D_idx)) {
      depth_loss = loss_normal_depth_track_anchor_.get();
    } else if (MaskValue(image.is_inlier, observation.point2D_idx)) {
      depth_loss = loss_normal_depth_inlier_.get();
    } else if (MaskValue(image.is_depth_outlier, observation.point2D_idx)) {
      depth_loss = loss_normal_depth_outlier_.get();
    } else {
      depth_loss = loss_normal_depth_.get();
    }

    problem_->AddResidualBlock(cost,
                               depth_loss,
                               CenterForImage(image).data(),
                               point_xyz_.at(point3D_id).data(),
                               &scale_it->second);
    if (!options_.use_log_scale_for_depth_map_scales) {
      problem_->SetParameterLowerBound(&scale_it->second, 0, 1e-5);
    }
    ++result_.diagnostics.num_metric_depth_residuals;
  }

  void AddDepthMapScalePriors() {
    if (!options_.use_metric_depth_constraint) return;
    for (auto& [image_id, scale] : dmap_scales_) {
      const double observation_count = static_cast<double>(
          dmap_scale_observation_counts_.count(image_id) == 0
              ? 1
              : dmap_scale_observation_counts_.at(image_id));
      const Eigen::Matrix<double, 1, 1> prior(
          options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0);
      const Eigen::Matrix<double, 1, 1> covariance(options_.scale_prior_stddev *
                                                   options_.scale_prior_stddev);
      ceres::CostFunction* cost = colmap::CovarianceWeightedCostFunctor<
          colmap::NormalPriorCostFunctor<1>>::Create(covariance, prior);
      per_image_scale_prior_losses_.push_back(
          std::make_unique<ceres::ScaledLoss>(loss_scale_prior_.get(),
                                              observation_count,
                                              ceres::DO_NOT_TAKE_OWNERSHIP));
      problem_->AddResidualBlock(
          cost, per_image_scale_prior_losses_.back().get(), &scale);
      ++result_.diagnostics.num_scale_prior_residuals;
    }
  }

  void AddCamerasAndPointsToParameterGroups() {
    solver_options_.linear_solver_ordering =
        std::make_shared<ceres::ParameterBlockOrdering>();
    ceres::ParameterBlockOrdering* ordering =
        solver_options_.linear_solver_ordering.get();
    const bool singleton =
        options_.parameter_ordering == GlobalPositioningOrdering::kSingleton;
    int group_id = 0;
    for (double& scale : scales_) {
      ordering->AddElementToGroup(&scale, group_id);
      if (singleton) ++group_id;
    }
    if (!singleton) ++group_id;

    bool had_points = false;
    for (auto& [point3D_id, xyz] : point_xyz_) {
      if (!problem_->HasParameterBlock(xyz.data())) continue;
      had_points = true;
      ordering->AddElementToGroup(xyz.data(), group_id);
      if (singleton) ++group_id;
    }
    if (!singleton && had_points) ++group_id;

    if (options_.center_mode == GlobalPositioningCenterMode::kImage) {
      for (const ImageId image_id : mapping_problem_->ImageIds()) {
        auto center_it = image_centers_.find(image_id);
        if (center_it == image_centers_.end() ||
            !problem_->HasParameterBlock(center_it->second.data())) {
          continue;
        }
        ordering->AddElementToGroup(center_it->second.data(), group_id);
        if (singleton) ++group_id;
      }
    } else {
      std::vector<FrameId> frame_ids;
      frame_ids.reserve(frame_centers_.size());
      for (const auto& [frame_id, center] : frame_centers_) {
        frame_ids.push_back(frame_id);
      }
      std::sort(frame_ids.begin(), frame_ids.end());
      for (const FrameId frame_id : frame_ids) {
        Eigen::Vector3d& center = frame_centers_.at(frame_id);
        if (!problem_->HasParameterBlock(center.data())) continue;
        ordering->AddElementToGroup(center.data(), group_id);
        if (singleton) ++group_id;
      }
    }
    for (auto& [image_id, scale] : dmap_scales_) {
      if (!problem_->HasParameterBlock(&scale)) continue;
      ordering->AddElementToGroup(&scale, group_id);
      if (singleton) ++group_id;
    }
  }

  void ParameterizeVariables() {
    if (!options_.optimize_positions) {
      for (auto& [image_id, center] : image_centers_) {
        if (problem_->HasParameterBlock(center.data())) {
          problem_->SetParameterBlockConstant(center.data());
        }
      }
      for (auto& [frame_id, center] : frame_centers_) {
        if (problem_->HasParameterBlock(center.data())) {
          problem_->SetParameterBlockConstant(center.data());
        }
      }
    }
    if (!options_.optimize_points) {
      for (auto& [point3D_id, xyz] : point_xyz_) {
        if (problem_->HasParameterBlock(xyz.data())) {
          problem_->SetParameterBlockConstant(xyz.data());
        }
      }
    }
    if (!options_.optimize_scales) {
      for (double& scale : scales_) {
        if (problem_->HasParameterBlock(&scale)) {
          problem_->SetParameterBlockConstant(&scale);
        }
      }
    }
    if (!options_.use_log_scale_for_depth_map_scales) {
      for (auto& [image_id, scale] : dmap_scales_) {
        if (problem_->HasParameterBlock(&scale)) {
          problem_->SetParameterLowerBound(&scale, 0, 1e-5);
        }
      }
    }
    if (!options_.use_metric_depth_constraint) {
      for (double& scale : scales_) {
        if (problem_->HasParameterBlock(&scale)) {
          problem_->SetParameterBlockConstant(&scale);
          break;
        }
      }
    }
    solver_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options_.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    solver_options_.num_threads =
        colmap::GetEffectiveNumThreads(solver_options_.num_threads);
  }

  void FindDepthOutliers() {
    for (const Point3DId point3D_id : mapping_problem_->Point3DIds()) {
      const TrackRecord& track = mapping_problem_->Track(point3D_id);
      FindDepthOutliers(track.observations, track.xyz);
      if (options_.use_lc_observations) {
        FindDepthOutliers(track.loop_closure_observations, track.xyz);
      }
    }
  }

  void FindDepthOutliers(const MatrixX2u& observations,
                         const Eigen::Vector3d& xyz) {
    for (Eigen::Index row = 0; row < observations.rows(); ++row) {
      const ImageId image_id = observations(row, 0);
      const std::uint32_t point2D_idx = observations(row, 1);
      if (image_ids_.count(image_id) == 0) continue;
      const ImageRecord& image = mapping_problem_->Image(image_id);
      if (!image.pose.has_pose ||
          !MaskValue(image.depth_validity, point2D_idx) ||
          point2D_idx >=
              static_cast<std::uint32_t>(image.depth_values.size()) ||
          point2D_idx >=
              static_cast<std::uint32_t>(image.depth_stddevs.size())) {
        continue;
      }
      const double prior = image.depth_values[point2D_idx];
      const double stddev = image.depth_stddevs[point2D_idx];
      if (prior <= 1e-6 || stddev <= 1e-9) continue;
      double scaled_prior = prior;
      const auto scale_it = dmap_scales_.find(image_id);
      if (scale_it != dmap_scales_.end()) {
        scaled_prior *= options_.use_log_scale_for_depth_map_scales
                            ? std::exp(scale_it->second)
                            : scale_it->second;
      }
      const Eigen::Vector3d point_camera =
          ImageRotation(image) * xyz + image.pose.translation;
      if (point_camera[2] <= 1e-6) continue;
      const double log_difference =
          std::abs(std::log(std::max(point_camera[2], 1e-6)) -
                   std::log(std::max(scaled_prior, 1e-6)));
      const double threshold = options_.filter_depth_outlier_sigma *
                               std::log(1.0 + std::max(stddev, 1e-6));
      if (log_difference >= threshold) {
        depth_outliers_.emplace(image_id, point2D_idx);
      }
    }
  }

  void PopulateResult(const ceres::Solver::Summary& summary) {
    result_.success = summary.IsSolutionUsable();
    result_.depth_map_scales = dmap_scales_;
    for (const auto& [key, index] : scale_indices_) {
      result_.final_bata_scales.emplace(key, scales_.at(index));
    }
    GlobalPositioningDiagnostics& diagnostics = result_.diagnostics;
    diagnostics.num_bata_scales = static_cast<int>(scales_.size());
    diagnostics.num_depth_map_scales = static_cast<int>(dmap_scales_.size());
    diagnostics.num_camera_centers = static_cast<int>(
        options_.center_mode == GlobalPositioningCenterMode::kImage
            ? image_centers_.size()
            : frame_centers_.size());
    diagnostics.num_point3D_parameters =
        static_cast<int>(result_.initial_point3D_xyz.size());
    diagnostics.num_residual_blocks = summary.num_residual_blocks;
    diagnostics.num_parameter_blocks = summary.num_parameter_blocks;
    diagnostics.num_parameters = summary.num_parameters;
    diagnostics.num_iterations = static_cast<int>(summary.iterations.size());
    diagnostics.termination_type = static_cast<int>(summary.termination_type);
    diagnostics.initial_cost = summary.initial_cost;
    diagnostics.final_cost = summary.final_cost;
  }

  void ConvertBackResults() {
    for (const ImageId image_id : mapping_problem_->ImageIds()) {
      ImageRecord image = mapping_problem_->Image(image_id);
      const bool has_center =
          options_.center_mode == GlobalPositioningCenterMode::kImage
              ? image_centers_.count(image_id) != 0
              : frame_centers_.count(image.frame_id) != 0;
      if (!has_center) continue;
      const Eigen::Vector3d& center = CenterForImage(image);
      const Eigen::Quaterniond rotation = ImageRotation(image);
      image.pose.translation = -(rotation * center);
      mapping_problem_->UpdateImage(image);
    }
    for (const auto& [point3D_id, xyz] : point_xyz_) {
      TrackRecord track = mapping_problem_->Track(point3D_id);
      track.xyz = xyz;
      mapping_problem_->UpdateTrack(track);
    }
  }

  const GlobalPositionerOptions& options_;
  MappingProblem* mapping_problem_;
  std::unordered_set<ImageId> image_ids_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Options solver_options_;
  std::map<Point3DId, Eigen::Vector3d> point_xyz_;
  std::unordered_map<ImageId, Eigen::Vector3d> image_centers_;
  std::unordered_map<FrameId, Eigen::Vector3d> frame_centers_;
  std::vector<double> scales_;
  std::map<std::string, std::size_t> scale_indices_;
  std::map<ImageId, double> dmap_scales_;
  std::unordered_map<ImageId, int> dmap_scale_observation_counts_;
  std::set<std::pair<ImageId, std::uint32_t>> depth_outliers_;
  std::vector<PlaybackObservation> playback_observations_;
  std::vector<ImageId> playback_image_ids_;
  std::vector<Point3DId> playback_point3D_ids_;
  std::vector<PlaybackEdge> playback_edges_;
  bool playback_topology_ready_ = false;

  std::shared_ptr<ceres::LossFunction> loss_;
  std::shared_ptr<ceres::LossFunction> calibrated_loss_;
  std::shared_ptr<ceres::LossFunction> uncalibrated_loss_;
  std::shared_ptr<ceres::LossFunction> loss_normal_geometry_;
  std::shared_ptr<ceres::LossFunction> loss_normal_depth_;
  std::shared_ptr<ceres::LossFunction> loss_lc_geometry_;
  std::shared_ptr<ceres::LossFunction> loss_lc_depth_;
  std::shared_ptr<ceres::LossFunction> loss_normal_geometry_inlier_;
  std::shared_ptr<ceres::LossFunction> loss_normal_depth_inlier_;
  std::shared_ptr<ceres::LossFunction> loss_normal_depth_outlier_;
  std::shared_ptr<ceres::LossFunction> loss_normal_geometry_track_anchor_;
  std::shared_ptr<ceres::LossFunction> loss_normal_depth_track_anchor_;
  std::shared_ptr<ceres::LossFunction> loss_scale_prior_;
  std::shared_ptr<ceres::LossFunction> loss_soft_outlier_fallback_;
  std::vector<std::unique_ptr<ceres::LossFunction>>
      per_image_scale_prior_losses_;
  std::vector<std::unique_ptr<ceres::LossFunction>>
      temporal_acceleration_losses_;
  GlobalPositioningResult result_;
};

}  // namespace

void GlobalPositionerOptions::Validate() const {
  playback.Validate();
  if (min_num_view_per_track <= 0 || random_seed < -1 ||
      !std::isfinite(random_init_scale) || random_init_scale < 0.0 ||
      !std::isfinite(uncalibrated_loss_downweight) ||
      uncalibrated_loss_downweight < 0.0 ||
      !std::isfinite(log_linear_threshold) || log_linear_threshold <= 0.0 ||
      !std::isfinite(scale_prior_stddev) || scale_prior_stddev <= 0.0 ||
      !std::isfinite(temporal_acceleration_prior_stddev) ||
      temporal_acceleration_prior_stddev <= 0.0 ||
      !std::isfinite(temporal_acceleration_prior_weight) ||
      temporal_acceleration_prior_weight < 0.0 ||
      !std::isfinite(temporal_acceleration_prior_loss_dead_zone) ||
      temporal_acceleration_prior_loss_dead_zone < 0.0 ||
      !std::isfinite(temporal_acceleration_prior_loss_huber_width) ||
      temporal_acceleration_prior_loss_huber_width <= 0.0 ||
      !std::isfinite(filter_depth_outlier_sigma) ||
      filter_depth_outlier_sigma <= 0.0 || num_threads == 0 ||
      max_num_iterations <= 0 || !std::isfinite(function_tolerance) ||
      function_tolerance < 0.0 || !std::isfinite(gradient_tolerance) ||
      gradient_tolerance < 0.0 || !std::isfinite(parameter_tolerance) ||
      parameter_tolerance < 0.0) {
    throw std::invalid_argument("invalid global positioning options");
  }
  if (use_temporal_acceleration_prior &&
      (temporal_acceleration_priors.empty() ||
       temporal_acceleration_prior_weight <= 0.0)) {
    throw std::invalid_argument(
        "enabled temporal acceleration requires priors and positive weight");
  }
  for (const TemporalAccelerationPrior& prior : temporal_acceleration_priors) {
    if (prior.prev_image_id == prior.image_id ||
        prior.prev_image_id == prior.next_image_id ||
        prior.image_id == prior.next_image_id ||
        !std::isfinite(prior.dt_prev) || prior.dt_prev <= 0.0 ||
        !std::isfinite(prior.dt_next) || prior.dt_next <= 0.0 ||
        !std::isfinite(prior.sqrt_observation_count) ||
        prior.sqrt_observation_count <= 0.0) {
      throw std::invalid_argument("invalid temporal acceleration prior");
    }
  }
  loss.Validate();
  loss_soft_outlier_fallback.Validate();
  loss_normal_geometry.Validate();
  loss_normal_depth.Validate();
  loss_lc_geometry.Validate();
  loss_lc_depth.Validate();
  loss_normal_geometry_inlier.Validate();
  loss_normal_depth_inlier.Validate();
  loss_normal_depth_outlier.Validate();
  loss_normal_geometry_track_anchor.Validate();
  loss_normal_depth_track_anchor.Validate();
  loss_scale_prior.Validate();
  for (const auto& [image_id, scale] : initial_dmap_scales) {
    if (!std::isfinite(scale) || scale <= 0.0) {
      throw std::invalid_argument("invalid initial depth-map scale");
    }
  }
}

GlobalPositioningResult RunGlobalPositioning(
    const GlobalPositionerOptions& options, MappingProblem* problem) {
  if (problem == nullptr) {
    throw std::invalid_argument("mapping problem must not be null");
  }
  return GlobalPositioner(options, problem).Solve();
}

}  // namespace vidmap
