#include "colmap/geometry/triangulation.h"
#include "colmap/math/math.h"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "track_records.h"
#include "vidmap_native/conversion.h"
#include "vidmap_native/tracks.h"

namespace vidmap {

using track_internal::NativeTrack;
using track_internal::Observation;
using track_internal::ToTrackMap;
using track_internal::ToTrackRecords;
using track_internal::TrackMap;

namespace {

constexpr double kProjectionEpsilon = 1e-12;

}  // namespace

TrackFilterResult FilterTracksByAngle(
    const MappingProblem& problem,
    const std::vector<TrackRecord>& track_records,
    const double max_angle_error_deg) {
  if (!std::isfinite(max_angle_error_deg) || max_angle_error_deg < 0.0) {
    throw std::invalid_argument("maximum angle error must be non-negative");
  }
  TrackMap tracks = ToTrackMap(track_records);
  const double threshold = std::cos(colmap::DegToRad(max_angle_error_deg));
  const double uncalibrated_threshold =
      std::cos(colmap::DegToRad(max_angle_error_deg * 2.0));
  std::size_t counter = 0;
  for (auto& [track_id, track] : tracks) {
    std::vector<Observation> filtered_observations;
    for (const Observation& observation : track.observations) {
      const ImageRecord& image = problem.Image(observation.image_id);
      if (!image.pose.has_pose ||
          observation.feature_id >=
              static_cast<std::uint32_t>(image.bearings.rows())) {
        throw std::invalid_argument(
            "angle filtering requires posed images and aligned bearings");
      }
      Eigen::Vector3d projected = ToColmapPose(image.pose) * track.xyz;
      if (projected.z() < kProjectionEpsilon) continue;
      projected.normalize();
      const double camera_threshold =
          problem.Camera(image.camera_id).has_prior_focal_length
              ? threshold
              : uncalibrated_threshold;
      const Eigen::Vector3d bearing =
          image.bearings.row(observation.feature_id);
      if (projected.dot(bearing) > camera_threshold) {
        filtered_observations.push_back(observation);
      }
    }
    if (filtered_observations.size() != track.observations.size()) {
      ++counter;
      track.observations = std::move(filtered_observations);
    }
  }
  return {ToTrackRecords(tracks), counter};
}

TrackFilterResult FilterTrackTriangulationAngle(
    const MappingProblem& problem,
    const std::vector<TrackRecord>& track_records,
    const double min_angle_deg) {
  if (!std::isfinite(min_angle_deg) || min_angle_deg < 0.0) {
    throw std::invalid_argument(
        "minimum triangulation angle must be non-negative");
  }
  TrackMap tracks = ToTrackMap(track_records);
  const double min_angle_rad = colmap::DegToRad(min_angle_deg);
  std::unordered_map<ImageId, Eigen::Vector3d> projection_centers;
  std::size_t counter = 0;
  for (auto& [track_id, track] : tracks) {
    bool keep_track = false;
    for (std::size_t index1 = 0;
         index1 < track.observations.size() && !keep_track;
         ++index1) {
      const ImageId image_id1 = track.observations[index1].image_id;
      auto center1_it = projection_centers.find(image_id1);
      if (center1_it == projection_centers.end()) {
        const PoseRecord& pose = problem.Image(image_id1).pose;
        if (!pose.has_pose) {
          throw std::invalid_argument(
              "triangulation filtering requires posed images");
        }
        const colmap::Rigid3d cam_from_world = ToColmapPose(pose);
        center1_it = projection_centers
                         .emplace(image_id1,
                                  cam_from_world.rotation().inverse() *
                                      -cam_from_world.translation())
                         .first;
      }
      for (std::size_t index2 = 0; index2 < index1; ++index2) {
        const ImageId image_id2 = track.observations[index2].image_id;
        auto center2_it = projection_centers.find(image_id2);
        if (center2_it == projection_centers.end()) {
          const PoseRecord& pose = problem.Image(image_id2).pose;
          if (!pose.has_pose) {
            throw std::invalid_argument(
                "triangulation filtering requires posed images");
          }
          const colmap::Rigid3d cam_from_world = ToColmapPose(pose);
          center2_it = projection_centers
                           .emplace(image_id2,
                                    cam_from_world.rotation().inverse() *
                                        -cam_from_world.translation())
                           .first;
        }
        if (colmap::CalculateTriangulationAngle(center1_it->second,
                                                center2_it->second,
                                                track.xyz) >= min_angle_rad) {
          keep_track = true;
          break;
        }
      }
    }
    if (!keep_track) {
      ++counter;
      track.observations.clear();
    }
  }
  return {ToTrackRecords(tracks), counter};
}

}  // namespace vidmap
