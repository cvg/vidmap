// Track establishment over explicit regular and loop-closure observations.
#include "colmap/math/union_find.h"
#include "colmap/scene/two_view_geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "vidmap_native/tracks.h"

namespace vidmap {

using colmap::TrackElement;

namespace {

constexpr double kDepthEpsilon = 1e-6;
using LoopClosureKey = std::pair<Point3DId, Point3DId>;

struct LoopClosureObservation {
  TrackElement observation;
  TrackElement anchor;
};

struct NativeTrack {
  std::vector<TrackElement> observations;
  std::vector<LoopClosureObservation> loop_closure_observations;
};

using TrackMap = std::unordered_map<Point3DId, NativeTrack>;

Point3DId EncodeObservationKey(ImageId image_id, std::uint32_t feature_id) {
  return (static_cast<Point3DId>(image_id) << 32) | feature_id;
}

TrackElement DecodeObservation(Point3DId encoded) {
  return {static_cast<ImageId>(encoded >> 32),
          static_cast<std::uint32_t>(encoded & 0xFFFFFFFFULL)};
}

TrackData LoopClosureData(const NativeTrack& source) {
  TrackData data;
  data.loop_closure_observations.resize(source.loop_closure_observations.size(),
                                        2);
  data.loop_closure_anchors.resize(source.loop_closure_observations.size(), 2);
  for (std::size_t i = 0; i < source.loop_closure_observations.size(); ++i) {
    const auto& lc = source.loop_closure_observations[i];
    data.loop_closure_observations.row(i) << lc.observation.image_id,
        lc.observation.point2D_idx;
    data.loop_closure_anchors.row(i) << lc.anchor.image_id,
        lc.anchor.point2D_idx;
  }
  return data;
}

void ValidateImageDomain(const colmap::Reconstruction& reconstruction,
                         const std::vector<ImageId>& image_ids) {
  std::unordered_set<ImageId> unique_ids;
  unique_ids.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    reconstruction.Image(image_id);
    if (!unique_ids.insert(image_id).second) {
      throw std::invalid_argument("duplicate image ID in traversal order");
    }
  }
}

void ValidatePairOrder(const colmap::PoseGraph& graph,
                       const MappingSidecars& sidecars,
                       const std::vector<PairId>& pair_ids) {
  std::unordered_set<PairId> unique_ids;
  unique_ids.reserve(pair_ids.size());
  for (const PairId pair_id : pair_ids) {
    sidecars.pairs.at(pair_id);
    if (!graph.IsValid(pair_id)) {
      throw std::invalid_argument("pair traversal contains an invalid pair");
    }
    if (!unique_ids.insert(pair_id).second) {
      throw std::invalid_argument("duplicate pair ID in traversal order");
    }
  }
}

void ValidatePairLoopClosureMetadata(const PairData& pair) {
  if (pair.are_loop_closure.size() != pair.all_matches.rows()) {
    throw std::invalid_argument(
        "loop-closure mask must be aligned with all matches");
  }
}

std::set<LoopClosureKey> CollectLoopClosureMatches(
    const MappingSidecars& sidecars, const std::vector<PairId>& pair_order) {
  std::set<LoopClosureKey> loop_closure_matches;
  for (const PairId pair_id : pair_order) {
    const auto& pair = sidecars.pairs.at(pair_id);
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    ValidatePairLoopClosureMetadata(pair);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      if (pair.are_loop_closure[row] == 0) continue;
      const Point3DId observation1 =
          EncodeObservationKey(image_id1, pair.all_matches(row, 0));
      const Point3DId observation2 =
          EncodeObservationKey(image_id2, pair.all_matches(row, 1));
      loop_closure_matches.emplace(observation1, observation2);
    }
  }
  return loop_closure_matches;
}

bool HasValidDepthPrior(const MappingSidecars& sidecars,
                        const TrackElement& observation) {
  const auto& data = sidecars.images.at(observation.image_id);
  const auto feature_id = static_cast<Eigen::Index>(observation.point2D_idx);
  return feature_id < data.depth_validity.size() &&
         data.depth_validity[feature_id] != 0 &&
         feature_id < data.depth_values.size() &&
         data.depth_values[feature_id] > kDepthEpsilon;
}

TrackMap EstablishTracks(const colmap::Reconstruction& reconstruction,
                         const MappingSidecars& sidecars,
                         const std::vector<PairId>& pair_order,
                         const TrackEstablishmentOptions& options,
                         const std::set<LoopClosureKey>& ignored_matches) {
  colmap::UnionFind<Point3DId> union_find;
  const auto should_ignore = [&ignored_matches](const Point3DId observation1,
                                                const Point3DId observation2) {
    return ignored_matches.count({observation1, observation2}) > 0;
  };

  for (const PairId pair_id : pair_order) {
    const auto& pair = sidecars.pairs.at(pair_id);
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      const Point3DId observation1 =
          EncodeObservationKey(image_id1, pair.all_matches(row, 0));
      const Point3DId observation2 =
          EncodeObservationKey(image_id2, pair.all_matches(row, 1));
      if (should_ignore(observation1, observation2)) continue;
      if (observation2 < observation1) {
        union_find.Union(observation1, observation2);
      } else {
        union_find.Union(observation2, observation1);
      }
    }
  }

  union_find.Compress();
  std::unordered_map<Point3DId, std::vector<Point3DId>> track_map;
  for (const auto& [observation, root] : union_find.Parents()) {
    track_map[root].push_back(observation);
  }

  TrackMap candidates;
  for (const auto& [track_id, encoded_observations] : track_map) {
    std::unordered_map<ImageId, std::vector<Eigen::Vector2d>> image_points;
    NativeTrack track;
    bool consistent = true;
    for (const Point3DId encoded_observation : encoded_observations) {
      const TrackElement observation = DecodeObservation(encoded_observation);
      const auto& image = reconstruction.Image(observation.image_id);
      if (observation.point2D_idx >= image.NumPoints2D()) {
        throw std::invalid_argument("track match references a missing feature");
      }
      const Eigen::Vector2d point = image.Point2D(observation.point2D_idx).xy;
      auto image_it = image_points.find(observation.image_id);
      if (image_it != image_points.end()) {
        const double squared_threshold =
            options.intra_image_consistency_threshold *
            options.intra_image_consistency_threshold;
        for (const Eigen::Vector2d& existing_point : image_it->second) {
          if ((existing_point - point).squaredNorm() > squared_threshold) {
            consistent = false;
            break;
          }
        }
        if (!consistent) break;
        image_it->second.push_back(point);
      } else {
        image_points[observation.image_id].push_back(point);
      }
      track.observations.push_back(observation);
    }
    if (!consistent ||
        image_points.size() <
            static_cast<std::size_t>(options.min_num_views_per_track)) {
      continue;
    }
    candidates.emplace(track_id, std::move(track));
  }

  return candidates;
}

void AppendLoopClosureObservationsToMap(const MappingSidecars& sidecars,
                                        const std::vector<PairId>& pair_order,
                                        TrackMap* tracks) {
  std::unordered_map<Point3DId, Point3DId> observation_to_track;
  for (const auto& [track_id, track] : *tracks) {
    for (const TrackElement& observation : track.observations) {
      observation_to_track.emplace(
          EncodeObservationKey(observation.image_id, observation.point2D_idx),
          track_id);
    }
  }

  for (const PairId pair_id : pair_order) {
    const auto& pair = sidecars.pairs.at(pair_id);
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    ValidatePairLoopClosureMetadata(pair);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      if (pair.are_loop_closure[row] == 0) continue;
      const TrackElement observation1 = {image_id1, pair.all_matches(row, 0)};
      const TrackElement observation2 = {image_id2, pair.all_matches(row, 1)};
      const Point3DId key1 =
          EncodeObservationKey(observation1.image_id, observation1.point2D_idx);
      const Point3DId key2 =
          EncodeObservationKey(observation2.image_id, observation2.point2D_idx);
      const auto track1_it = observation_to_track.find(key1);
      const auto track2_it = observation_to_track.find(key2);
      const bool has_track1 = track1_it != observation_to_track.end();
      const bool has_track2 = track2_it != observation_to_track.end();

      if (!has_track1 && !has_track2) {
        NativeTrack track1;
        track1.observations.push_back(observation1);
        track1.loop_closure_observations.push_back(
            {observation2, observation1});
        NativeTrack track2;
        track2.observations.push_back(observation2);
        track2.loop_closure_observations.push_back(
            {observation1, observation2});
        if (!tracks->emplace(key1, std::move(track1)).second ||
            !tracks->emplace(key2, std::move(track2)).second) {
          throw std::logic_error("loop-closure track ID collision");
        }
        observation_to_track[key1] = key1;
        observation_to_track[key2] = key2;
      } else if (has_track1 && has_track2) {
        if (track1_it->second != track2_it->second) {
          tracks->at(track1_it->second)
              .loop_closure_observations.push_back(
                  {observation2, observation1});
          tracks->at(track2_it->second)
              .loop_closure_observations.push_back(
                  {observation1, observation2});
        }
      } else if (has_track1) {
        tracks->at(track1_it->second)
            .loop_closure_observations.push_back({observation2, observation1});
      } else {
        tracks->at(track2_it->second)
            .loop_closure_observations.push_back({observation1, observation2});
      }
    }
  }
}

}  // namespace

void TrackEstablishmentOptions::Validate() const {
  if (!std::isfinite(intra_image_consistency_threshold) ||
      intra_image_consistency_threshold < 0.0 || min_num_views_per_track <= 0 ||
      max_num_views_per_track < min_num_views_per_track) {
    throw std::invalid_argument("invalid track establishment options");
  }
}

namespace {

TrackMap EstablishTracksFromCorrGraph(
    const colmap::Reconstruction& reconstruction,
    const colmap::PoseGraph& graph,
    const MappingSidecars& sidecars,
    const std::vector<ImageId>& image_order,
    const std::vector<PairId>& pair_order,
    const TrackEstablishmentOptions& options,
    const bool loop_closure_second_pass) {
  sidecars.Validate(reconstruction);
  options.Validate();
  ValidateImageDomain(reconstruction, image_order);
  ValidatePairOrder(graph, sidecars, pair_order);

  std::set<LoopClosureKey> ignored_matches;
  if (loop_closure_second_pass) {
    ignored_matches = CollectLoopClosureMatches(sidecars, pair_order);
  }

  TrackMap tracks = EstablishTracks(
      reconstruction, sidecars, pair_order, options, ignored_matches);
  if (loop_closure_second_pass) {
    AppendLoopClosureObservationsToMap(sidecars, pair_order, &tracks);
  }
  return tracks;
}

TrackMap FilterTracksForProblem(
    const MappingSidecars& sidecars,
    const std::vector<ImageId>& registered_image_ids,
    const TrackMap& tracks_full,
    const TrackEstablishmentOptions& options) {
  std::unordered_set<ImageId> registered_image_id_set(
      registered_image_ids.begin(), registered_image_ids.end());
  TrackMap selected;
  for (const auto& [track_id, source] : tracks_full) {
    if (source.observations.size() <
            static_cast<std::size_t>(options.min_num_views_per_track) ||
        source.observations.size() >
            static_cast<std::size_t>(options.max_num_views_per_track)) {
      continue;
    }
    NativeTrack candidate;
    std::unordered_set<ImageId> distinct_image_ids;
    for (const TrackElement& observation : source.observations) {
      if (registered_image_id_set.count(observation.image_id) == 0) continue;
      candidate.observations.push_back(observation);
      distinct_image_ids.insert(observation.image_id);
    }
    for (const LoopClosureObservation& observation :
         source.loop_closure_observations) {
      if (registered_image_id_set.count(observation.observation.image_id) ==
          0) {
        continue;
      }
      candidate.loop_closure_observations.push_back(observation);
      distinct_image_ids.insert(observation.observation.image_id);
    }
    if (candidate.observations.size() <
        static_cast<std::size_t>(options.min_num_views_per_track)) {
      continue;
    }
    if (options.two_view_depth_gate && distinct_image_ids.size() == 2) {
      const bool regular_depths_valid =
          std::all_of(candidate.observations.begin(),
                      candidate.observations.end(),
                      [&](const TrackElement& observation) {
                        return HasValidDepthPrior(sidecars, observation);
                      });
      const bool loop_closure_depths_valid = std::all_of(
          candidate.loop_closure_observations.begin(),
          candidate.loop_closure_observations.end(),
          [&](const LoopClosureObservation& observation) {
            return HasValidDepthPrior(sidecars, observation.observation);
          });
      if (!regular_depths_valid || !loop_closure_depths_valid) continue;
    }
    selected.emplace(track_id, std::move(candidate));
  }
  return selected;
}

}  // namespace

TrackEstablishmentResult EstablishAndCommitTracks(
    colmap::Reconstruction& reconstruction,
    const colmap::PoseGraph& graph,
    MappingSidecars& sidecars,
    const std::vector<ImageId>& image_order,
    const std::vector<PairId>& pair_order,
    const TrackEstablishmentOptions& options,
    bool loop_closure_second_pass,
    bool include_loop_closure_observations,
    bool capture_tracks) {
  auto full = EstablishTracksFromCorrGraph(reconstruction,
                                           graph,
                                           sidecars,
                                           image_order,
                                           pair_order,
                                           options,
                                           loop_closure_second_pass);
  auto selected = FilterTracksForProblem(sidecars, image_order, full, options);
  TrackEstablishmentResult result;
  result.num_full_tracks = full.size();
  result.num_tracks = selected.size();
  if (capture_tracks) {
    for (const auto& [id, source] : full) {
      colmap::Track track;
      track.SetElements(source.observations);
      result.full_tracks.emplace(id, std::move(track));
      result.full_track_data.emplace(id, LoopClosureData(source));
    }
  }
  for (const auto id : reconstruction.Point3DIds())
    reconstruction.DeletePoint3D(id);
  sidecars.tracks.clear();
  for (const auto& [id, source] : selected) {
    colmap::Point3D point;
    point.xyz.setZero();
    point.track.SetElements(source.observations);
    reconstruction.AddPoint3D(id, point);
    if (!include_loop_closure_observations ||
        source.loop_closure_observations.empty())
      continue;
    sidecars.tracks.emplace(id, LoopClosureData(source));
  }
  return result;
}

std::shared_ptr<colmap::CorrespondenceGraph> CreateCorrespondenceGraph(
    const colmap::Reconstruction& reconstruction,
    const MappingSidecars& sidecars,
    const std::vector<PairId>& pair_order) {
  auto result = std::make_shared<colmap::CorrespondenceGraph>();
  for (const auto& [id, image] : reconstruction.Images())
    result->AddImage(id, image.NumPoints2D());
  for (const auto id : pair_order) {
    const auto& pair = sidecars.pairs.at(id);
    const auto [id1, id2] = colmap::PairIdToImagePair(id);
    if (!reconstruction.ExistsImage(id1) || !reconstruction.ExistsImage(id2) ||
        pair.inlier_indices.size() == 0)
      continue;
    colmap::TwoViewGeometry geometry;
    geometry.config = colmap::TwoViewGeometry::CALIBRATED;
    for (Eigen::Index i = 0; i < pair.inlier_indices.size(); ++i) {
      const auto row = pair.inlier_indices[i];
      geometry.inlier_matches.push_back(
          {pair.all_matches(row, 0), pair.all_matches(row, 1)});
    }
    result->AddTwoViewGeometry(id1, id2, geometry);
  }
  result->Finalize();
  return result;
}
std::shared_ptr<colmap::CorrespondenceGraph> FilterCorrespondenceGraph(
    const colmap::CorrespondenceGraph& source,
    const colmap::Reconstruction& reconstruction,
    const std::vector<PairId>& pair_order) {
  auto result = std::make_shared<colmap::CorrespondenceGraph>();
  for (const auto& [id, image] : reconstruction.Images())
    result->AddImage(id, image.NumPoints2D());
  for (const auto id : pair_order) {
    const auto [first, second] = colmap::PairIdToImagePair(id);
    if (!reconstruction.ExistsImage(first) ||
        !reconstruction.ExistsImage(second) ||
        source.NumMatchesBetweenImages(first, second) == 0)
      continue;
    result->AddTwoViewGeometry(
        first, second, source.ExtractTwoViewGeometry(first, second, true));
  }
  result->Finalize();
  return result;
}
}  // namespace vidmap
