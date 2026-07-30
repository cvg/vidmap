// Track establishment over explicit regular and loop-closure observations.
#include "colmap/math/union_find.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "track_records.h"
#include "vidmap_native/tracks.h"

namespace vidmap {

using track_internal::DecodeObservation;
using track_internal::LoopClosureObservation;
using track_internal::NativeTrack;
using track_internal::Observation;
using track_internal::ToTrackMap;
using track_internal::ToTrackRecords;
using track_internal::TrackMap;

namespace {

constexpr double kDepthEpsilon = 1e-6;
using LoopClosureKey = std::pair<Point3DId, Point3DId>;

void ValidateImageDomain(const MappingProblem& problem,
                         const std::vector<ImageId>& image_ids) {
  std::unordered_set<ImageId> unique_ids;
  unique_ids.reserve(image_ids.size());
  for (const ImageId image_id : image_ids) {
    problem.Image(image_id);
    if (!unique_ids.insert(image_id).second) {
      throw std::invalid_argument("duplicate image ID in traversal order");
    }
  }
}

void ValidatePairOrder(const MappingProblem& problem,
                       const std::vector<PairId>& pair_ids) {
  std::unordered_set<PairId> unique_ids;
  unique_ids.reserve(pair_ids.size());
  for (const PairId pair_id : pair_ids) {
    const PairRecord& pair = problem.Pair(pair_id);
    if (!pair.is_valid) {
      throw std::invalid_argument("pair traversal contains an invalid pair");
    }
    if (!unique_ids.insert(pair_id).second) {
      throw std::invalid_argument("duplicate pair ID in traversal order");
    }
  }
}

void ValidatePairLoopClosureMetadata(const PairRecord& pair) {
  if (pair.are_loop_closure.size() != pair.all_matches.rows()) {
    throw std::invalid_argument(
        "loop-closure mask must be aligned with all matches");
  }
}

std::set<LoopClosureKey> CollectLoopClosureMatches(
    const MappingProblem& problem, const std::vector<PairId>& pair_order) {
  std::set<LoopClosureKey> loop_closure_matches;
  for (const PairId pair_id : pair_order) {
    const PairRecord& pair = problem.Pair(pair_id);
    ValidatePairLoopClosureMetadata(pair);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      if (pair.are_loop_closure[row] == 0) continue;
      const Point3DId observation1 =
          EncodeObservationKey(pair.image_id1, pair.all_matches(row, 0));
      const Point3DId observation2 =
          EncodeObservationKey(pair.image_id2, pair.all_matches(row, 1));
      loop_closure_matches.emplace(observation1, observation2);
      loop_closure_matches.emplace(observation2, observation1);
    }
  }
  return loop_closure_matches;
}

bool HasValidDepthPrior(const MappingProblem& problem,
                        const Observation& observation) {
  const ImageRecord& image = problem.Image(observation.image_id);
  const auto feature_id = static_cast<Eigen::Index>(observation.feature_id);
  return feature_id < image.depth_validity.size() &&
         image.depth_validity[feature_id] != 0 &&
         feature_id < image.depth_values.size() &&
         image.depth_values[feature_id] > kDepthEpsilon;
}

TrackMap EstablishTracks(const MappingProblem& problem,
                         const std::vector<ImageId>& image_order,
                         const std::vector<PairId>& pair_order,
                         const TrackEstablishmentOptions& options,
                         const std::set<LoopClosureKey>& ignored_matches) {
  colmap::UnionFind<Point3DId> union_find;
  const auto should_ignore = [&ignored_matches](const Point3DId observation1,
                                                const Point3DId observation2) {
    return ignored_matches.count({observation1, observation2}) > 0;
  };

  for (const PairId pair_id : pair_order) {
    const PairRecord& pair = problem.Pair(pair_id);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      const Point3DId observation1 =
          EncodeObservationKey(pair.image_id1, pair.all_matches(row, 0));
      const Point3DId observation2 =
          EncodeObservationKey(pair.image_id2, pair.all_matches(row, 1));
      if (should_ignore(observation1, observation2)) continue;
      if (observation2 < observation1) {
        union_find.Union(observation1, observation2);
      } else {
        union_find.Union(observation2, observation1);
      }
    }
  }

  std::unordered_map<Point3DId, std::unordered_set<Point3DId>> track_map;
  for (const PairId pair_id : pair_order) {
    const PairRecord& pair = problem.Pair(pair_id);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      const Point3DId observation1 =
          EncodeObservationKey(pair.image_id1, pair.all_matches(row, 0));
      const Point3DId observation2 =
          EncodeObservationKey(pair.image_id2, pair.all_matches(row, 1));
      if (should_ignore(observation1, observation2)) continue;
      const Point3DId track_id1 = union_find.Find(observation1);
      const Point3DId track_id2 = union_find.Find(observation2);
      if (track_id1 == track_id2) {
        track_map[track_id1].insert(observation1);
        track_map[track_id1].insert(observation2);
      }
    }
  }

  TrackMap candidates;
  std::vector<std::pair<std::size_t, Point3DId>> track_lengths;
  for (const auto& [track_id, encoded_observations] : track_map) {
    std::unordered_map<ImageId, std::vector<Eigen::Vector2d>> image_points;
    NativeTrack track;
    bool consistent = true;
    for (const Point3DId encoded_observation : encoded_observations) {
      const Observation observation = DecodeObservation(encoded_observation);
      const ImageRecord& image = problem.Image(observation.image_id);
      if (observation.feature_id >= image.NumFeatures()) {
        throw std::invalid_argument("track match references a missing feature");
      }
      const Eigen::Vector2d point = image.keypoints.row(observation.feature_id);
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
    track_lengths.emplace_back(track.observations.size(), track_id);
    candidates.emplace(track_id, std::move(track));
  }

  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());
  std::unordered_map<ImageId, std::size_t> tracks_per_image;
  std::size_t images_left = image_order.size();
  TrackMap selected;
  for (const auto& [track_length, track_id] : track_lengths) {
    NativeTrack& track = candidates.at(track_id);
    const bool should_add = std::any_of(
        track.observations.begin(),
        track.observations.end(),
        [&](const Observation& observation) {
          return tracks_per_image[observation.image_id] <=
                 static_cast<std::size_t>(options.required_tracks_per_view);
        });
    if (!should_add) continue;

    for (const Observation& observation : track.observations) {
      std::size_t& count = tracks_per_image[observation.image_id];
      if (count == static_cast<std::size_t>(options.required_tracks_per_view)) {
        --images_left;
      }
      ++count;
    }
    selected.emplace(track_id, std::move(track));
    if (images_left == 0) break;
  }
  return selected;
}

void AppendLoopClosureObservationsToMap(const MappingProblem& problem,
                                        const std::vector<PairId>& pair_order,
                                        TrackMap* tracks) {
  std::unordered_map<Point3DId, Point3DId> observation_to_track;
  for (const auto& [track_id, track] : *tracks) {
    for (const Observation& observation : track.observations) {
      observation_to_track.emplace(
          EncodeObservationKey(observation.image_id, observation.feature_id),
          track_id);
    }
  }

  for (const PairId pair_id : pair_order) {
    const PairRecord& pair = problem.Pair(pair_id);
    ValidatePairLoopClosureMetadata(pair);
    for (Eigen::Index index = 0; index < pair.inlier_indices.size(); ++index) {
      const int row = pair.inlier_indices[index];
      if (pair.are_loop_closure[row] == 0) continue;
      const Observation observation1 = {pair.image_id1,
                                        pair.all_matches(row, 0)};
      const Observation observation2 = {pair.image_id2,
                                        pair.all_matches(row, 1)};
      const Point3DId key1 =
          EncodeObservationKey(observation1.image_id, observation1.feature_id);
      const Point3DId key2 =
          EncodeObservationKey(observation2.image_id, observation2.feature_id);
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
      required_tracks_per_view < 0) {
    throw std::invalid_argument("invalid track establishment options");
  }
}

void TrackProblemFilterOptions::Validate() const {
  if (min_num_views_per_track <= 0 ||
      max_num_views_per_track < min_num_views_per_track) {
    throw std::invalid_argument("invalid track problem filter options");
  }
}

Point3DId EncodeObservationKey(const ImageId image_id,
                               const std::uint32_t feature_id) {
  return (static_cast<Point3DId>(image_id) << 32) |
         static_cast<Point3DId>(feature_id);
}

std::vector<TrackRecord> EstablishTracksFromCorrGraph(
    const MappingProblem& problem,
    const std::vector<ImageId>& image_order,
    const std::vector<PairId>& pair_order,
    const TrackEstablishmentOptions& options,
    const bool loop_closure_second_pass,
    const std::vector<PairId>& loop_closure_pair_order) {
  problem.Validate();
  options.Validate();
  ValidateImageDomain(problem, image_order);
  ValidatePairOrder(problem, pair_order);

  std::set<LoopClosureKey> ignored_matches;
  const std::vector<PairId>& lc_pair_order =
      loop_closure_pair_order.empty() ? pair_order : loop_closure_pair_order;
  TrackEstablishmentOptions effective_options = options;
  if (loop_closure_second_pass) {
    ValidatePairOrder(problem, lc_pair_order);
    ignored_matches = CollectLoopClosureMatches(problem, lc_pair_order);
    effective_options.required_tracks_per_view =
        std::numeric_limits<int>::max();
  }

  TrackMap tracks = EstablishTracks(
      problem, image_order, pair_order, effective_options, ignored_matches);
  if (loop_closure_second_pass) {
    AppendLoopClosureObservationsToMap(problem, lc_pair_order, &tracks);
  }
  return ToTrackRecords(tracks);
}

std::vector<TrackRecord> AppendLoopClosureObservations(
    const MappingProblem& problem,
    const std::vector<PairId>& pair_order,
    const std::vector<TrackRecord>& track_records) {
  ValidatePairOrder(problem, pair_order);
  TrackMap tracks = ToTrackMap(track_records);
  AppendLoopClosureObservationsToMap(problem, pair_order, &tracks);
  return ToTrackRecords(tracks);
}

std::vector<TrackRecord> FilterTracksForProblem(
    const MappingProblem& problem,
    const std::vector<ImageId>& registered_image_ids,
    const std::vector<TrackRecord>& track_records,
    const TrackProblemFilterOptions& options) {
  options.Validate();
  ValidateImageDomain(problem, registered_image_ids);
  const TrackMap tracks_full = ToTrackMap(track_records);
  std::vector<std::pair<std::size_t, Point3DId>> track_lengths;
  for (const auto& [track_id, track] : tracks_full) {
    if (track.observations.size() <
            static_cast<std::size_t>(options.min_num_views_per_track) ||
        track.observations.size() >
            static_cast<std::size_t>(options.max_num_views_per_track)) {
      continue;
    }
    track_lengths.emplace_back(track.observations.size(), track_id);
  }
  std::sort(track_lengths.begin(), track_lengths.end(), std::greater<>());

  std::unordered_set<ImageId> registered_image_id_set(
      registered_image_ids.begin(), registered_image_ids.end());
  TrackMap selected;
  for (const auto& [track_length, track_id] : track_lengths) {
    const NativeTrack& source = tracks_full.at(track_id);
    NativeTrack candidate;
    std::unordered_set<ImageId> distinct_image_ids;
    for (const Observation& observation : source.observations) {
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
                      [&](const Observation& observation) {
                        return HasValidDepthPrior(problem, observation);
                      });
      const bool loop_closure_depths_valid = std::all_of(
          candidate.loop_closure_observations.begin(),
          candidate.loop_closure_observations.end(),
          [&](const LoopClosureObservation& observation) {
            return HasValidDepthPrior(problem, observation.observation);
          });
      if (!regular_depths_valid || !loop_closure_depths_valid) continue;
    }
    selected.emplace(track_id, std::move(candidate));
  }
  return ToTrackRecords(selected);
}

}  // namespace vidmap
