#include "track_records.h"

#include <algorithm>
#include <stdexcept>

namespace vidmap {
namespace track_internal {
namespace {

MatrixX2u ToObservationMatrix(const std::vector<Observation>& observations) {
  MatrixX2u values(observations.size(), 2);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    values(index, 0) = observations[index].image_id;
    values(index, 1) = observations[index].feature_id;
  }
  return values;
}

std::vector<Observation> FromObservationMatrix(const MatrixX2u& values) {
  std::vector<Observation> observations;
  observations.reserve(values.rows());
  for (Eigen::Index row = 0; row < values.rows(); ++row) {
    observations.push_back({values(row, 0), values(row, 1)});
  }
  return observations;
}

MatrixX2u LoopClosureObservationMatrix(
    const std::vector<LoopClosureObservation>& observations) {
  MatrixX2u values(observations.size(), 2);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    values(index, 0) = observations[index].observation.image_id;
    values(index, 1) = observations[index].observation.feature_id;
  }
  return values;
}

MatrixX2u LoopClosureAnchorMatrix(
    const std::vector<LoopClosureObservation>& observations) {
  const bool has_anchors = std::any_of(
      observations.begin(), observations.end(), [](const auto& observation) {
        return observation.anchor.has_value();
      });
  const bool has_missing_anchors = std::any_of(
      observations.begin(), observations.end(), [](const auto& observation) {
        return !observation.anchor.has_value();
      });
  if (has_anchors && has_missing_anchors) {
    throw std::logic_error("partially populated loop-closure provenance");
  }
  if (!has_anchors) return MatrixX2u(0, 2);

  MatrixX2u values(observations.size(), 2);
  for (std::size_t index = 0; index < observations.size(); ++index) {
    values(index, 0) = observations[index].anchor->image_id;
    values(index, 1) = observations[index].anchor->feature_id;
  }
  return values;
}

std::vector<LoopClosureObservation> FromLoopClosureMatrices(
    const MatrixX2u& observations, const MatrixX2u& anchors) {
  std::vector<LoopClosureObservation> values;
  values.reserve(observations.rows());
  for (Eigen::Index row = 0; row < observations.rows(); ++row) {
    std::optional<Observation> anchor;
    if (anchors.rows() != 0) {
      anchor = Observation{anchors(row, 0), anchors(row, 1)};
    }
    values.push_back({{observations(row, 0), observations(row, 1)}, anchor});
  }
  return values;
}

TrackRecord ToRecord(const Point3DId point3D_id, const NativeTrack& track) {
  TrackRecord record;
  record.point3D_id = point3D_id;
  record.xyz = track.xyz;
  record.color = track.color;
  record.error = track.error;
  record.observations = ToObservationMatrix(track.observations);
  record.loop_closure_observations =
      LoopClosureObservationMatrix(track.loop_closure_observations);
  record.loop_closure_anchors =
      LoopClosureAnchorMatrix(track.loop_closure_observations);
  return record;
}

NativeTrack FromRecord(const TrackRecord& record) {
  record.Validate();
  NativeTrack track;
  track.xyz = record.xyz;
  track.color = record.color;
  track.error = record.error;
  track.observations = FromObservationMatrix(record.observations);
  track.loop_closure_observations = FromLoopClosureMatrices(
      record.loop_closure_observations, record.loop_closure_anchors);
  return track;
}

}  // namespace

Observation DecodeObservation(const Point3DId encoded) {
  return {static_cast<ImageId>(encoded >> 32),
          static_cast<std::uint32_t>(encoded & 0xFFFFFFFFULL)};
}

TrackMap ToTrackMap(const std::vector<TrackRecord>& records) {
  TrackMap tracks;
  for (const TrackRecord& record : records) {
    const auto [unused, inserted] =
        tracks.emplace(record.point3D_id, FromRecord(record));
    if (!inserted) {
      throw std::invalid_argument("duplicate track ID");
    }
  }
  return tracks;
}

std::vector<TrackRecord> ToTrackRecords(const TrackMap& tracks) {
  std::vector<TrackRecord> records;
  records.reserve(tracks.size());
  for (const auto& [point3D_id, track] : tracks) {
    records.push_back(ToRecord(point3D_id, track));
  }
  return records;
}

}  // namespace track_internal
}  // namespace vidmap
