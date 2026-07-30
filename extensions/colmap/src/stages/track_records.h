#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "vidmap_native/tracks.h"

namespace vidmap {
namespace track_internal {

struct Observation {
  ImageId image_id;
  std::uint32_t feature_id;
};

struct LoopClosureObservation {
  Observation observation;
  std::optional<Observation> anchor;
};

struct NativeTrack {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Matrix<std::uint8_t, 3, 1> color =
      Eigen::Matrix<std::uint8_t, 3, 1>::Zero();
  double error = -1.0;
  std::vector<Observation> observations;
  std::vector<LoopClosureObservation> loop_closure_observations;
};

using TrackMap = std::unordered_map<Point3DId, NativeTrack>;

Observation DecodeObservation(Point3DId encoded);
TrackMap ToTrackMap(const std::vector<TrackRecord>& records);
std::vector<TrackRecord> ToTrackRecords(const TrackMap& tracks);

}  // namespace track_internal
}  // namespace vidmap
