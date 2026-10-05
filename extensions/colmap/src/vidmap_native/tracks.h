#pragma once

#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "vidmap_native/mapping_sidecars.h"

namespace vidmap {

struct TrackEstablishmentOptions {
  double intra_image_consistency_threshold = 10.0;
  int min_num_views_per_track = 3;
  int max_num_views_per_track = std::numeric_limits<int>::max();
  bool two_view_depth_gate = false;

  void Validate() const;
};

struct TrackEstablishmentResult {
  std::size_t num_full_tracks = 0;
  std::size_t num_tracks = 0;
  std::map<Point3DId, colmap::Track> full_tracks;
  std::map<Point3DId, TrackData> full_track_data;
};
TrackEstablishmentResult EstablishAndCommitTracks(
    colmap::Reconstruction&,
    const colmap::PoseGraph&,
    MappingSidecars&,
    const std::vector<ImageId>&,
    const std::vector<PairId>&,
    const TrackEstablishmentOptions&,
    bool,
    bool,
    bool);
std::shared_ptr<colmap::CorrespondenceGraph> CreateCorrespondenceGraph(
    const colmap::Reconstruction&,
    const MappingSidecars&,
    const std::vector<PairId>&);
std::shared_ptr<colmap::CorrespondenceGraph> FilterCorrespondenceGraph(
    const colmap::CorrespondenceGraph&,
    const colmap::Reconstruction&,
    const std::vector<PairId>&);

}  // namespace vidmap
