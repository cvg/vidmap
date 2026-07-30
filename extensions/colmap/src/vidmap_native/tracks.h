#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "vidmap_native/mapping_problem.h"

namespace vidmap {

struct TrackEstablishmentOptions {
  double intra_image_consistency_threshold = 10.0;
  int min_num_views_per_track = 3;
  int required_tracks_per_view = std::numeric_limits<int>::max();

  void Validate() const;
};

struct TrackProblemFilterOptions {
  int min_num_views_per_track = 3;
  int max_num_views_per_track = std::numeric_limits<int>::max();
  bool two_view_depth_gate = false;

  void Validate() const;
};

struct TrackFilterResult {
  std::vector<TrackRecord> tracks;
  std::size_t counter = 0;
};

Point3DId EncodeObservationKey(ImageId image_id, std::uint32_t feature_id);

std::vector<TrackRecord> EstablishTracksFromCorrGraph(
    const MappingProblem& problem,
    const std::vector<ImageId>& image_order,
    const std::vector<PairId>& pair_order,
    const TrackEstablishmentOptions& options,
    bool loop_closure_second_pass,
    const std::vector<PairId>& loop_closure_pair_order = {});

std::vector<TrackRecord> AppendLoopClosureObservations(
    const MappingProblem& problem,
    const std::vector<PairId>& pair_order,
    const std::vector<TrackRecord>& tracks);

std::vector<TrackRecord> FilterTracksForProblem(
    const MappingProblem& problem,
    const std::vector<ImageId>& registered_image_ids,
    const std::vector<TrackRecord>& tracks_full,
    const TrackProblemFilterOptions& options);

TrackFilterResult FilterTracksByAngle(const MappingProblem& problem,
                                      const std::vector<TrackRecord>& tracks,
                                      double max_angle_error_deg = 1.0);

TrackFilterResult FilterTrackTriangulationAngle(
    const MappingProblem& problem,
    const std::vector<TrackRecord>& tracks,
    double min_angle_deg = 1.0);

}  // namespace vidmap
