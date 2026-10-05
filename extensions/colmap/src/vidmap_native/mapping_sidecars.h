#pragma once

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <map>

#include "vidmap_native/types.h"

namespace vidmap {
struct MappingSidecars {
  std::map<ImageId, ImageData> images;
  std::map<PairId, PairData> pairs;
  std::map<Point3DId, TrackData> tracks;
  void Validate(const colmap::Reconstruction& reconstruction) const;
};
}  // namespace vidmap
