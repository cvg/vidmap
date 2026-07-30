#pragma once

#include <cstddef>
#include <vector>

#include "vidmap_native/mapping_problem.h"

namespace vidmap {

struct VideoRotationAveragingOptions {
  int random_seed = 1;
  int image_order_passes = 1;
  bool filter_unregistered = true;
  bool skip_risky_lc_pairs = false;
  double max_rotation_error_deg = 0.0;
  double video_tracking_huber_scale = 0.1;
  double video_lc_cauchy_scale = 0.05;
  int num_threads = 1;
  int max_num_iterations = 100;

  void Validate() const;
};

struct RotationAveragingResult {
  bool success = false;
  std::vector<ImageId> registered_image_ids;
};

RotationAveragingResult RunVideoRotationAveraging(
    const VideoRotationAveragingOptions& options,
    const std::vector<ImageId>& image_map_order,
    const std::vector<PairId>& pair_map_order,
    MappingProblem* problem);

}  // namespace vidmap
