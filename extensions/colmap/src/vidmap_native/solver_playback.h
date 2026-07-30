#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "vidmap_native/types.h"

namespace vidmap {

struct SolverPlaybackCapture {
  std::string phase;
  int iteration = -1;
  std::vector<ImageId> image_ids;
  MatrixX3d centers;
  std::vector<Point3DId> point3D_ids;
  MatrixX3d points_xyz;
  MatrixX2u loop_closure_pairs;
  std::vector<std::uint64_t> loop_closure_support_counts;
  VectorXd loop_closure_raw_scores;
};

struct SolverPlaybackOptions {
  int snapshot_every_n_iterations = 1;
  std::vector<ImageId> image_ids;
  std::vector<Point3DId> point3D_ids;
  std::function<void(const SolverPlaybackCapture&)> callback;

  bool IsEnabled() const { return static_cast<bool>(callback); }
  void Validate() const;
};

}  // namespace vidmap
