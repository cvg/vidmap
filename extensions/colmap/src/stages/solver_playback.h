#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "vidmap_native/solver_playback.h"
#include <ceres/ceres.h>

namespace vidmap {

class SolverPlaybackIterationCallback final : public ceres::IterationCallback {
 public:
  SolverPlaybackIterationCallback(const int interval,
                                  const int iteration_offset,
                                  std::function<void(int)> capture)
      : interval_(interval),
        iteration_offset_(iteration_offset),
        capture_(std::move(capture)) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    const int iteration = summary.iteration + iteration_offset_;
    if (iteration % interval_ == 0) {
      capture_(iteration);
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  int interval_;
  int iteration_offset_;
  std::function<void(int)> capture_;
};

inline std::uint64_t PlaybackSelectionKey(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline std::vector<Point3DId> SelectPlaybackPoints(
    std::vector<Point3DId> point3D_ids, const std::size_t limit = 200000) {
  if (point3D_ids.size() <= limit) return point3D_ids;
  std::sort(point3D_ids.begin(),
            point3D_ids.end(),
            [](const Point3DId lhs, const Point3DId rhs) {
              return std::pair(PlaybackSelectionKey(lhs), lhs) <
                     std::pair(PlaybackSelectionKey(rhs), rhs);
            });
  point3D_ids.resize(limit);
  std::sort(point3D_ids.begin(), point3D_ids.end());
  return point3D_ids;
}

template <typename Capture>
void SolveWithPlayback(const SolverPlaybackOptions& playback,
                       ceres::Solver::Options solver_options,
                       ceres::Problem* problem,
                       Capture&& capture,
                       ceres::Solver::Summary* summary,
                       const int iteration_offset = 0) {
  if (!playback.IsEnabled()) {
    ceres::Solve(solver_options, problem, summary);
    return;
  }

  if (iteration_offset == 0) {
    capture("initial", -1);
  }
  SolverPlaybackIterationCallback callback(
      playback.snapshot_every_n_iterations,
      iteration_offset,
      [&capture](const int iteration) { capture("iteration", iteration); });
  solver_options.update_state_every_iteration = true;
  solver_options.callbacks.push_back(&callback);
  ceres::Solve(solver_options, problem, summary);
  if (summary->IsSolutionUsable()) {
    const int iteration =
        summary->iterations.empty()
            ? iteration_offset - 1
            : summary->iterations.back().iteration + iteration_offset;
    capture("final", iteration);
  }
}

}  // namespace vidmap
