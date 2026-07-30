#include "vidmap_native/solver_playback.h"

#include <stdexcept>
#include <unordered_set>

namespace vidmap {
namespace {

template <typename Id>
void ValidateUniqueIds(const std::vector<Id>& ids, const char* name) {
  const std::unordered_set<Id> unique(ids.begin(), ids.end());
  if (unique.size() != ids.size()) {
    throw std::invalid_argument(std::string("duplicate ") + name +
                                " in playback selection");
  }
}

}  // namespace

void SolverPlaybackOptions::Validate() const {
  if (snapshot_every_n_iterations <= 0) {
    throw std::invalid_argument("playback snapshot interval must be positive");
  }
  ValidateUniqueIds(image_ids, "image ID");
  ValidateUniqueIds(point3D_ids, "point3D ID");
}

}  // namespace vidmap
