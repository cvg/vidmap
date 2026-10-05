#include "vidmap_native/mapping_sidecars.h"

#include <stdexcept>
namespace vidmap {
void MappingSidecars::Validate(
    const colmap::Reconstruction& reconstruction) const {
  for (const auto& [id, data] : images)
    data.Validate(reconstruction.Image(id).NumPoints2D());
  for (const auto& [id, data] : pairs) {
    data.Validate();
    const auto [id1, id2] = colmap::PairIdToImagePair(id);
    const auto num_points1 = reconstruction.Image(id1).NumPoints2D();
    const auto num_points2 = reconstruction.Image(id2).NumPoints2D();
    for (Eigen::Index row = 0; row < data.all_matches.rows(); ++row) {
      if (data.all_matches(row, 0) >= num_points1 ||
          data.all_matches(row, 1) >= num_points2)
        throw std::invalid_argument("pair match references an unknown feature");
    }
  }
  for (const auto& [id, data] : tracks) data.Validate();
}
}  // namespace vidmap
