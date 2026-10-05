#include "colmap/scene/two_view_geometry.h"

#include <stdexcept>
#include <string>

#include "vidmap_native/view_graph.h"

namespace vidmap {
void PrepareImageBearings(const colmap::Reconstruction& reconstruction,
                          MappingSidecars& sidecars) {
  for (const auto& [image_id, image] : reconstruction.Images()) {
    auto& data = sidecars.images.at(image_id);
    const auto& camera = *image.CameraPtr();
    data.bearings.resize(image.NumPoints2D(), 3);
    for (std::size_t row = 0; row < image.NumPoints2D(); ++row) {
      const auto point = camera.CamFromImg(image.Point2D(row).xy);
      if (!point)
        throw std::runtime_error("CamFromImg failed for image " +
                                 std::to_string(image_id));
      data.bearings.row(row) = point->homogeneous().normalized();
    }
  }
}

void ReclassifyCalibratedPlanarPairs(
    const colmap::Reconstruction& reconstruction,
    const colmap::PoseGraph& graph,
    MappingSidecars& sidecars) {
  sidecars.Validate(reconstruction);
  for (auto& [pair_id, pair] : sidecars.pairs) {
    const auto [image_id1, image_id2] = colmap::PairIdToImagePair(pair_id);
    if (!graph.IsValid(pair_id)) continue;
    const auto& image1 = reconstruction.Image(image_id1);
    const auto& image2 = reconstruction.Image(image_id2);
    const colmap::Camera& camera1 = *image1.CameraPtr();
    const colmap::Camera& camera2 = *image2.CameraPtr();
    if (!camera1.has_prior_focal_length || !camera2.has_prior_focal_length) {
      continue;
    }

    if (pair.geometry.config == colmap::TwoViewGeometry::PLANAR) {
      pair.geometry.config = colmap::TwoViewGeometry::CALIBRATED;
    }
  }
}

std::size_t FilterPairsByInlierNum(int min_inlier_count,
                                   colmap::PoseGraph& graph,
                                   MappingSidecars& sidecars) {
  if (min_inlier_count < 0) {
    throw std::invalid_argument("min_inlier_count must be non-negative");
  }
  std::size_t filtered = 0;
  for (auto& [pair_id, pair] : sidecars.pairs) {
    if (graph.IsValid(pair_id) &&
        pair.inlier_indices.size() < min_inlier_count) {
      graph.SetInvalidEdge(pair_id);
      ++filtered;
    }
  }
  return filtered;
}

std::size_t FilterPairsByInlierRatio(double min_inlier_ratio,
                                     colmap::PoseGraph& graph,
                                     MappingSidecars& sidecars) {
  if (min_inlier_ratio < 0.0 || min_inlier_ratio > 1.0) {
    throw std::invalid_argument("min_inlier_ratio must be in [0, 1]");
  }
  std::size_t filtered = 0;
  for (auto& [pair_id, pair] : sidecars.pairs) {
    if (!graph.IsValid(pair_id) || pair.all_matches.rows() == 0) continue;
    const double ratio = static_cast<double>(pair.inlier_indices.size()) /
                         pair.all_matches.rows();
    if (ratio < min_inlier_ratio) {
      graph.SetInvalidEdge(pair_id);
      ++filtered;
    }
  }
  return filtered;
}

}  // namespace vidmap
