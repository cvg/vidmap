#pragma once

#include "colmap/scene/two_view_geometry.h"

#include <cstddef>
#include <cstdint>

namespace vidmap {

using CameraId = std::uint32_t;
using ImageId = std::uint32_t;
using FrameId = std::uint32_t;
using PairId = std::uint64_t;
using Point3DId = std::uint64_t;

using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixX2u =
    Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using VectorXb = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1>;

struct ImageData {
  MatrixX3d bearings;
  VectorXd depth_values;
  VectorXd depth_stddevs;
  VectorXb depth_validity;
  VectorXb is_depth_outlier;

  void Validate(std::size_t num_features) const;
};

struct PairData {
  colmap::TwoViewGeometry geometry;
  bool has_relative_pose = false;
  MatrixX2u all_matches;
  VectorXi inlier_indices;
  VectorXb are_loop_closure;

  void Validate() const;
};

struct TrackData {
  MatrixX2u loop_closure_observations;
  MatrixX2u loop_closure_anchors;

  void Validate() const;
};

}  // namespace vidmap
