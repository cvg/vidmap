#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace vidmap {

using CameraId = std::uint32_t;
using ImageId = std::uint32_t;
using FrameId = std::uint32_t;
using PairId = std::uint64_t;
using Point3DId = std::uint64_t;

using MatrixX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatrixX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatrixX2u =
    Eigen::Matrix<std::uint32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;
using VectorXb = Eigen::Matrix<std::uint8_t, Eigen::Dynamic, 1>;

struct CameraRecord {
  CameraId camera_id = 0;
  int model_id = -1;
  std::uint64_t width = 0;
  std::uint64_t height = 0;
  VectorXd params;
  bool has_prior_focal_length = false;

  void Validate() const;
};

struct PoseRecord {
  bool has_pose = false;
  Eigen::Vector4d rotation_xyzw = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  void Validate() const;
};

struct ImageRecord {
  ImageId image_id = 0;
  CameraId camera_id = 0;
  FrameId frame_id = 0;
  std::string name;
  PoseRecord pose;
  MatrixX2d keypoints;
  MatrixX3d bearings;
  VectorXd depth_values;
  VectorXd depth_stddevs;
  VectorXb depth_validity;
  MatrixX2d angular_stddevs;
  VectorXb is_inlier;
  VectorXb is_track_anchor;
  VectorXb is_depth_outlier;

  void Validate() const;
  std::size_t NumFeatures() const;
};

struct TwoViewGeometryRecord {
  int configuration = 0;
  bool has_essential = false;
  bool has_fundamental = false;
  bool has_homography = false;
  Eigen::Matrix3d essential = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d fundamental = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d homography = Eigen::Matrix3d::Zero();
  PoseRecord cam2_from_cam1;

  void Validate() const;
};

struct PairRecord {
  PairId pair_id = 0;
  ImageId image_id1 = 0;
  ImageId image_id2 = 0;
  bool is_valid = true;
  TwoViewGeometryRecord geometry;
  MatrixX2u all_matches;
  VectorXi inlier_indices;
  VectorXb are_loop_closure;

  void Validate() const;
};

struct TrackRecord {
  Point3DId point3D_id = 0;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Matrix<std::uint8_t, 3, 1> color =
      Eigen::Matrix<std::uint8_t, 3, 1>::Zero();
  double error = -1.0;
  MatrixX2u observations;
  MatrixX2u loop_closure_observations;
  MatrixX2u loop_closure_anchors;

  void Validate() const;
};

PairId CanonicalPairId(ImageId image_id1, ImageId image_id2);

}  // namespace vidmap
