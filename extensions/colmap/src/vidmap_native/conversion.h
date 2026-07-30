#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/two_view_geometry.h"

#include <vector>

#include "vidmap_native/types.h"

namespace vidmap {

colmap::Camera ToColmapCamera(const CameraRecord& record);
colmap::Rigid3d ToColmapPose(const PoseRecord& record);
PoseRecord FromColmapPose(const colmap::Rigid3d& pose);
std::vector<Eigen::Vector2d> ToColmapPoints(const MatrixX2d& points);
colmap::TwoViewGeometry ToColmapGeometry(const PairRecord& pair);
void UpdateGeometryRecord(const colmap::TwoViewGeometry& geometry,
                          TwoViewGeometryRecord* record);

}  // namespace vidmap
