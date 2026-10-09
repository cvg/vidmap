#pragma once

#include "colmap/estimators/view_graph_calibration.h"

#include <cstddef>
#include <vector>

#include "vidmap_native/focal_prior.h"

namespace vidmap {

struct InlierThresholdOptions {
  double max_epipolar_error_essential = 1.0;
  double max_epipolar_error_fundamental = 4.0;
  double max_epipolar_error_homography = 4.0;
  double min_angle_from_epipole_deg = 3.0;

  void Validate() const;
};

void PrepareImageBearings(const colmap::Reconstruction&, MappingSidecars&);
void ReclassifyCalibratedPlanarPairs(const colmap::Reconstruction&,
                                     const colmap::PoseGraph&,
                                     MappingSidecars&);
void ImagePairsInlierCount(const InlierThresholdOptions&,
                           const colmap::Reconstruction&,
                           const colmap::PoseGraph&,
                           MappingSidecars&);
std::size_t FilterPairsByInlierNum(int, colmap::PoseGraph&, MappingSidecars&);
std::size_t FilterPairsByInlierRatio(double,
                                     colmap::PoseGraph&,
                                     MappingSidecars&);
std::size_t CalibrateFocalLengths(const colmap::ViewGraphCalibrationOptions&,
                                  colmap::Reconstruction&,
                                  colmap::PoseGraph&,
                                  const MappingSidecars&,
                                  const std::vector<LogFocalPriorRecord>&,
                                  const std::vector<LogRelativeFocalPriorRecord>&);
}  // namespace vidmap
