#include <stdexcept>
#include <string>

#include "vidmap_native/bundle_adjustment.h"
#include "vidmap_native/global_positioning.h"
#include "vidmap_native/tracks.h"
#include "vidmap_native/types.h"
#include "vidmap_native/video_rotation_averaging.h"
#include "vidmap_native/view_graph.h"

namespace {

void Check(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Callable>
void CheckInvalidArgument(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

void TestStableIdentifiers() {
  Check(vidmap::CanonicalPairId(1, 2) == vidmap::CanonicalPairId(2, 1),
        "canonical pair IDs must be orientation independent");
  Check(vidmap::EncodeObservationKey(7, 11) ==
            (static_cast<vidmap::Point3DId>(7) << 32 | 11),
        "track observation encoding changed");
}

void TestSolverDefaults() {
  const vidmap::GlobalPositionerOptions options;
  Check(
      options.parameter_ordering == vidmap::GlobalPositioningOrdering::kGrouped,
      "global positioning ordering default changed");
  Check(options.center_mode == vidmap::GlobalPositioningCenterMode::kFrame,
        "global positioning center default changed");
  options.Validate();
}

void TestOptionValidation() {
  vidmap::TrackEstablishmentOptions track_options;
  track_options.required_tracks_per_view = -1;
  CheckInvalidArgument([&] { track_options.Validate(); },
                       "negative track quota was accepted");

  vidmap::InlierThresholdOptions inlier_options;
  inlier_options.min_angle_from_epipole_deg = 181.0;
  CheckInvalidArgument([&] { inlier_options.Validate(); },
                       "invalid epipole angle was accepted");

  vidmap::VideoRotationAveragingOptions rotation_options;
  rotation_options.num_threads = 0;
  CheckInvalidArgument([&] { rotation_options.Validate(); },
                       "zero rotation-averaging threads were accepted");

  vidmap::BundleAdjustmentOptions bundle_options;
  bundle_options.max_num_iterations = 0;
  CheckInvalidArgument([&] { bundle_options.Validate(); },
                       "zero bundle-adjustment iterations were accepted");
}

}  // namespace

int main() {
  TestStableIdentifiers();
  TestSolverDefaults();
  TestOptionValidation();
  return 0;
}
