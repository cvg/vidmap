#pragma once

#include <map>
#include <vector>

#include "vidmap_native/types.h"

namespace vidmap {

class MappingProblem {
 public:
  void AddCamera(const CameraRecord& camera);
  void AddImage(const ImageRecord& image);
  void AddPair(const PairRecord& pair);
  void AddTrack(const TrackRecord& track);

  void UpdateCamera(const CameraRecord& camera);
  void UpdateImage(const ImageRecord& image);
  void UpdatePair(const PairRecord& pair);
  void UpdateTrack(const TrackRecord& track);

  void ClearTracks();

  const CameraRecord& Camera(CameraId camera_id) const;
  const ImageRecord& Image(ImageId image_id) const;
  const PairRecord& Pair(PairId pair_id) const;
  const TrackRecord& Track(Point3DId point3D_id) const;

  std::vector<CameraId> CameraIds() const;
  std::vector<ImageId> ImageIds() const;
  std::vector<PairId> PairIds() const;
  std::vector<Point3DId> Point3DIds() const;

  std::size_t NumCameras() const;
  std::size_t NumImages() const;
  std::size_t NumPairs() const;
  std::size_t NumTracks() const;

  void Validate() const;

 private:
  std::map<CameraId, CameraRecord> cameras_;
  std::map<ImageId, ImageRecord> images_;
  std::map<PairId, PairRecord> pairs_;
  std::map<Point3DId, TrackRecord> tracks_;
};

}  // namespace vidmap
