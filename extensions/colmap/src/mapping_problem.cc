#include "vidmap_native/mapping_problem.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace vidmap {
namespace {

template <typename Key, typename Value>
void AddUnique(const Key key,
               const Value& value,
               const std::string& label,
               std::map<Key, Value>* values) {
  const auto [unused, inserted] = values->emplace(key, value);
  if (!inserted) {
    throw std::invalid_argument(label +
                                " already exists: " + std::to_string(key));
  }
}

template <typename Key, typename Value>
void UpdateExisting(const Key key,
                    const Value& value,
                    const std::string& label,
                    std::map<Key, Value>* values) {
  auto it = values->find(key);
  if (it == values->end()) {
    throw std::out_of_range(label + " does not exist: " + std::to_string(key));
  }
  it->second = value;
}

template <typename Key, typename Value>
const Value& GetExisting(const Key key,
                         const std::string& label,
                         const std::map<Key, Value>& values) {
  auto it = values.find(key);
  if (it == values.end()) {
    throw std::out_of_range(label + " does not exist: " + std::to_string(key));
  }
  return it->second;
}

template <typename Key, typename Value>
std::vector<Key> Keys(const std::map<Key, Value>& values) {
  std::vector<Key> keys;
  keys.reserve(values.size());
  for (const auto& [key, unused] : values) {
    keys.push_back(key);
  }
  return keys;
}

}  // namespace

void MappingProblem::AddCamera(const CameraRecord& camera) {
  camera.Validate();
  AddUnique(camera.camera_id, camera, "camera", &cameras_);
}

void MappingProblem::AddImage(const ImageRecord& image) {
  image.Validate();
  AddUnique(image.image_id, image, "image", &images_);
}

void MappingProblem::AddPair(const PairRecord& pair) {
  pair.Validate();
  AddUnique(pair.pair_id, pair, "pair", &pairs_);
}

void MappingProblem::AddTrack(const TrackRecord& track) {
  track.Validate();
  AddUnique(track.point3D_id, track, "track", &tracks_);
}

void MappingProblem::UpdateCamera(const CameraRecord& camera) {
  camera.Validate();
  UpdateExisting(camera.camera_id, camera, "camera", &cameras_);
}

void MappingProblem::UpdateImage(const ImageRecord& image) {
  image.Validate();
  UpdateExisting(image.image_id, image, "image", &images_);
}

void MappingProblem::UpdatePair(const PairRecord& pair) {
  pair.Validate();
  UpdateExisting(pair.pair_id, pair, "pair", &pairs_);
}

void MappingProblem::UpdateTrack(const TrackRecord& track) {
  track.Validate();
  UpdateExisting(track.point3D_id, track, "track", &tracks_);
}

void MappingProblem::ClearTracks() { tracks_.clear(); }

const CameraRecord& MappingProblem::Camera(CameraId camera_id) const {
  return GetExisting(camera_id, "camera", cameras_);
}

const ImageRecord& MappingProblem::Image(ImageId image_id) const {
  return GetExisting(image_id, "image", images_);
}

const PairRecord& MappingProblem::Pair(PairId pair_id) const {
  return GetExisting(pair_id, "pair", pairs_);
}

const TrackRecord& MappingProblem::Track(Point3DId point3D_id) const {
  return GetExisting(point3D_id, "track", tracks_);
}

std::vector<CameraId> MappingProblem::CameraIds() const {
  return Keys(cameras_);
}

std::vector<ImageId> MappingProblem::ImageIds() const { return Keys(images_); }

std::vector<PairId> MappingProblem::PairIds() const { return Keys(pairs_); }

std::vector<Point3DId> MappingProblem::Point3DIds() const {
  return Keys(tracks_);
}

std::size_t MappingProblem::NumCameras() const { return cameras_.size(); }

std::size_t MappingProblem::NumImages() const { return images_.size(); }

std::size_t MappingProblem::NumPairs() const { return pairs_.size(); }

std::size_t MappingProblem::NumTracks() const { return tracks_.size(); }

void MappingProblem::Validate() const {
  for (const auto& [camera_id, camera] : cameras_) {
    camera.Validate();
    if (camera_id != camera.camera_id) {
      throw std::logic_error("camera key does not match camera_id");
    }
  }
  for (const auto& [image_id, image] : images_) {
    image.Validate();
    if (image_id != image.image_id) {
      throw std::logic_error("image key does not match image_id");
    }
    if (cameras_.find(image.camera_id) == cameras_.end()) {
      throw std::invalid_argument("image references a missing camera");
    }
  }
  for (const auto& [pair_id, pair] : pairs_) {
    pair.Validate();
    if (pair_id != pair.pair_id) {
      throw std::logic_error("pair key does not match pair_id");
    }
    const auto image1 = images_.find(pair.image_id1);
    const auto image2 = images_.find(pair.image_id2);
    if (image1 == images_.end() || image2 == images_.end()) {
      throw std::invalid_argument("pair references a missing image");
    }
    for (Eigen::Index row = 0; row < pair.all_matches.rows(); ++row) {
      if (pair.all_matches(row, 0) >= image1->second.NumFeatures() ||
          pair.all_matches(row, 1) >= image2->second.NumFeatures()) {
        throw std::invalid_argument("pair match references a missing feature");
      }
    }
  }
  for (const auto& [point3D_id, track] : tracks_) {
    track.Validate();
    if (point3D_id != track.point3D_id) {
      throw std::logic_error("track key does not match point3D_id");
    }
    const auto validate_observations = [this](const MatrixX2u& observations) {
      for (Eigen::Index row = 0; row < observations.rows(); ++row) {
        const auto image = images_.find(observations(row, 0));
        if (image == images_.end() ||
            observations(row, 1) >= image->second.NumFeatures()) {
          throw std::invalid_argument(
              "track observation references a missing feature");
        }
      }
    };
    validate_observations(track.observations);
    validate_observations(track.loop_closure_observations);
    validate_observations(track.loop_closure_anchors);
  }
}

}  // namespace vidmap
