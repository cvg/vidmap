#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/reconstruction.h"

#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "stages/metric_depth.h"
#include "stages/weighted_motion_averaging.h"
#include "vidmap_native/mapping_sidecars.h"
#include <Eigen/LU>
#include <ceres/loss_function.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

class DeadZoneHuberLoss final : public ceres::LossFunction {
 public:
  DeadZoneHuberLoss(double dead_zone, double huber_width)
      : dead_zone_(dead_zone), huber_width_(huber_width) {}

  void Evaluate(double squared_norm, double rho[3]) const override {
    const double residual_norm = std::sqrt(std::max(0.0, squared_norm));
    if (residual_norm <= dead_zone_) {
      rho[0] = 0.0;
      rho[1] = 0.0;
      rho[2] = 0.0;
      return;
    }
    const double shifted_norm = residual_norm - dead_zone_;
    if (shifted_norm <= huber_width_) {
      rho[0] = shifted_norm * shifted_norm;
      rho[1] = shifted_norm / residual_norm;
      rho[2] =
          dead_zone_ / (2.0 * residual_norm * residual_norm * residual_norm);
      return;
    }
    rho[0] = 2.0 * huber_width_ * shifted_norm - huber_width_ * huber_width_;
    rho[1] = huber_width_ / residual_norm;
    rho[2] =
        -huber_width_ / (2.0 * residual_norm * residual_norm * residual_norm);
  }

 private:
  const double dead_zone_;
  const double huber_width_;
};

// Observation identities and borrowed parameters from the current Ceres
// problem.
struct Observation {
  colmap::point3D_t point_id;
  colmap::TrackElement element;
  colmap::TrackElement anchor;
  bool loop;
  bool support;
  double* center = nullptr;
  double* point = nullptr;
  double* scale = nullptr;
  ceres::CostFunction* cost = nullptr;

  colmap::image_t ImageId() const { return element.image_id; }
  colmap::point2D_t Index() const { return element.point2D_idx; }
  std::string Key() const {
    return std::to_string(point_id) + ":" + std::to_string(ImageId()) + ":" +
           std::to_string(Index()) + ":" + std::to_string(loop);
  }
};

struct ObservationSelection {
  std::vector<Observation> observations;
};
using TrackMap = std::map<colmap::point3D_t, colmap::Track>;
using Centers = std::unordered_map<colmap::image_t, double*>;

Centers CenterPointers(const py::dict& arrays) {
  Centers centers;
  for (const auto& item : arrays) {
    auto array = item.second.cast<py::array_t<double>>();
    centers.emplace(item.first.cast<colmap::image_t>(), array.mutable_data());
  }
  return centers;
}

bool MaskValue(const vidmap::VectorXb& mask, colmap::point2D_t index) {
  return index < mask.size() && mask(index);
}

using ObservationKind = std::tuple<bool, bool, bool>;
std::map<ObservationKind, ObservationSelection> PrepareObservations(
    const colmap::Reconstruction& reconstruction,
    const vidmap::MappingSidecars& sidecars,
    size_t minimum_track_length,
    bool include_loop,
    const std::vector<colmap::image_t>& image_order,
    size_t support_count) {
  std::map<ObservationKind, ObservationSelection> selected;
  std::unordered_map<colmap::image_t, size_t> timeline;
  for (size_t i = 0; i < image_order.size(); ++i)
    timeline.emplace(image_order[i], i);
  for (const auto& [point_id, point] : reconstruction.Points3D()) {
    const auto& regular = point.track.Elements();
    if (regular.size() < minimum_track_length) continue;
    std::vector<bool> support(regular.size(), false);
    if (support_count) {
      std::vector<size_t> order(regular.size());
      std::iota(order.begin(), order.end(), 0);
      const size_t count = std::min(order.size(), support_count);
      std::partial_sort(
          order.begin(),
          order.begin() + count,
          order.end(),
          [&](auto a, auto b) {
            return std::make_pair(timeline.at(regular[a].image_id),
                                  regular[a].point2D_idx) <
                   std::make_pair(timeline.at(regular[b].image_id),
                                  regular[b].point2D_idx);
          });
      for (size_t i = 0; i < count; ++i) support[order[i]] = true;
    }
    auto append = [&](const colmap::TrackElement& element,
                      const colmap::TrackElement& anchor,
                      bool loop,
                      bool early) {
      const auto& image = reconstruction.Image(element.image_id);
      selected[{loop, image.CameraPtr()->has_prior_focal_length, early}]
          .observations.push_back({point_id, element, anchor, loop, early});
    };
    std::unordered_set<colmap::image_t> regular_images;
    for (size_t row = 0; row < regular.size(); ++row) {
      if (!regular_images.insert(regular[row].image_id).second)
        throw std::invalid_argument(
            "A regular track must have at most one observation per image");
      append(regular[row], {}, false, support[row]);
    }
    const auto extra = sidecars.tracks.find(point_id);
    if (!include_loop || extra == sidecars.tracks.end()) continue;
    const auto& elements = extra->second.loop_closure_observations;
    const auto& anchors = extra->second.loop_closure_anchors;
    for (Eigen::Index row = 0; row < elements.rows(); ++row) {
      colmap::TrackElement anchor;
      if (anchors.rows() == elements.rows())
        anchor = {anchors(row, 0), anchors(row, 1)};
      append({elements(row, 0), elements(row, 1)}, anchor, true, false);
    }
  }
  return selected;
}

ObservationSelection CombineObservations(
    const std::vector<const ObservationSelection*>& selections) {
  ObservationSelection combined;
  size_t count = 0;
  for (const auto* selection : selections)
    count += selection->observations.size();
  combined.observations.reserve(count);
  for (const auto* selection : selections) {
    combined.observations.insert(combined.observations.end(),
                                 selection->observations.begin(),
                                 selection->observations.end());
  }
  return combined;
}

}  // namespace

PYBIND11_MAKE_OPAQUE(TrackMap);

PYBIND11_MODULE(global_positioning, m) {
  py::module_::import("pyceres");
  py::classh<ceres::LossFunctionWrapper, ceres::LossFunction>(m, "MutableLoss")
      .def(py::init([](ceres::LossFunction* loss) {
             return std::make_unique<ceres::LossFunctionWrapper>(
                 loss, ceres::DO_NOT_TAKE_OWNERSHIP);
           }),
           py::keep_alive<1, 2>())
      .def(
          "reset",
          [](ceres::LossFunctionWrapper& self, ceres::LossFunction* loss) {
            self.Reset(loss, ceres::DO_NOT_TAKE_OWNERSHIP);
          },
          py::keep_alive<1, 2>());
  py::class_<ObservationSelection>(m, "ObservationSelection")
      .def("__len__",
           [](const ObservationSelection& self) {
             return self.observations.size();
           })
      .def("scale_values", [](const ObservationSelection& self) {
        py::dict values;
        for (const auto& obs : self.observations)
          values[py::str(obs.Key())] = *obs.scale;
        return values;
      });
  py::class_<TrackMap>(m, "Point3DTrackMap");
  m.def("prepare_observations", &PrepareObservations);
  m.def("combine_observations", &CombineObservations);
  m.def("selected_tracks",
        [](const colmap::Reconstruction& reconstruction,
           const ObservationSelection& selected) {
          TrackMap tracks;
          for (const auto& [id, point] : reconstruction.Points3D())
            tracks.emplace(id, colmap::Track());
          for (const auto& obs : selected.observations)
            tracks.at(obs.point_id).AddElement(obs.ImageId(), obs.Index());
          return tracks;
        });
  m.def("swap_tracks",
        [](colmap::Reconstruction& reconstruction, TrackMap& tracks) {
          for (auto& [id, track] : tracks)
            std::swap(reconstruction.Point3D(id).track, track);
        });
  m.def("collect_default_observations",
        [](ceres::Problem& problem,
           colmap::Reconstruction& reconstruction,
           const py::dict& centers,
           ObservationSelection& selected) {
          std::unordered_map<const double*, colmap::point3D_t> points;
          for (const auto& [id, point] : reconstruction.Points3D())
            points.emplace(point.xyz.data(), id);
          std::unordered_map<const double*, colmap::image_t> images;
          for (const auto& [id, pointer] : CenterPointers(centers))
            images.emplace(pointer, id);
          std::map<std::pair<colmap::point3D_t, colmap::image_t>, Observation*>
              observations;
          for (auto& obs : selected.observations)
            observations.emplace(std::make_pair(obs.point_id, obs.ImageId()),
                                 &obs);
          std::vector<ceres::ResidualBlockId> residuals;
          problem.GetResidualBlocks(&residuals);
          for (auto residual : residuals) {
            std::vector<double*> blocks;
            problem.GetParameterBlocksForResidualBlock(residual, &blocks);
            auto& obs =
                *observations.at({points.at(blocks[1]), images.at(blocks[0])});
            obs.center = blocks[0];
            obs.point = blocks[1];
            obs.scale = blocks[2];
          }
          auto& obs = selected.observations;
          obs.erase(std::remove_if(obs.begin(),
                                   obs.end(),
                                   [](const auto& o) { return !o.scale; }),
                    obs.end());
        });
  m.def("initialize_points",
        [](ceres::Problem& problem,
           colmap::Reconstruction& reconstruction,
           size_t minimum_track_length,
           py::object random) {
          std::vector<colmap::point3D_t> extra;
          for (const auto& [id, point] : reconstruction.Points3D()) {
            if (point.track.Length() >= minimum_track_length &&
                !problem.HasParameterBlock(point.xyz.data()))
              extra.push_back(id);
          }
          std::sort(extra.begin(), extra.end());
          const auto values =
              random
                  .attr("uniform")(
                      -100.0, 100.0, py::make_tuple(extra.size(), 3))
                  .cast<py::array_t<double>>();
          for (size_t i = 0; i < extra.size(); ++i)
            reconstruction.Point3D(extra[i]).xyz =
                Eigen::Map<const Eigen::Vector3d>(values.data(i, 0));
        });
  m.def("track_observation_counts",
        [](const colmap::Reconstruction& reconstruction,
           const vidmap::MappingSidecars& sidecars) {
          std::map<colmap::image_t, size_t> counts;
          for (const auto& [id, point] : reconstruction.Points3D()) {
            for (const auto& element : point.track.Elements())
              ++counts[element.image_id];
            const auto extra = sidecars.tracks.find(id);
            if (extra != sidecars.tracks.end())
              for (Eigen::Index row = 0;
                   row < extra->second.loop_closure_observations.rows();
                   ++row)
                ++counts[extra->second.loop_closure_observations(row, 0)];
          }
          return counts;
        });
  m.def("observation_counts",
        [](const std::vector<ObservationSelection*>& selections) {
          size_t regular = 0, loop = 0;
          bool support = false;
          std::unordered_set<vidmap::Point3DId> points;
          for (const auto* selection : selections)
            for (const auto& obs : selection->observations) {
              (obs.loop ? loop : regular)++;
              support |= obs.support;
              points.insert(obs.point_id);
            }
          return py::make_tuple(regular, loop, points.size(), support);
        });
  m.def("fix_first_observation_scale",
        [](ceres::Problem& problem,
           const std::vector<ObservationSelection*>& selections) {
          for (const auto* selection : selections) {
            if (selection->observations.empty()) continue;
            problem.SetParameterBlockConstant(
                selection->observations.front().scale);
            break;
          }
        });
  m.def("dead_zone_huber_loss", [](double dead_zone, double width) {
    return std::shared_ptr<ceres::LossFunction>(
        new DeadZoneHuberLoss(dead_zone, width));
  });
  m.def(
      "scaled_loss",
      [](std::shared_ptr<ceres::LossFunction> loss, double weight) {
        return std::shared_ptr<ceres::LossFunction>(new ceres::ScaledLoss(
            loss.get(), weight, ceres::DO_NOT_TAKE_OWNERSHIP));
      },
      py::keep_alive<0, 1>());
  m.def(
      "append_observations",
      [](colmap::GlobalPositioner& positioner,
         colmap::Reconstruction& reconstruction,
         ObservationSelection& selection,
         const py::dict& centers,
         const std::shared_ptr<ceres::LossFunction>& loss,
         const std::optional<double> stddev,
         bool initialize_from_geometry,
         bool optimize_scales) {
        auto& problem = positioner.Problem();
        const auto& ordering =
            positioner.SolverOptions().linear_solver_ordering;
        auto& observations = selection.observations;
        py::array_t<double> scales(observations.size());
        const auto center_pointers = CenterPointers(centers);
        size_t count = 0;
        for (auto& observation : observations) {
          const auto point_id = observation.point_id;
          const auto image_id = observation.ImageId();
          const auto index = observation.Index();
          const auto& image = reconstruction.Image(image_id);
          if (!image.HasPose()) continue;
          const Eigen::Matrix3d rotation =
              image.CamFromWorld().rotation().toRotationMatrix();
          std::optional<Eigen::Matrix3d> covariance;
          std::optional<Eigen::Vector3d> ray;
          const auto& camera = *image.CameraPtr();
          const auto& pixel = image.Point2D(index).xy;
          if (stddev) {
            const auto value = camera.CamRayFromImgWithJac(pixel);
            if (!value) continue;
            const Eigen::Matrix2d gram =
                value->jacobian.transpose() * value->jacobian;
            const double radial_variance = 2.0 / gram.inverse().trace();
            if (!(radial_variance > 0.0) || !std::isfinite(radial_variance))
              continue;
            covariance =
                (*stddev * *stddev) * rotation.transpose() *
                (value->jacobian * value->jacobian.transpose() +
                 radial_variance * value->ray * value->ray.transpose()) *
                rotation;
            ray = value->ray;
          } else {
            ray = camera.CamRayFromImg(pixel);
          }
          if (!ray) continue;
          double* center_data = center_pointers.at(image_id);
          auto& point = reconstruction.Point3D(point_id).xyz;
          const Eigen::Vector3d direction = rotation.transpose() * *ray;
          const Eigen::Vector3d delta =
              point - Eigen::Map<const Eigen::Vector3d>(center_data);
          double* scale = scales.mutable_data(count);
          *scale =
              initialize_from_geometry
                  ? (delta.squaredNorm() > 0.0
                         ? std::max(1e-5,
                                    direction.dot(delta) / delta.squaredNorm())
                         : 1e-5)
                  : 1.0;
          auto* cost =
              covariance
                  ? colmap::CovarianceWeightedCostFunctor<
                        colmap::BATAPairwiseDirectionCostFunctor>::
                        Create(*covariance, direction)
                  : colmap::BATAPairwiseDirectionCostFunctor::Create(direction);
          problem.AddResidualBlock(
              cost, loss.get(), center_data, point.data(), scale);
          if (!ordering->IsMember(point.data()))
            ordering->AddElementToGroup(point.data(), 1);
          problem.SetParameterLowerBound(scale, 0, 1e-5);
          if (!optimize_scales) problem.SetParameterBlockConstant(scale);
          observation.center = center_data;
          observation.point = point.data();
          observation.scale = scale;
          observation.cost = cost;
          observations[count++] = observation;
        }
        observations.resize(count);
        return scales[py::slice(0, count, 1)].cast<py::array>();
      },
      py::arg("positioner"),
      py::arg("reconstruction"),
      py::arg("observations"),
      py::arg("centers"),
      py::arg("loss"),
      py::arg("stddev"),
      py::arg("initialize_from_geometry"),
      py::arg("optimize_scales"),
      "Append selected observations. "
      "Keep the returned scales, reconstruction, centers and loss alive "
      "while using the problem.");
  m.def("split_depth_observations",
        [](const std::vector<ObservationSelection*>& selections,
           const vidmap::MappingSidecars& sidecars,
           const std::map<colmap::image_t, vidmap::VectorXb>& outlier_masks) {
          std::map<std::pair<bool, bool>, ObservationSelection> groups;
          std::map<colmap::image_t, size_t> counts;
          for (const auto* selection : selections) {
            for (const auto& obs : selection->observations) {
              const auto& data = sidecars.images.at(obs.ImageId());
              const auto index = obs.Index();
              if (!MaskValue(data.depth_validity, index) ||
                  data.depth_stddevs(index) <= 1e-9)
                continue;
              const auto mask = outlier_masks.find(obs.ImageId());
              const bool outlier = mask == outlier_masks.end()
                                       ? MaskValue(data.is_depth_outlier, index)
                                       : MaskValue(mask->second, index);
              groups[{obs.loop, outlier}].observations.push_back(obs);
              ++counts[obs.ImageId()];
            }
          }
          return py::make_tuple(groups, counts);
        });
  m.def("append_depth_observations",
        [](ceres::Problem& problem,
           const colmap::Reconstruction& reconstruction,
           const vidmap::MappingSidecars& sidecars,
           const ObservationSelection& selection,
           const py::dict& scales,
           const std::shared_ptr<ceres::LossFunction>& loss,
           bool use_log_scales,
           vidmap::MetricDepthResidualType residual_type,
           bool zero_residual_behind,
           double log_linear_threshold) {
          const auto scale_pointers = CenterPointers(scales);
          for (const auto& obs : selection.observations) {
            const auto& data = sidecars.images.at(obs.ImageId());
            auto* cost = vidmap::MetricDepthError::Create(
                reconstruction.Image(obs.ImageId()).CamFromWorld().rotation(),
                data.depth_values(obs.Index()),
                data.depth_stddevs(obs.Index()),
                use_log_scales,
                residual_type,
                zero_residual_behind,
                log_linear_threshold);
            problem.AddResidualBlock(cost,
                                     loss.get(),
                                     obs.center,
                                     obs.point,
                                     scale_pointers.at(obs.ImageId()));
          }
        });
  m.def("playback_point_ids",
        [](ceres::Problem& problem,
           const colmap::Reconstruction& reconstruction,
           const std::vector<colmap::point3D_t>& requested,
           size_t maximum) {
          std::vector<colmap::point3D_t> ids = requested;
          if (ids.empty()) {
            for (const auto& [id, point] : reconstruction.Points3D())
              if (problem.HasParameterBlock(point.xyz.data()))
                ids.push_back(id);
            std::sort(ids.begin(), ids.end());
          }
          const size_t stride =
              requested.empty()
                  ? std::max(size_t(1), (ids.size() + maximum - 1) / maximum)
                  : 1;
          py::array_t<colmap::point3D_t> selected((ids.size() + stride - 1) /
                                                  stride);
          for (size_t i = 0; i < ids.size(); i += stride) {
            if (!reconstruction.ExistsPoint3D(ids[i]) ||
                !problem.HasParameterBlock(
                    reconstruction.Point3D(ids[i]).xyz.data()))
              throw std::invalid_argument(
                  "Playback point is not active in global positioning");
            *selected.mutable_data(i / stride) = ids[i];
          }
          return selected;
        });
  m.def("playback_edges",
        [](const std::vector<ObservationSelection*>& selections,
           const std::vector<std::shared_ptr<ceres::LossFunction>>& losses,
           const std::unordered_set<colmap::image_t>& images) {
          using Feature = std::pair<colmap::image_t, colmap::point2D_t>;
          using Edge = std::pair<colmap::image_t, colmap::image_t>;
          struct EdgeScore {
            std::set<std::pair<Feature, Feature>> matches;
            double score = 0.0;
          };
          std::map<Edge, EdgeScore> edges;
          for (size_t group = 0; group < selections.size(); ++group) {
            for (const auto& obs : selections[group]->observations) {
              if (!obs.loop) continue;
              if (obs.anchor.image_id == colmap::kInvalidImageId)
                throw std::invalid_argument(
                    "Playback requires exact loop-closure anchors");
              const auto anchor_image = obs.anchor.image_id;
              if (!images.count(obs.ImageId()) || !images.count(anchor_image))
                continue;
              const Feature a{obs.ImageId(), obs.Index()},
                  b{anchor_image, obs.anchor.point2D_idx};
              auto& edge = edges[std::minmax(obs.ImageId(), anchor_image)];
              edge.matches.insert(std::minmax(a, b));
              const double* parameters[] = {obs.center, obs.point, obs.scale};
              Eigen::Vector3d residual;
              // Evaluate the cost directly: Problem::Evaluate is unavailable in
              // callbacks.
              obs.cost->Evaluate(parameters, residual.data(), nullptr);
              double rho[3];
              losses.at(group)->Evaluate(residual.squaredNorm(), rho);
              const double score =
                  std::max(rho[1], 0.0) * std::sqrt(std::max(rho[0], 0.0));
              if (std::isfinite(score)) edge.score += score;
            }
          }
          std::vector<Edge> order;
          for (const auto& [edge, score] : edges) order.push_back(edge);
          std::sort(
              order.begin(), order.end(), [&](const Edge& a, const Edge& b) {
                const auto na = edges.at(a).matches.size(),
                           nb = edges.at(b).matches.size();
                return na != nb ? na > nb : a < b;
              });
          vidmap::MatrixX2u pairs(order.size(), 2);
          Eigen::Matrix<uint64_t, Eigen::Dynamic, 1> counts(order.size());
          Eigen::VectorXd scores(order.size());
          for (size_t i = 0; i < order.size(); ++i) {
            pairs(i, 0) = order[i].first;
            pairs(i, 1) = order[i].second;
            counts(i) = edges.at(order[i]).matches.size();
            scores(i) = edges.at(order[i]).score;
          }
          return py::make_tuple(pairs, counts, scores);
        });
  m.def("temporal_acceleration_cost",
        [](double dt_prev, double dt_next, double stddev) {
          return std::shared_ptr<ceres::CostFunction>(
              vidmap::TemporalAccelerationCostFunctor::Create(
                  dt_prev, dt_next, 1.0 / stddev));
        });
}
