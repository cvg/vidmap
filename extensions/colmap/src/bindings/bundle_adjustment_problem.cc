#include "colmap/estimators/ceres_loss_function.h"
#include "colmap/scene/reconstruction.h"

#include <stdexcept>

#include "stages/depth_prior.h"
#include "stages/intrinsics_prior.h"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(bundle_adjustment, m) {
  py::module_::import("pyceres");
  m.def("append_depth_observations",
        [](ceres::Problem& problem,
           colmap::Reconstruction& reconstruction,
           colmap::image_t image_id,
           const Eigen::Matrix<colmap::point3D_t, Eigen::Dynamic, 1>& point_ids,
           const Eigen::VectorXd& depths,
           const Eigen::VectorXd& robust_scales,
           const Eigen::VectorXd& weights,
           const Eigen::Matrix<bool, Eigen::Dynamic, 1>& robust_mask,
           colmap::CeresLossFunctionType loss_type,
           py::array_t<double, py::array::c_style> log_scale,
           bool fix_pose) {
          const auto count = point_ids.size();
          if (depths.size() != count || robust_scales.size() != count ||
              weights.size() != count || robust_mask.size() != count ||
              log_scale.size() != 1 || !log_scale.writeable()) {
            throw std::invalid_argument("invalid BA depth observation arrays");
          }
          if (!depths.allFinite() || (depths.array() <= 0).any() ||
              !robust_scales.allFinite() ||
              (robust_scales.array() <= 0).any() || !weights.allFinite() ||
              (weights.array() < 0).any()) {
            throw std::invalid_argument("invalid BA depth or loss values");
          }
          // The upstream BA problem owns costs, but not loss functions.
          using Losses = std::vector<std::unique_ptr<ceres::LossFunction>>;
          auto losses = std::make_unique<Losses>();
          auto& image = reconstruction.Image(image_id);
          double* pose = image.FramePtr()->RigFromWorld().params.data();
          if (fix_pose || problem.HasParameterBlock(pose)) {
            losses->reserve(count);
            for (Eigen::Index i = 0; i < count; ++i) {
              if (weights[i] == 0.0) continue;
              double* point = reconstruction.Point3D(point_ids[i]).xyz.data();
              if (!problem.HasParameterBlock(point)) continue;
              if (!image.IsRefInFrame()) {
                throw std::invalid_argument(
                    "VidMap depth constraints require a reference camera");
              }
              losses->push_back(colmap::CreateCeresLossFunction(
                  robust_mask[i] ? loss_type
                                 : colmap::CeresLossFunctionType::TRIVIAL,
                  robust_scales[i],
                  weights[i]));
              problem.AddResidualBlock(
                  vidmap::LogScaledDepthErrorCostFunctor::Create(depths[i]),
                  losses->back().get(),
                  pose,
                  point,
                  log_scale.mutable_data());
            }
            // Constant-pose reprojection costs omit this parameter block.
            if (!losses->empty() && fix_pose) {
              problem.SetParameterBlockConstant(pose);
            }
          }
          const auto added = losses->size();
          return py::make_tuple(
              py::capsule(losses.release(),
                          [](void* ptr) { delete static_cast<Losses*>(ptr); }),
              added);
        });
  m.def("focal_prior_cost",
        [](const colmap::Camera& camera, double focal, double stddev) {
          const auto indices = camera.FocalLengthIdxs();
          return std::shared_ptr<ceres::CostFunction>(
              new vidmap::LogMeanFocalPriorCostFunction(
                  camera.params.size(),
                  std::vector<std::size_t>(indices.begin(), indices.end()),
                  focal,
                  stddev));
        });
  m.def("relative_focal_prior_cost",
        [](const colmap::Camera& camera1,
           const colmap::Camera& camera2,
           double target_log_ratio,
           double sigma_log_ratio) {
          const auto indices1 = camera1.FocalLengthIdxs();
          const auto indices2 = camera2.FocalLengthIdxs();
          return std::shared_ptr<ceres::CostFunction>(
              new vidmap::LogRelativeFocalPriorCostFunction(
                  camera1.params.size(),
                  std::vector<std::size_t>(indices1.begin(), indices1.end()),
                  camera2.params.size(),
                  std::vector<std::size_t>(indices2.begin(), indices2.end()),
                  target_log_ratio,
                  sigma_log_ratio));
        });
}
