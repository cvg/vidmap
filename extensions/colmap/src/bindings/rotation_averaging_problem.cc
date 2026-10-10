#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// 3-DoF rotation error in the camera frame, r = Log(R_cw * R_prior_cw^T), on an
// Eigen quaternion (x, y, z, w) parameter block. This matches the tangent space
// of COLMAP's absolute pose priors.
struct AbsoluteRotationPriorCostFunctor
    : public colmap::
          AutoDiffCostFunctor<AbsoluteRotationPriorCostFunctor, 3, 4> {
  explicit AbsoluteRotationPriorCostFunctor(
      const Eigen::Quaterniond& cam_from_world_prior)
      : world_from_cam_prior_(cam_from_world_prior.normalized().inverse()) {}

  template <typename T>
  bool operator()(const T* const cam_from_world, T* residuals) const {
    const Eigen::Map<const Eigen::Quaternion<T>> rotation(cam_from_world);
    const Eigen::Quaternion<T> error =
        rotation * world_from_cam_prior_.cast<T>();
    const T error_wxyz[4] = {error.w(), error.x(), error.y(), error.z()};
    ceres::QuaternionToAngleAxis(error_wxyz, residuals);
    return true;
  }

  const Eigen::Quaterniond world_from_cam_prior_;
};

struct RotationPrior {
  double* block;
  Eigen::Quaterniond cam_from_world;
  Eigen::Matrix3d covariance;
  double confidence;
};

double* FrameRotationBlock(colmap::Reconstruction& reconstruction,
                           colmap::image_t image_id) {
  auto& image = reconstruction.Image(image_id);
  if (!image.IsRefInFrame()) {
    throw std::invalid_argument(
        "Rotation priors require reference cameras in their frames");
  }
  return image.FramePtr()->RigFromWorld().rotation().coeffs().data();
}

// Rotate the world frame of all solved rotations so that they agree with the
// priors: R_cw <- R_cw * R_align. This preserves all relative rotations.
void AlignToPriors(ceres::Problem& problem,
                   const std::vector<RotationPrior>& priors,
                   double max_error_deg) {
  const auto rotation = [](const double* block) {
    return Eigen::Quaterniond(Eigen::Map<const Eigen::Quaterniond>(block))
        .normalized();
  };
  const double truncation = colmap::DegToRad(max_error_deg);
  double best_cost = std::numeric_limits<double>::infinity();
  Eigen::Quaterniond best_align = Eigen::Quaterniond::Identity();
  for (const RotationPrior& candidate : priors) {
    // R_prior = R_cw * R_align  =>  R_align = R_cw^-1 * R_prior.
    const Eigen::Quaterniond align =
        rotation(candidate.block).inverse() * candidate.cam_from_world;
    double cost = 0.0;
    for (const RotationPrior& prior : priors) {
      const double angle =
          (rotation(prior.block) * align).angularDistance(prior.cam_from_world);
      cost += prior.confidence * std::min(angle, truncation);
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_align = align.normalized();
    }
  }

  // Refine with a confidence-weighted tangent mean over the inliers.
  Eigen::Vector3d delta_sum = Eigen::Vector3d::Zero();
  double weight_sum = 0.0;
  for (const RotationPrior& prior : priors) {
    const Eigen::Quaterniond align =
        rotation(prior.block).inverse() * prior.cam_from_world;
    if (best_align.angularDistance(align) > truncation) continue;
    const Eigen::AngleAxisd angle_axis(best_align.inverse() * align);
    Eigen::Vector3d delta = angle_axis.angle() * angle_axis.axis();
    if (angle_axis.angle() > EIGEN_PI) {
      delta = (angle_axis.angle() - 2.0 * EIGEN_PI) * angle_axis.axis();
    }
    delta_sum += prior.confidence * delta;
    weight_sum += prior.confidence;
  }
  if (weight_sum > 0.0) {
    const Eigen::Vector3d mean_delta = delta_sum / weight_sum;
    const double mean_angle = mean_delta.norm();
    if (mean_angle > 1e-12) {
      best_align = (best_align * Eigen::Quaterniond(Eigen::AngleAxisd(
                                     mean_angle, mean_delta / mean_angle)))
                       .normalized();
    }
  }

  std::vector<double*> blocks;
  problem.GetParameterBlocks(&blocks);
  for (double* block : blocks) {
    if (problem.ParameterBlockSize(block) != 4 ||
        problem.IsParameterBlockConstant(block)) {
      continue;
    }
    Eigen::Map<Eigen::Quaterniond> aligned(block);
    aligned = (rotation(block) * best_align).normalized();
  }
}

}  // namespace

PYBIND11_MODULE(rotation_averaging, m) {
  py::module_::import("pyceres");
  m.def(
      "add_rotation_priors",
      [](ceres::Problem& problem,
         colmap::Reconstruction& reconstruction,
         const std::vector<colmap::image_t>& image_ids,
         const std::vector<Eigen::Vector4d>& cam_from_world_xyzw,
         const std::vector<Eigen::Matrix3d>& covariances,
         const std::vector<double>& confidences,
         double weight,
         double cauchy_scale,
         double ref_sigma_deg,
         double min_sigma_deg,
         bool align,
         double align_max_error_deg) {
        const size_t count = image_ids.size();
        if (cam_from_world_xyzw.size() != count ||
            covariances.size() != count || confidences.size() != count) {
          throw std::invalid_argument("invalid rotation prior arrays");
        }
        if (!(weight >= 0.0) || !(cauchy_scale > 0.0) ||
            !(ref_sigma_deg > 0.0) || !(min_sigma_deg > 0.0) ||
            !(align_max_error_deg > 0.0)) {
          throw std::invalid_argument("invalid rotation prior options");
        }
        std::vector<RotationPrior> priors;
        for (size_t i = 0; i < count; ++i) {
          const Eigen::Vector4d& xyzw = cam_from_world_xyzw[i];
          if (!xyzw.allFinite() || xyzw.squaredNorm() <= 1e-12 ||
              !covariances[i].allFinite() || !std::isfinite(confidences[i]) ||
              confidences[i] < 0.0) {
            throw std::invalid_argument("invalid rotation prior");
          }
          if (confidences[i] == 0.0 ||
              !reconstruction.ExistsImage(image_ids[i]) ||
              !reconstruction.Image(image_ids[i]).HasFramePtr() ||
              !reconstruction.Image(image_ids[i]).FramePtr()->HasPose()) {
            continue;
          }
          double* block = FrameRotationBlock(reconstruction, image_ids[i]);
          if (!problem.HasParameterBlock(block)) continue;
          priors.push_back(
              {block,
               Eigen::Quaterniond(xyzw[3], xyzw[0], xyzw[1], xyzw[2])
                   .normalized(),
               covariances[i],
               confidences[i]});
        }

        using Losses = std::vector<std::unique_ptr<ceres::LossFunction>>;
        auto losses = std::make_unique<Losses>();
        if (weight > 0.0 && !priors.empty()) {
          // The priors fix the gauge, so free the frame held constant by the
          // averager.
          for (const auto& [frame_id, const_frame] : reconstruction.Frames()) {
            if (!const_frame.HasPose()) continue;
            double* block = reconstruction.Frame(frame_id)
                                .RigFromWorld()
                                .rotation()
                                .coeffs()
                                .data();
            if (problem.HasParameterBlock(block) &&
                problem.IsParameterBlockConstant(block)) {
              problem.SetParameterBlockVariable(block);
            }
          }
          if (align) AlignToPriors(problem, priors, align_max_error_deg);

          const double min_sigma = colmap::DegToRad(min_sigma_deg);
          const double ref_sigma = colmap::DegToRad(ref_sigma_deg);
          losses->reserve(priors.size());
          for (const RotationPrior& prior : priors) {
            Eigen::Matrix3d covariance =
                0.5 * (prior.covariance + prior.covariance.transpose());
            covariance.diagonal().array() += min_sigma * min_sigma;
            losses->push_back(std::make_unique<ceres::ScaledLoss>(
                new ceres::CauchyLoss(cauchy_scale),
                weight * prior.confidence * ref_sigma * ref_sigma,
                ceres::TAKE_OWNERSHIP));
            problem.AddResidualBlock(
                colmap::CovarianceWeightedCostFunctor<
                    AbsoluteRotationPriorCostFunctor>::
                    Create(covariance, prior.cam_from_world),
                losses->back().get(),
                prior.block);
          }
        }
        const size_t added = losses->size();
        return py::make_tuple(
            py::capsule(losses.release(),
                        [](void* ptr) { delete static_cast<Losses*>(ptr); }),
            added);
      },
      py::arg("problem"),
      py::arg("reconstruction"),
      py::arg("image_ids"),
      py::arg("cam_from_world_xyzw"),
      py::arg("covariances"),
      py::arg("confidences"),
      py::arg("weight"),
      py::arg("cauchy_scale"),
      py::arg("ref_sigma_deg"),
      py::arg("min_sigma_deg"),
      py::arg("align"),
      py::arg("align_max_error_deg") = 10.0,
      "Add absolute rotation priors on the frame rotations of a rotation "
      "averaging problem. With align, first rotate the world frame of the "
      "current rotations onto the priors. Keep the returned losses alive "
      "while using the problem.");
}
