#include "vidmap_native/ceres_loss.h"

#include <cmath>
#include <stdexcept>

namespace vidmap {

void LossConfig::Validate() const {
  if (!std::isfinite(scale) || scale <= 0.0 || !std::isfinite(weight) ||
      weight < 0.0) {
    throw std::invalid_argument("invalid Ceres loss configuration");
  }
}

std::unique_ptr<ceres::LossFunction> LossConfig::Create() const {
  Validate();
  std::unique_ptr<ceres::LossFunction> loss;
  switch (type) {
    case LossFunctionType::kTrivial:
      loss = std::make_unique<ceres::TrivialLoss>();
      break;
    case LossFunctionType::kSoftL1:
      loss = std::make_unique<ceres::SoftLOneLoss>(scale);
      break;
    case LossFunctionType::kCauchy:
      loss = std::make_unique<ceres::CauchyLoss>(scale);
      break;
    case LossFunctionType::kHuber:
      loss = std::make_unique<ceres::HuberLoss>(scale);
      break;
  }
  if (weight != 1.0) {
    loss = std::make_unique<ceres::ScaledLoss>(
        loss.release(), weight, ceres::TAKE_OWNERSHIP);
  }
  return loss;
}

}  // namespace vidmap
