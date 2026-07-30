#pragma once

#include <memory>

#include <ceres/loss_function.h>

namespace vidmap {

enum class LossFunctionType {
  kTrivial,
  kSoftL1,
  kCauchy,
  kHuber,
};

struct LossConfig {
  LossFunctionType type = LossFunctionType::kTrivial;
  double scale = 1.0;
  double weight = 1.0;

  void Validate() const;
  std::unique_ptr<ceres::LossFunction> Create() const;
};

}  // namespace vidmap
