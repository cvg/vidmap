#pragma once

#include <ceres/solver.h>

namespace vidmap {

enum class LinearSolverType {
  kSparseSchur,
  kIterativeSchur,
};

enum class PreconditionerType {
  kJacobi,
  kSchurJacobi,
  kClusterJacobi,
  kClusterTridiagonal,
};

// Which linear solver the large Schur-complement solves use. The sparse direct
// default factorizes on one thread, so the iterative solver is the option that
// scales with cores; CUDA moves the linear algebra onto the GPU and requires a
// Ceres built with CUDA support.
struct SolverBackendOptions {
  LinearSolverType linear_solver = LinearSolverType::kSparseSchur;
  PreconditionerType preconditioner = PreconditionerType::kSchurJacobi;
  bool use_cuda = false;

  void Validate() const;
  void Apply(ceres::Solver::Options* solver_options) const;
};

}  // namespace vidmap
