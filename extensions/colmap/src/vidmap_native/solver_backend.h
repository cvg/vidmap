#pragma once

#include <ceres/solver.h>

namespace vidmap {

enum class LinearSolverType {
  kSparseSchur,
  kIterativeSchur,
  kDenseSchur,
};

enum class PreconditionerType {
  kJacobi,
  kSchurJacobi,
  kClusterJacobi,
  kClusterTridiagonal,
};

// Which linear solver the large Schur-complement solves use. CUDA support is
// backend-specific: dense Schur requires CUDA-enabled Ceres 2.2 or newer,
// while sparse Schur requires Ceres 2.3 or newer built with CUDA and cuDSS.
// Iterative Schur is CPU-only here. Individual stages may further restrict a
// backend when their problem structure is unsuitable for it.
struct SolverBackendOptions {
  LinearSolverType linear_solver = LinearSolverType::kSparseSchur;
  PreconditionerType preconditioner = PreconditionerType::kSchurJacobi;
  bool use_cuda = false;

  void Validate() const;
  void Apply(ceres::Solver::Options* solver_options) const;
};

const char* CeresVersion();
bool IsCudaDenseSolverAvailable();
bool IsCudaSparseSolverAvailable();

}  // namespace vidmap
