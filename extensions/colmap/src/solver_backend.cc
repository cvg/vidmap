#include "vidmap_native/solver_backend.h"

#include <stdexcept>

#include <ceres/version.h>

#define VIDMAP_CERES_HAS_CUDSS \
  (CERES_VERSION_MAJOR > 2 ||  \
   (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3))

namespace vidmap {

void SolverBackendOptions::Validate() const {
  if (!use_cuda) return;

  switch (linear_solver) {
    case LinearSolverType::kDenseSchur:
      if (!ceres::IsDenseLinearAlgebraLibraryTypeAvailable(ceres::CUDA)) {
        throw std::invalid_argument("Ceres was built without CUDA support");
      }
      return;
    case LinearSolverType::kSparseSchur:
#if VIDMAP_CERES_HAS_CUDSS && !defined(CERES_NO_CUDSS)
      if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(
              ceres::CUDA_SPARSE)) {
        return;
      }
#endif
      throw std::invalid_argument(
          "sparse_schur requires Ceres 2.3 with CUDA and cuDSS");
    case LinearSolverType::kIterativeSchur:
      throw std::invalid_argument("iterative_schur is CPU-only");
  }
}

void SolverBackendOptions::Apply(ceres::Solver::Options* solver_options) const {
  Validate();
  switch (linear_solver) {
    case LinearSolverType::kDenseSchur:
      solver_options->linear_solver_type = ceres::DENSE_SCHUR;
      break;
    case LinearSolverType::kSparseSchur:
      solver_options->linear_solver_type = ceres::SPARSE_SCHUR;
      break;
    case LinearSolverType::kIterativeSchur:
      solver_options->linear_solver_type = ceres::ITERATIVE_SCHUR;
      break;
  }
  switch (preconditioner) {
    case PreconditionerType::kJacobi:
      solver_options->preconditioner_type = ceres::JACOBI;
      break;
    case PreconditionerType::kSchurJacobi:
      solver_options->preconditioner_type = ceres::SCHUR_JACOBI;
      break;
    case PreconditionerType::kClusterJacobi:
      solver_options->preconditioner_type = ceres::CLUSTER_JACOBI;
      break;
    case PreconditionerType::kClusterTridiagonal:
      solver_options->preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
      break;
  }
  if (use_cuda && linear_solver == LinearSolverType::kDenseSchur) {
    solver_options->dense_linear_algebra_library_type = ceres::CUDA;
  }
  if (use_cuda && linear_solver == LinearSolverType::kSparseSchur) {
#if VIDMAP_CERES_HAS_CUDSS
    solver_options->sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
#endif
  }
}

}  // namespace vidmap
