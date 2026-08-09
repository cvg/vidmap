#include "vidmap_native/solver_backend.h"

#include <stdexcept>

#include <ceres/version.h>

// Sparse CUDA factorization only exists from Ceres 2.2 on; 2.1 offers the dense
// CUDA backend alone.
#define VIDMAP_CERES_HAS_CUDA_SPARSE \
  (CERES_VERSION_MAJOR > 2 || (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2))

namespace vidmap {

void SolverBackendOptions::Validate() const {
  if (!use_cuda) return;
  if (!ceres::IsDenseLinearAlgebraLibraryTypeAvailable(ceres::CUDA)) {
    throw std::invalid_argument("Ceres was built without CUDA support");
  }
#if VIDMAP_CERES_HAS_CUDA_SPARSE
  if (linear_solver == LinearSolverType::kSparseSchur &&
      !ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
    throw std::invalid_argument(
        "Ceres was built without the CUDA sparse linear solver");
  }
#else
  if (linear_solver == LinearSolverType::kSparseSchur) {
    throw std::invalid_argument(
        "sparse_schur has no CUDA backend before Ceres 2.2; use iterative_schur "
        "or build against a newer Ceres");
  }
#endif
}

void SolverBackendOptions::Apply(ceres::Solver::Options* solver_options) const {
  Validate();
  switch (linear_solver) {
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
  if (use_cuda) {
    solver_options->dense_linear_algebra_library_type = ceres::CUDA;
#if VIDMAP_CERES_HAS_CUDA_SPARSE
    solver_options->sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
#endif
  }
}

}  // namespace vidmap
