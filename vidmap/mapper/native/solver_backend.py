"""Shared translation of solver-backend configuration into native Ceres selections."""

from vidmap.mapper.native.extension import native
from vidmap.mapper.options.solver import SolverBackendOptions


def apply_solver_backend(native_options, options: SolverBackendOptions) -> None:
    """Point one native stage's solver at the configured Ceres linear solver."""
    backend = native_options.solver_backend
    backend.linear_solver = {
        "dense_schur": native.LinearSolverType.DENSE_SCHUR,
        "sparse_schur": native.LinearSolverType.SPARSE_SCHUR,
        "iterative_schur": native.LinearSolverType.ITERATIVE_SCHUR,
    }[options.linear_solver]
    backend.preconditioner = {
        "jacobi": native.PreconditionerType.JACOBI,
        "schur_jacobi": native.PreconditionerType.SCHUR_JACOBI,
        "cluster_jacobi": native.PreconditionerType.CLUSTER_JACOBI,
        "cluster_tridiagonal": native.PreconditionerType.CLUSTER_TRIDIAGONAL,
    }[options.preconditioner]
    backend.use_cuda = options.use_cuda
    native_options.solver_backend = backend
