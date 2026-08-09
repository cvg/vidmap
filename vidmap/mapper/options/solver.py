"""Ceres linear-solver backend selection shared by the large global solves."""

from typing import Literal

from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass as pydantic_dataclass

LinearSolverName = Literal["sparse_schur", "iterative_schur"]
PreconditionerName = Literal["jacobi", "schur_jacobi", "cluster_jacobi", "cluster_tridiagonal"]


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class SolverBackendOptions:
    """Which Ceres linear solver runs one stage's Schur complement.

    ``sparse_schur`` factorizes on a single thread, so ``iterative_schur`` is the
    setting that keeps scaling with cores; ``use_cuda`` moves the linear algebra onto
    the GPU and needs a CUDA-enabled Ceres. Both change the numerical path, so the
    defaults reproduce the direct sparse solve. ``preconditioner`` is only consulted
    by ``iterative_schur``.
    """

    linear_solver: LinearSolverName = "sparse_schur"
    preconditioner: PreconditionerName = "schur_jacobi"
    use_cuda: bool = False
