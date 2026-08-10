"""Ceres linear-solver backend selection shared by the large global solves."""

from typing import Literal

from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass as pydantic_dataclass

LinearSolverName = Literal["dense_schur", "sparse_schur", "iterative_schur"]
PreconditionerName = Literal["jacobi", "schur_jacobi", "cluster_jacobi", "cluster_tridiagonal"]


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class SolverBackendOptions:
    """Which Ceres linear solver runs one stage's Schur complement.

    The CPU ``sparse_schur`` default preserves the established numerical path.
    With ``use_cuda``, ``dense_schur`` requires CUDA-enabled Ceres 2.2 or newer,
    while ``sparse_schur`` requires Ceres 2.3 or newer built with CUDA and cuDSS.
    Dense Schur is supported for bundle adjustment but not global positioning.
    ``iterative_schur`` is CPU-only. ``preconditioner`` is only consulted by
    ``iterative_schur``.
    """

    linear_solver: LinearSolverName = "sparse_schur"
    preconditioner: PreconditionerName = "schur_jacobi"
    use_cuda: bool = False
