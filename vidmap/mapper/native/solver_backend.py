"""Shared Ceres solver configuration and diagnostics."""

from dataclasses import dataclass

from vidmap.mapper.options.solver import SolverBackendOptions


@dataclass
class SolverDiagnostics:
    num_residual_blocks: int = 0
    num_parameter_blocks: int = 0
    num_parameters: int = 0
    num_iterations: int = 0
    termination_type: int = 0
    initial_cost: float = 0.0
    final_cost: float = 0.0

    def update(self, summary) -> None:
        for name in ("num_residual_blocks", "num_parameter_blocks", "num_parameters", "initial_cost", "final_cost"):
            setattr(self, name, getattr(summary, name))
        self.num_iterations = summary.num_successful_steps + summary.num_unsuccessful_steps
        self.termination_type = int(summary.termination_type)


def apply_solver_backend(solver, options: SolverBackendOptions) -> None:
    """Configure and validate the Ceres solver backend."""
    import pyceres

    if options.use_cuda and options.linear_solver == "iterative_schur":
        raise ValueError("iterative_schur is CPU-only")
    solver.linear_solver_type = getattr(pyceres.LinearSolverType, options.linear_solver.upper())
    solver.preconditioner_type = getattr(pyceres.PreconditionerType, options.preconditioner.upper())
    if options.use_cuda:
        if options.linear_solver == "dense_schur":
            solver.dense_linear_algebra_library_type = pyceres.DenseLinearAlgebraLibraryType.CUDA
        else:
            solver.sparse_linear_algebra_library_type = pyceres.SparseLinearAlgebraLibraryType("CUDA_SPARSE")
    valid, error = solver.IsValid()
    if not valid:
        raise ValueError(error)
