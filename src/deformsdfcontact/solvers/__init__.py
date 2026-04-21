"""Prototype solver paths."""

from .diagnostics import (
    BlockMatrixDescription,
    NonlinearIterationDiagnostics,
    ResidualNormSplit,
    SolverDiagnosticsResult,
    describe_block_matrix,
    monitor_snes_iteration,
    split_residual_norms,
)
from .load_stepping import LoadStepResult, LoadSteppingResult, solve_with_load_stepping
from .petsc_snes import PETScSNESSolveResult, solve_with_petsc_snes

__all__ = [
    "BlockMatrixDescription",
    "LoadStepResult",
    "LoadSteppingResult",
    "NonlinearIterationDiagnostics",
    "PETScSNESSolveResult",
    "ResidualNormSplit",
    "SolverDiagnosticsResult",
    "describe_block_matrix",
    "monitor_snes_iteration",
    "solve_with_load_stepping",
    "solve_with_petsc_snes",
    "split_residual_norms",
]
