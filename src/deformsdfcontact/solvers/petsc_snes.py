"""Prototype PETSc SNES solver path for the transition monolithic problem."""

from __future__ import annotations

from dataclasses import dataclass

from petsc4py import PETSc

from .diagnostics import SolverDiagnosticsResult, describe_block_matrix, monitor_snes_iteration, split_residual_norms


@dataclass(frozen=True)
class PETScSNESSolveResult:
    """Outcome of one prototype PETSc SNES solve."""

    solution: PETSc.Vec
    converged_reason: int
    iteration_count: int
    residual_norm_history: tuple[float, ...]
    final_residual_norm: float
    diagnostics: SolverDiagnosticsResult


def solve_with_petsc_snes(
    problem: object,
    *,
    initial_guess: object | None = None,
    snes_type: str = "newtonls",
    ksp_type: str = "preonly",
    pc_type: str = "lu",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    max_it: int = 20,
    collect_diagnostics: bool = True,
) -> PETScSNESSolveResult:
    """Solve a transition monolithic problem through a minimal PETSc SNES path."""

    state = problem.create_initial_guess() if initial_guess is None else initial_guess.copy()
    residual = problem.create_residual_vector()
    jacobian = problem.create_jacobian_matrix()
    snes = PETSc.SNES().create(comm=problem.comm)
    residual_history: list[float] = []
    iteration_diagnostics: list[object] = []

    def _function_callback(snes_obj, x, f) -> None:
        problem.assemble_residual_into(x, f)

    def _jacobian_callback(snes_obj, x, A, B):
        problem.assemble_jacobian_into(x, B)
        if A is not B:
            B.copy(A)
        return PETSc.Mat.Structure.SAME_NONZERO_PATTERN

    def _monitor(snes_obj, iteration: int, norm: float) -> None:
        residual_history.append(float(norm))
        if collect_diagnostics:
            iteration_diagnostics.append(monitor_snes_iteration(problem, snes_obj, iteration, norm))

    snes.setType(snes_type)
    snes.setFunction(_function_callback, residual)
    snes.setJacobian(_jacobian_callback, jacobian, jacobian)
    snes.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
    snes.setMonitor(_monitor)

    ksp = snes.getKSP()
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)

    snes.solve(None, state)
    final_system = problem.assemble_system(state)
    final_residual = final_system.R
    final_jacobian = final_system.K
    diagnostics = SolverDiagnosticsResult(
        iterations=tuple(iteration_diagnostics),
        final_residual_split=split_residual_norms(problem.layout, final_residual),
        jacobian_description=describe_block_matrix(problem.layout, final_jacobian),
        converged_reason=int(snes.getConvergedReason()),
        iteration_count=int(snes.getIterationNumber()),
        linear_solve_iterations=int(snes.getLinearSolveIterations()),
        linear_solve_failures=int(snes.getLinearSolveFailures()),
        contact_summary=final_system.contact_summary,
    )
    return PETScSNESSolveResult(
        solution=state,
        converged_reason=int(snes.getConvergedReason()),
        iteration_count=int(snes.getIterationNumber()),
        residual_norm_history=tuple(residual_history),
        final_residual_norm=float(final_residual.norm()),
        diagnostics=diagnostics,
    )


__all__ = ["PETScSNESSolveResult", "solve_with_petsc_snes"]
