"""Prototype load-stepping / continuation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadStepResult:
    """Outcome of one prototype continuation step."""

    step_index: int
    parameter_name: str
    parameter_value: float
    converged: bool
    solve_result: object


@dataclass(frozen=True)
class LoadSteppingResult:
    """Outcome of one prototype load-stepping run."""

    step_results: tuple[LoadStepResult, ...]
    completed_all_steps: bool
    failed_step_index: int | None
    last_converged_step_index: int | None
    last_converged_parameter_value: float | None


def solve_with_load_stepping(
    problem_factory: object,
    parameter_values: object,
    *,
    parameter_name: str = "continuation_parameter",
    solve_function: object,
    initial_guess: object | None = None,
    solve_kwargs: dict[str, object] | None = None,
) -> LoadSteppingResult:
    """Solve a sequence of prototype problems with warm-start continuation."""

    solve_kwargs = {} if solve_kwargs is None else dict(solve_kwargs)
    step_values = tuple(float(value) for value in parameter_values)
    step_results: list[LoadStepResult] = []
    current_initial_guess = initial_guess
    last_converged_step_index: int | None = None
    last_converged_parameter_value: float | None = None

    for step_index, parameter_value in enumerate(step_values):
        problem = problem_factory(parameter_value)
        if current_initial_guess is not None:
            current_initial_guess = problem.create_state_vector(
                monolithic_values=current_initial_guess.getArray(),
                apply_dirichlet=True,
            )
        solve_result = solve_function(
            problem,
            initial_guess=current_initial_guess,
            **solve_kwargs,
        )
        converged = bool(getattr(solve_result, "converged_reason", -1) > 0)
        step_results.append(
            LoadStepResult(
                step_index=step_index,
                parameter_name=parameter_name,
                parameter_value=parameter_value,
                converged=converged,
                solve_result=solve_result,
            )
        )
        if not converged:
            return LoadSteppingResult(
                step_results=tuple(step_results),
                completed_all_steps=False,
                failed_step_index=step_index,
                last_converged_step_index=last_converged_step_index,
                last_converged_parameter_value=last_converged_parameter_value,
            )
        last_converged_step_index = step_index
        last_converged_parameter_value = parameter_value
        current_initial_guess = solve_result.solution.copy()

    return LoadSteppingResult(
        step_results=tuple(step_results),
        completed_all_steps=True,
        failed_step_index=None,
        last_converged_step_index=last_converged_step_index,
        last_converged_parameter_value=last_converged_parameter_value,
    )


__all__ = ["LoadStepResult", "LoadSteppingResult", "solve_with_load_stepping"]
