#!/usr/bin/env python3
"""Smoke tests for switchable transition/query-point contact backends."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.backend.dolfinx0p3.problem import build_unit_square_toy_problem
from deformsdfcontact.solvers import solve_with_load_stepping, solve_with_petsc_snes


def _clamped_direction(problem: object) -> np.ndarray:
    direction = np.linspace(0.1, 1.0, problem.layout.total_dofs, dtype=float)
    for condition in problem.boundary_conditions.conditions:
        if condition.block == "u":
            direction[np.asarray(condition.dofs, dtype=np.int32)] = 0.0
        else:
            direction[problem.layout.phi_offset + np.asarray(condition.dofs, dtype=np.int32)] = 0.0
    direction /= np.linalg.norm(direction)
    return direction


def _dense_matrix(mat: object) -> np.ndarray:
    rows = np.arange(mat.getSize()[0], dtype=np.int32)
    cols = np.arange(mat.getSize()[1], dtype=np.int32)
    return np.asarray(mat.getValues(rows, cols), dtype=float)


def _run_contact_backend_stepping(contact_backend: str):
    return solve_with_load_stepping(
        lambda phi_scale: build_unit_square_toy_problem(
            contact_backend=contact_backend,
            gap_offset=0.5,
            phi_scale=phi_scale,
        ),
        [0.6, 0.8, 1.0],
        parameter_name="phi_scale",
        solve_function=solve_with_petsc_snes,
        solve_kwargs={"max_it": 20},
    )


def test_transition_contact_backend_baseline_stepped_solve_runs() -> None:
    stepped = _run_contact_backend_stepping("transition")

    assert stepped.completed_all_steps
    assert all(step.converged for step in stepped.step_results)
    assert all(step.solve_result.diagnostics.final_residual_split.total < 1.0e-8 for step in stepped.step_results)


def test_query_point_contact_backend_stepped_solve_runs() -> None:
    stepped = _run_contact_backend_stepping("query_point")

    assert stepped.completed_all_steps
    assert all(step.converged for step in stepped.step_results)
    assert all(step.solve_result.diagnostics.final_residual_split.total < 1.0e-7 for step in stepped.step_results)


def test_query_point_backend_jacobian_matches_directional_finite_difference() -> None:
    problem = build_unit_square_toy_problem(contact_backend="query_point", gap_offset=0.5)
    state = problem.create_initial_guess()
    state_array = np.asarray(state.getArray(), dtype=float)
    direction = _clamped_direction(problem)
    jacobian = _dense_matrix(problem.assemble_jacobian(state))
    eps = 1.0e-7

    residual_plus = np.asarray(
        problem.assemble_residual(
            problem.create_state_vector(monolithic_values=state_array + eps * direction, apply_dirichlet=True)
        ).getArray(),
        dtype=float,
    )
    residual_minus = np.asarray(
        problem.assemble_residual(
            problem.create_state_vector(monolithic_values=state_array - eps * direction, apply_dirichlet=True)
        ).getArray(),
        dtype=float,
    )
    directional_fd = (residual_plus - residual_minus) / (2.0 * eps)
    directional_jacobian = jacobian @ direction
    relative_error = np.linalg.norm(directional_fd - directional_jacobian) / max(
        np.linalg.norm(directional_fd),
        1.0e-14,
    )

    assert relative_error < 1.0e-2


def test_switching_contact_backend_changes_contact_response_but_not_solver_interface() -> None:
    transition_problem = build_unit_square_toy_problem(contact_backend="transition", gap_offset=0.5)
    query_problem = build_unit_square_toy_problem(contact_backend="query_point", gap_offset=0.5)
    transition_initial = transition_problem.create_initial_guess()
    query_initial = query_problem.create_initial_guess()
    transition_residual = transition_problem.assemble_residual(transition_initial)
    query_residual = query_problem.assemble_residual(query_initial)

    assert transition_problem.layout.total_dofs == query_problem.layout.total_dofs
    assert transition_problem.assemble_jacobian(transition_initial).getSize() == (
        query_problem.layout.total_dofs,
        query_problem.layout.total_dofs,
    )
    assert not np.isclose(float(transition_residual.norm()), float(query_residual.norm()))
