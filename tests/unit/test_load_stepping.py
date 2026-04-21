#!/usr/bin/env python3
"""Unit tests for prototype load stepping."""

from __future__ import annotations

from deformsdfcontact.backend.dolfinx0p3.problem import build_unit_square_toy_problem
from deformsdfcontact.solvers import solve_with_load_stepping, solve_with_petsc_snes


def test_transition_load_stepping_completes_sequence() -> None:
    stepped = solve_with_load_stepping(
        lambda phi_scale: build_unit_square_toy_problem(
            contact_backend="transition",
            gap_offset=0.5,
            phi_scale=phi_scale,
        ),
        [0.6, 0.8, 1.0],
        parameter_name="phi_scale",
        solve_function=solve_with_petsc_snes,
        solve_kwargs={"max_it": 20},
    )

    assert stepped.completed_all_steps
    assert stepped.failed_step_index is None
    assert stepped.last_converged_step_index == 2
    assert stepped.last_converged_parameter_value == 1.0
    assert [step.parameter_value for step in stepped.step_results] == [0.6, 0.8, 1.0]
    assert all(step.converged for step in stepped.step_results)
    assert all(step.solve_result.converged_reason > 0 for step in stepped.step_results)
    assert all(step.solve_result.final_residual_norm < 1.0e-8 for step in stepped.step_results)
