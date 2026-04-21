#!/usr/bin/env python3
"""Smoke tests for the first solver-capable monolithic prototype."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from deformsdfcontact.backend.dolfinx0p3.problem import build_unit_square_toy_problem
from deformsdfcontact.solvers import solve_with_petsc_snes


def _mat_to_dense(mat: PETSc.Mat) -> np.ndarray:
    rows = np.arange(mat.getSize()[0], dtype=np.int32)
    cols = np.arange(mat.getSize()[1], dtype=np.int32)
    return np.asarray(mat.getValues(rows, cols), dtype=float)


def test_reference_state_is_an_exact_transition_solution() -> None:
    problem = build_unit_square_toy_problem()
    reference = problem.create_reference_state()
    residual = problem.assemble_residual(reference)

    assert residual.norm() < 1.0e-10


def test_petsc_snes_solves_the_transition_toy_problem() -> None:
    problem = build_unit_square_toy_problem()
    initial_guess = problem.create_initial_guess()
    initial_residual_norm = float(problem.assemble_residual(initial_guess).norm())

    result = solve_with_petsc_snes(problem, initial_guess=initial_guess, max_it=20)
    solution = result.solution
    jacobian = problem.assemble_jacobian(solution)
    dense_jacobian = _mat_to_dense(jacobian)
    layout = problem.layout
    diagnostics = result.diagnostics

    assert result.converged_reason > 0
    assert result.iteration_count >= 1
    assert result.final_residual_norm < initial_residual_norm
    assert result.final_residual_norm < 1.0e-8
    assert len(result.residual_norm_history) >= 1
    assert diagnostics.converged_reason == result.converged_reason
    assert diagnostics.iteration_count == result.iteration_count
    assert diagnostics.final_residual_split.total <= result.final_residual_norm + 1.0e-12
    assert diagnostics.jacobian_description.shape == (layout.total_dofs, layout.total_dofs)
    assert diagnostics.jacobian_description.block_nnz["uu"] > 0
    assert diagnostics.jacobian_description.block_nnz["uphi"] > 0
    assert diagnostics.jacobian_description.block_nnz["phiu"] > 0
    assert diagnostics.jacobian_description.block_nnz["phiphi"] > 0
    assert dense_jacobian.shape == (layout.total_dofs, layout.total_dofs)
    assert np.linalg.norm(dense_jacobian[: layout.ndof_u, : layout.ndof_u]) > 0.0
    assert np.linalg.norm(dense_jacobian[: layout.ndof_u, layout.phi_offset :]) > 0.0
    assert np.linalg.norm(dense_jacobian[layout.phi_offset :, : layout.ndof_u]) > 0.0
    assert np.linalg.norm(dense_jacobian[layout.phi_offset :, layout.phi_offset :]) > 0.0
