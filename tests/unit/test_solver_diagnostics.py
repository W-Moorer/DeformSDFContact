#!/usr/bin/env python3
"""Unit tests for prototype solver diagnostics."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.backend.dolfinx0p3.problem import build_unit_square_toy_problem
from deformsdfcontact.solvers import (
    describe_block_matrix,
    solve_with_petsc_snes,
    split_residual_norms,
)


def test_split_residual_norms_matches_monolithic_partition() -> None:
    problem = build_unit_square_toy_problem(gap_offset=0.5)
    residual = problem.assemble_residual(problem.create_initial_guess())
    split = split_residual_norms(problem.layout, residual)
    residual_array = np.asarray(residual.getArray(), dtype=float)

    assert np.isclose(split.total, np.linalg.norm(residual_array))
    assert np.isclose(split.u, np.linalg.norm(residual_array[: problem.layout.ndof_u]))
    assert np.isclose(split.phi, np.linalg.norm(residual_array[problem.layout.phi_offset :]))


def test_describe_block_matrix_reports_shapes_and_nonzeros() -> None:
    problem = build_unit_square_toy_problem(gap_offset=0.5)
    jacobian = problem.assemble_jacobian(problem.create_initial_guess())
    description = describe_block_matrix(problem.layout, jacobian)
    layout = problem.layout

    assert description.shape == (layout.total_dofs, layout.total_dofs)
    assert description.block_shapes == dict(layout.block_shapes)
    assert description.block_nnz["uu"] > 0
    assert description.block_nnz["uphi"] > 0
    assert description.block_nnz["phiu"] > 0
    assert description.block_nnz["phiphi"] > 0
    assert description.total_nnz >= sum(description.block_nnz.values())


def test_snes_collects_iteration_and_block_diagnostics() -> None:
    problem = build_unit_square_toy_problem(gap_offset=0.5)
    result = solve_with_petsc_snes(problem, initial_guess=problem.create_initial_guess(), max_it=20)
    diagnostics = result.diagnostics

    assert result.converged_reason > 0
    assert diagnostics.converged_reason == result.converged_reason
    assert diagnostics.iteration_count == result.iteration_count
    assert diagnostics.linear_solve_failures == 0
    assert diagnostics.final_residual_split.total <= result.final_residual_norm + 1.0e-12
    assert diagnostics.final_residual_split.u >= 0.0
    assert diagnostics.final_residual_split.phi >= 0.0
    assert diagnostics.contact_summary is not None
    assert diagnostics.contact_summary.backend_name == problem.metadata["contact_backend"]
    assert diagnostics.contact_summary.candidate_count >= len(diagnostics.contact_summary.point_observations)
    assert len(diagnostics.iterations) >= 1
    assert diagnostics.iterations[0].iteration == 0
