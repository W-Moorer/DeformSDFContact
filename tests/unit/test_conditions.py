#!/usr/bin/env python3
"""Unit tests for prototype external-load and boundary-condition contracts."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc

from deformsdfcontact.assembly import (
    BlockDirichletCondition,
    BoundaryConditionContract,
    MonolithicBlockLayout,
    StructuralNodalLoad,
    accumulate_structural_nodal_loads,
    apply_dirichlet_to_residual_and_jacobian,
    apply_dirichlet_values_to_state,
)


def _dense_matrix(mat: PETSc.Mat) -> np.ndarray:
    rows = np.arange(mat.getSize()[0], dtype=np.int32)
    cols = np.arange(mat.getSize()[1], dtype=np.int32)
    return np.asarray(mat.getValues(rows, cols), dtype=float)


def test_structural_nodal_loads_accumulate_into_one_vector() -> None:
    loads = (
        StructuralNodalLoad(
            u_dofs=np.array([0, 2], dtype=np.int32),
            values=np.array([1.5, -0.5], dtype=float),
        ),
        StructuralNodalLoad(
            u_dofs=np.array([2, 3], dtype=np.int32),
            values=np.array([0.25, 2.0], dtype=float),
        ),
    )

    accumulated = accumulate_structural_nodal_loads(5, loads)
    assert np.allclose(accumulated, np.array([1.5, 0.0, -0.25, 2.0, 0.0], dtype=float))


def test_apply_dirichlet_values_to_state_clamps_both_blocks() -> None:
    layout = MonolithicBlockLayout(ndof_u=4, ndof_phi=3)
    boundary_conditions = BoundaryConditionContract(
        conditions=(
            BlockDirichletCondition("u", np.array([1], dtype=np.int32), np.array([2.5], dtype=float)),
            BlockDirichletCondition("phi", np.array([0, 2], dtype=np.int32), np.array([-1.0, 3.0], dtype=float)),
        )
    )
    state = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    clamped = apply_dirichlet_values_to_state(layout, state, boundary_conditions)
    assert np.allclose(clamped, np.array([0.0, 2.5, 2.0, 3.0, -1.0, 5.0, 3.0], dtype=float))


def test_apply_dirichlet_to_residual_and_jacobian_sets_identity_rows() -> None:
    layout = MonolithicBlockLayout(ndof_u=3, ndof_phi=2)
    boundary_conditions = BoundaryConditionContract(
        conditions=(
            BlockDirichletCondition("u", np.array([1], dtype=np.int32), np.array([1.5], dtype=float)),
            BlockDirichletCondition("phi", np.array([0], dtype=np.int32), np.array([-0.25], dtype=float)),
        )
    )
    state = np.array([0.2, 2.0, -0.3, 0.6, 1.2], dtype=float)

    residual = PETSc.Vec().createSeq(layout.total_dofs)
    residual.setValues(np.arange(layout.total_dofs, dtype=np.int32), np.arange(10.0, 15.0, dtype=float))
    residual.assemblyBegin()
    residual.assemblyEnd()

    jacobian = PETSc.Mat().createAIJ(size=(layout.total_dofs, layout.total_dofs), nnz=layout.total_dofs)
    jacobian.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    dense_values = np.arange(1.0, 26.0, dtype=float).reshape(5, 5)
    jacobian.setValues(np.arange(5, dtype=np.int32), np.arange(5, dtype=np.int32), dense_values)
    jacobian.assemblyBegin()
    jacobian.assemblyEnd()

    apply_dirichlet_to_residual_and_jacobian(layout, state, residual, jacobian, boundary_conditions)

    residual_values = np.asarray(residual.getArray(), dtype=float)
    dense_jacobian = _dense_matrix(jacobian)
    assert np.isclose(residual_values[1], state[1] - 1.5)
    assert np.isclose(residual_values[3], state[3] + 0.25)
    assert np.allclose(dense_jacobian[1], np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(dense_jacobian[3], np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float))
    assert np.allclose(dense_jacobian[:, 1], np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float))
    assert np.allclose(dense_jacobian[:, 3], np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float))
