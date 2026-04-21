#!/usr/bin/env python3
"""Smoke tests for transition monolithic residual/Jacobian callables."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, generation

from deformsdfcontact.assembly import MonolithicBlockLayout
from deformsdfcontact.backend.dolfinx0p3 import assemble_monolithic_dry_run
from deformsdfcontact.backend.dolfinx0p3.callables import Dolfinx0p3ResidualJacobianCallables
from deformsdfcontact.contact import PenaltyContactLaw
from deformsdfcontact.materials import IsotropicElasticParameters


def _vector_space_dimension(space: object) -> int:
    return int(space.dofmap.index_map.size_global * space.dofmap.index_map_bs)


def _mat_to_dense(mat: PETSc.Mat) -> np.ndarray:
    rows = np.arange(mat.getSize()[0], dtype=np.int32)
    cols = np.arange(mat.getSize()[1], dtype=np.int32)
    return np.asarray(mat.getValues(rows, cols), dtype=float)


def _build_toy_functions(mesh: object, displacement_space: object, phi_space: object) -> tuple[object, object]:
    u = fem.Function(displacement_space)
    phi = fem.Function(phi_space)

    u_coordinates = displacement_space.tabulate_dof_coordinates()[:, : mesh.topology.dim]
    phi_coordinates = phi_space.tabulate_dof_coordinates()[:, : mesh.topology.dim]

    u_values = np.zeros(_vector_space_dimension(displacement_space), dtype=float)
    for dof, coordinate in enumerate(u_coordinates):
        x, y = coordinate
        u_values[2 * dof] = 0.05 * x + 0.01 * y
        u_values[2 * dof + 1] = -0.04 * x + 0.03 * y
    u.vector.setArray(u_values)

    phi_values = np.array([coordinate[1] + 0.1 * coordinate[0] for coordinate in phi_coordinates], dtype=float)
    phi.vector.setArray(phi_values)
    return u, phi


def test_transition_callables_match_assembled_dry_run_without_constraints() -> None:
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, phi = _build_toy_functions(mesh, displacement_space, phi_space)
    layout = MonolithicBlockLayout(
        ndof_u=_vector_space_dimension(displacement_space),
        ndof_phi=_vector_space_dimension(phi_space),
    )
    solid_params = IsotropicElasticParameters(E=11.0, nu=0.22)
    law = PenaltyContactLaw(penalty=18.0)

    callables = Dolfinx0p3ResidualJacobianCallables(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_function=u,
        phi_function=phi,
        layout=layout,
        solid_params=solid_params,
        contact_law=law,
        contact_gap_offset=0.6,
    )
    state = callables.create_state_vector(
        u_values=np.asarray(u.vector.getArray(), dtype=float),
        phi_values=np.asarray(phi.vector.getArray(), dtype=float),
    )
    dry_run = assemble_monolithic_dry_run(
        mesh,
        displacement_space,
        phi_space,
        u,
        phi,
        solid_params=solid_params,
        contact_law=law,
        contact_gap_offset=0.6,
    )

    residual = callables.assemble_residual(state)
    jacobian = callables.assemble_jacobian(state)
    assert np.allclose(np.asarray(residual.getArray(), dtype=float), np.asarray(dry_run.R.getArray(), dtype=float))
    assert np.allclose(_mat_to_dense(jacobian), _mat_to_dense(dry_run.K))


def test_transition_callables_jacobian_matches_finite_difference_on_free_direction() -> None:
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, phi = _build_toy_functions(mesh, displacement_space, phi_space)
    layout = MonolithicBlockLayout(
        ndof_u=_vector_space_dimension(displacement_space),
        ndof_phi=_vector_space_dimension(phi_space),
    )
    callables = Dolfinx0p3ResidualJacobianCallables(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_function=u,
        phi_function=phi,
        layout=layout,
        solid_params=IsotropicElasticParameters(E=9.0, nu=0.21),
        contact_law=PenaltyContactLaw(penalty=20.0),
        reinitialize_beta=2.0,
        phi_target=np.asarray(phi.vector.getArray(), dtype=float).copy(),
        contact_gap_offset=0.75,
    )
    state = callables.create_state_vector(
        u_values=np.asarray(u.vector.getArray(), dtype=float),
        phi_values=np.asarray(phi.vector.getArray(), dtype=float),
    )
    jacobian = _mat_to_dense(callables.assemble_jacobian(state))
    direction = np.linspace(0.1, 0.6, layout.total_dofs, dtype=float)
    eps = 1.0e-7
    state_array = np.asarray(state.getArray(), dtype=float)

    residual_plus = np.asarray(
        callables.assemble_residual(
            callables.create_state_vector(monolithic_values=state_array + eps * direction)
        ).getArray(),
        dtype=float,
    )
    residual_minus = np.asarray(
        callables.assemble_residual(
            callables.create_state_vector(monolithic_values=state_array - eps * direction)
        ).getArray(),
        dtype=float,
    )
    fd = (residual_plus - residual_minus) / (2.0 * eps)
    assert np.allclose(jacobian @ direction, fd, rtol=1.0e-6, atol=1.0e-8)
