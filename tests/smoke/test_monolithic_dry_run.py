#!/usr/bin/env python3
"""Smoke tests for the DOLFINx 0.3.0-compatible assembled monolithic dry run."""

from __future__ import annotations

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, generation

from deformsdfcontact.backend.dolfinx0p3 import assemble_monolithic_dry_run
from deformsdfcontact.contact import PenaltyContactLaw
from deformsdfcontact.materials import IsotropicElasticParameters


def _vector_space_dimension(space: object) -> int:
    return int(space.dofmap.index_map.size_global * space.dofmap.index_map_bs)


def _mat_to_dense(mat: PETSc.Mat) -> np.ndarray:
    nrow, ncol = mat.getSize()
    rows = np.arange(nrow, dtype=np.int32)
    cols = np.arange(ncol, dtype=np.int32)
    return np.asarray(mat.getValues(rows, cols), dtype=float)


def _build_toy_functions(mesh: object, displacement_space: object, phi_space: object) -> tuple[object, object]:
    """Return DOLFINx 0.3.0-compatible toy fields with nontrivial block content."""

    u = fem.Function(displacement_space)
    phi = fem.Function(phi_space)

    u_coordinates = displacement_space.tabulate_dof_coordinates()[:, : mesh.topology.dim]
    phi_coordinates = phi_space.tabulate_dof_coordinates()[:, : mesh.topology.dim]

    u_values = np.zeros(_vector_space_dimension(displacement_space), dtype=float)
    for dof, coordinate in enumerate(u_coordinates):
        x, y = coordinate
        u_values[2 * dof] = 0.10 * x + 0.02 * y
        u_values[2 * dof + 1] = -0.03 * x + 0.07 * y
    u.vector.setArray(u_values)

    phi_values = np.array([0.8 * coordinate[1] - 0.05 for coordinate in phi_coordinates], dtype=float)
    phi.vector.setArray(phi_values)
    return u, phi


def _manual_sum(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layout = result.plan.layout
    R_u = np.zeros(layout.ndof_u, dtype=float)
    R_phi = np.zeros(layout.ndof_phi, dtype=float)
    K_uu = np.zeros((layout.ndof_u, layout.ndof_u), dtype=float)
    K_uphi = np.zeros((layout.ndof_u, layout.ndof_phi), dtype=float)
    K_phiu = np.zeros((layout.ndof_phi, layout.ndof_u), dtype=float)
    K_phiphi = np.zeros((layout.ndof_phi, layout.ndof_phi), dtype=float)

    for contribution in result.solid_contributions:
        R_u[contribution.u_dofs] += contribution.R_u
        K_uu[np.ix_(contribution.u_dofs, contribution.u_dofs)] += contribution.K_uu
    for contribution in result.contact_contributions:
        R_u[contribution.u_dofs] += contribution.R_u
        K_uu[np.ix_(contribution.u_dofs, contribution.u_dofs)] += contribution.K_uu
        K_uphi[np.ix_(contribution.u_dofs, contribution.phi_dofs)] += contribution.K_uphi
    for contribution in result.sdf_contributions:
        R_phi[contribution.phi_dofs] += contribution.R_phi
        K_phiu[np.ix_(contribution.phi_dofs, contribution.u_dofs)] += contribution.K_phiu
        K_phiphi[np.ix_(contribution.phi_dofs, contribution.phi_dofs)] += contribution.K_phiphi

    return R_u, R_phi, K_uu, K_uphi, K_phiu, K_phiphi


def test_monolithic_dry_run_builds_expected_petsc_blocks() -> None:
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, phi = _build_toy_functions(mesh, displacement_space, phi_space)

    result = assemble_monolithic_dry_run(
        mesh,
        displacement_space,
        phi_space,
        u,
        phi,
        solid_params=IsotropicElasticParameters(E=12.0, nu=0.25),
        contact_law=PenaltyContactLaw(penalty=30.0),
        reinitialize_beta=0.0,
        phi_target=0.0,
        contact_gap_offset=0.15,
    )

    layout = result.plan.layout
    assert result.plan.backend_name == "dolfinx0p3"
    assert layout.ndof_u == 8
    assert layout.ndof_phi == 4
    assert result.R.getSize() == 12
    assert result.K.getSize() == (12, 12)
    assert len(result.solid_contributions) == 2
    assert len(result.sdf_contributions) == 2
    assert len(result.contact_contributions) > 0
    assert np.linalg.norm(result.R_u.getArray()) > 0.0
    assert np.linalg.norm(result.R_phi.getArray()) > 0.0
    assert np.linalg.norm(_mat_to_dense(result.K_uu)) > 0.0
    assert np.linalg.norm(_mat_to_dense(result.K_uphi)) > 0.0
    assert np.linalg.norm(_mat_to_dense(result.K_phiu)) > 0.0
    assert np.linalg.norm(_mat_to_dense(result.K_phiphi)) > 0.0


def test_monolithic_dry_run_matches_manual_local_sum_and_block_placement() -> None:
    mesh = generation.UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    u, phi = _build_toy_functions(mesh, displacement_space, phi_space)

    result = assemble_monolithic_dry_run(
        mesh,
        displacement_space,
        phi_space,
        u,
        phi,
        solid_params=IsotropicElasticParameters(E=9.0, nu=0.2),
        contact_law=PenaltyContactLaw(penalty=40.0),
        reinitialize_beta=0.0,
        phi_target=0.0,
        contact_gap_offset=0.12,
    )

    manual_R_u, manual_R_phi, manual_K_uu, manual_K_uphi, manual_K_phiu, manual_K_phiphi = _manual_sum(result)
    layout = result.plan.layout
    monolithic_dense = _mat_to_dense(result.K)

    assert np.allclose(result.R_u.getArray(), manual_R_u)
    assert np.allclose(result.R_phi.getArray(), manual_R_phi)
    assert np.allclose(_mat_to_dense(result.K_uu), manual_K_uu)
    assert np.allclose(_mat_to_dense(result.K_uphi), manual_K_uphi)
    assert np.allclose(_mat_to_dense(result.K_phiu), manual_K_phiu)
    assert np.allclose(_mat_to_dense(result.K_phiphi), manual_K_phiphi)

    assert np.allclose(result.R.getArray()[: layout.ndof_u], manual_R_u)
    assert np.allclose(result.R.getArray()[layout.phi_offset :], manual_R_phi)
    assert np.allclose(monolithic_dense[: layout.ndof_u, : layout.ndof_u], manual_K_uu)
    assert np.allclose(monolithic_dense[: layout.ndof_u, layout.phi_offset :], manual_K_uphi)
    assert np.allclose(monolithic_dense[layout.phi_offset :, : layout.ndof_u], manual_K_phiu)
    assert np.allclose(monolithic_dense[layout.phi_offset :, layout.phi_offset :], manual_K_phiphi)
