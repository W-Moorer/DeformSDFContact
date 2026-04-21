"""DOLFINx 0.3.0-compatible assembled monolithic dry run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc

from ...assembly import (
    AssemblyPlan,
    ContactLocalContribution,
    MonolithicBlockLayout,
    SDFLocalContribution,
    SolidLocalContribution,
    StructuralNodalLoad,
    accumulate_structural_nodal_loads,
)
from ...contact import ContactLaw
from ...materials import IsotropicElasticParameters
from .common import (
    add_values_to_matrix,
    add_values_to_vector,
    assemble_petsc_objects,
    create_petsc_matrix,
    create_petsc_vector,
    function_space_dimension,
)
from .contact_adapter import assemble_contact_local_contributions
from .contact_pairing_backend import assemble_contact_pairing_local_contributions
from .contact_query_backend import assemble_contact_query_local_contributions
from .contact_summary import ContactAssemblySummary
from .sdf_adapter import assemble_sdf_local_contributions
from .solid_adapter import assemble_solid_local_contributions


@dataclass(frozen=True)
class Dolfinx0p3MonolithicDryRunResult:
    """Assembled PETSc objects and local contributions for the dry run."""

    plan: AssemblyPlan
    R_u: PETSc.Vec
    R_phi: PETSc.Vec
    R: PETSc.Vec
    K_uu: PETSc.Mat
    K_uphi: PETSc.Mat
    K_phiu: PETSc.Mat
    K_phiphi: PETSc.Mat
    K: PETSc.Mat
    solid_contributions: tuple[SolidLocalContribution, ...]
    sdf_contributions: tuple[SDFLocalContribution, ...]
    contact_contributions: tuple[ContactLocalContribution, ...]
    contact_summary: ContactAssemblySummary


def _assemble_solid_block_pair(
    contribution: SolidLocalContribution,
    residual_u: PETSc.Vec,
    residual: PETSc.Vec,
    K_uu: PETSc.Mat,
    K: PETSc.Mat,
) -> None:
    add_values_to_vector(residual_u, contribution.u_dofs, contribution.R_u)
    add_values_to_vector(residual, contribution.u_dofs, contribution.R_u)
    add_values_to_matrix(K_uu, contribution.u_dofs, contribution.u_dofs, contribution.K_uu)
    add_values_to_matrix(K, contribution.u_dofs, contribution.u_dofs, contribution.K_uu)


def _assemble_contact_block_pair(
    contribution: ContactLocalContribution,
    layout: MonolithicBlockLayout,
    residual_u: PETSc.Vec,
    residual: PETSc.Vec,
    K_uu: PETSc.Mat,
    K_uphi: PETSc.Mat,
    K: PETSc.Mat,
) -> None:
    phi_rows = layout.lift_phi_dofs(contribution.phi_dofs)
    add_values_to_vector(residual_u, contribution.u_dofs, contribution.R_u)
    add_values_to_vector(residual, contribution.u_dofs, contribution.R_u)
    add_values_to_matrix(K_uu, contribution.u_dofs, contribution.u_dofs, contribution.K_uu)
    add_values_to_matrix(K_uphi, contribution.u_dofs, contribution.phi_dofs, contribution.K_uphi)
    add_values_to_matrix(K, contribution.u_dofs, contribution.u_dofs, contribution.K_uu)
    add_values_to_matrix(K, contribution.u_dofs, phi_rows, contribution.K_uphi)


def _assemble_sdf_block_pair(
    contribution: SDFLocalContribution,
    layout: MonolithicBlockLayout,
    residual_phi: PETSc.Vec,
    residual: PETSc.Vec,
    K_phiu: PETSc.Mat,
    K_phiphi: PETSc.Mat,
    K: PETSc.Mat,
) -> None:
    phi_rows = layout.lift_phi_dofs(contribution.phi_dofs)
    add_values_to_vector(residual_phi, contribution.phi_dofs, contribution.R_phi)
    add_values_to_vector(residual, phi_rows, contribution.R_phi)
    add_values_to_matrix(K_phiu, contribution.phi_dofs, contribution.u_dofs, contribution.K_phiu)
    add_values_to_matrix(K_phiphi, contribution.phi_dofs, contribution.phi_dofs, contribution.K_phiphi)
    add_values_to_matrix(K, phi_rows, contribution.u_dofs, contribution.K_phiu)
    add_values_to_matrix(K, phi_rows, phi_rows, contribution.K_phiphi)


def _apply_structural_external_loads(
    layout: MonolithicBlockLayout,
    residual_u: PETSc.Vec,
    residual: PETSc.Vec,
    external_loads: tuple[StructuralNodalLoad, ...],
) -> None:
    external_vector = accumulate_structural_nodal_loads(layout.ndof_u, external_loads)
    if not np.any(external_vector):
        return
    structural_dofs = np.arange(layout.ndof_u, dtype=np.int32)
    add_values_to_vector(residual_u, structural_dofs, -external_vector)
    add_values_to_vector(residual, structural_dofs, -external_vector)


def assemble_monolithic_from_local_contributions(
    layout: MonolithicBlockLayout,
    comm: object,
    solid_contributions: tuple[SolidLocalContribution, ...],
    sdf_contributions: tuple[SDFLocalContribution, ...],
    contact_contributions: tuple[ContactLocalContribution, ...],
    contact_summary: ContactAssemblySummary,
    *,
    external_loads: tuple[StructuralNodalLoad, ...] = (),
    plan_note: str = "Transition adapter for the first assembled monolithic dry run.",
) -> Dolfinx0p3MonolithicDryRunResult:
    """Assemble PETSc residual and tangent blocks from validated local contributions."""

    plan = AssemblyPlan(
        layout=layout,
        backend_name="dolfinx0p3",
        note=plan_note,
    )

    R_u = create_petsc_vector(layout.ndof_u, comm)
    R_phi = create_petsc_vector(layout.ndof_phi, comm)
    R = create_petsc_vector(layout.total_dofs, comm)
    K_uu = create_petsc_matrix(layout.ndof_u, layout.ndof_u, comm)
    K_uphi = create_petsc_matrix(layout.ndof_u, layout.ndof_phi, comm)
    K_phiu = create_petsc_matrix(layout.ndof_phi, layout.ndof_u, comm)
    K_phiphi = create_petsc_matrix(layout.ndof_phi, layout.ndof_phi, comm)
    K = create_petsc_matrix(layout.total_dofs, layout.total_dofs, comm)

    for contribution in solid_contributions:
        _assemble_solid_block_pair(contribution, R_u, R, K_uu, K)
    for contribution in sdf_contributions:
        _assemble_sdf_block_pair(contribution, layout, R_phi, R, K_phiu, K_phiphi, K)
    for contribution in contact_contributions:
        _assemble_contact_block_pair(contribution, layout, R_u, R, K_uu, K_uphi, K)
    _apply_structural_external_loads(layout, R_u, R, tuple(external_loads))

    assemble_petsc_objects(R_u, R_phi, R, K_uu, K_uphi, K_phiu, K_phiphi, K)
    return Dolfinx0p3MonolithicDryRunResult(
        plan=plan,
        R_u=R_u,
        R_phi=R_phi,
        R=R,
        K_uu=K_uu,
        K_uphi=K_uphi,
        K_phiu=K_phiu,
        K_phiphi=K_phiphi,
        K=K,
        solid_contributions=solid_contributions,
        sdf_contributions=sdf_contributions,
        contact_contributions=contact_contributions,
        contact_summary=contact_summary,
    )


def assemble_monolithic_dry_run(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    phi_function: object,
    *,
    solid_params: IsotropicElasticParameters,
    contact_law: ContactLaw,
    reinitialize_beta: float = 0.0,
    phi_target: object = 0.0,
    contact_gap_offset: float = 0.0,
    contact_backend: str = "transition",
    contact_query_point_fraction: float = 0.35,
    contact_query_point_normal_offset: float = 0.15,
    contact_slave_boundary: str = "all",
    contact_master_boundary: str = "all",
    external_loads: tuple[StructuralNodalLoad, ...] = (),
) -> Dolfinx0p3MonolithicDryRunResult:
    """Assemble a monolithic PETSc residual and tangent without solving.

    This is a DOLFINx 0.3.0-compatible transition adapter.
    """

    comm = mesh.mpi_comm()
    layout = MonolithicBlockLayout(
        ndof_u=function_space_dimension(displacement_space),
        ndof_phi=function_space_dimension(phi_space),
    )

    solid_contributions = assemble_solid_local_contributions(
        mesh,
        displacement_space,
        phi_space,
        displacement_function,
        solid_params,
    )
    sdf_contributions = assemble_sdf_local_contributions(
        mesh,
        displacement_space,
        phi_space,
        displacement_function,
        phi_function,
        beta=reinitialize_beta,
        phi_target=phi_target,
    )
    if contact_backend == "transition":
        contact_result = assemble_contact_local_contributions(
            mesh,
            displacement_space,
            phi_space,
            displacement_function,
            phi_function,
            contact_law,
            gap_offset=contact_gap_offset,
            slave_boundary=contact_slave_boundary,
        )
    elif contact_backend == "query_point":
        contact_result = assemble_contact_query_local_contributions(
            mesh,
            displacement_space,
            phi_space,
            displacement_function,
            phi_function,
            contact_law,
            gap_offset=contact_gap_offset,
            query_point_fraction=contact_query_point_fraction,
            query_point_normal_offset=contact_query_point_normal_offset,
            slave_boundary=contact_slave_boundary,
        )
    elif contact_backend == "pairing":
        contact_result = assemble_contact_pairing_local_contributions(
            mesh,
            displacement_space,
            phi_space,
            displacement_function,
            phi_function,
            contact_law,
            gap_offset=contact_gap_offset,
            slave_boundary=contact_slave_boundary,
            master_boundary=contact_master_boundary,
        )
    else:
        raise ValueError(
            "contact_backend must be one of {'transition', 'query_point', 'pairing'}, "
            f"got {contact_backend!r}"
        )

    return assemble_monolithic_from_local_contributions(
        layout,
        comm,
        solid_contributions,
        sdf_contributions,
        contact_result.contributions,
        contact_result.summary,
        external_loads=tuple(external_loads),
        plan_note=(
            "Transition DOLFINx 0.3.0 monolithic dry run "
            f"with contact backend '{contact_backend}'."
        ),
    )


__all__ = [
    "Dolfinx0p3MonolithicDryRunResult",
    "assemble_monolithic_dry_run",
    "assemble_monolithic_from_local_contributions",
]
