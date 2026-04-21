"""DOLFINx 0.3.0-compatible transition adapter for contact local contributions."""

from __future__ import annotations

import numpy as np

from ...assembly import ContactLocalContribution
from ...contact import ContactLaw, build_contact_surface_mapping, execute_contact_surface_local_loop
from .common import (
    barycentric_coordinates,
    expand_block_dofs,
    facet_midpoint_length_and_outward_normal,
    locate_boundary_facets,
    scalar_function_values,
)
from .contact_summary import ContactAssemblyResult, ContactAssemblySummary, ContactPointObservation


def assemble_contact_local_contributions(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    phi_function: object,
    law: ContactLaw,
    *,
    gap_offset: float = 0.0,
    slave_boundary: str = "all",
) -> ContactAssemblyResult:
    """Assemble transition contact contributions on boundary facets.

    This is a DOLFINx 0.3.0-compatible midpoint-only adapter. It does not
    perform contact search or query-point ownership logic.
    """

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    boundary_facets = locate_boundary_facets(mesh, slave_boundary)

    geometry_coordinates = np.asarray(mesh.geometry.x[:, :tdim], dtype=float)
    phi_coordinates = phi_space.tabulate_dof_coordinates()[:, :tdim]
    phi_values = scalar_function_values(phi_function)
    displacement_values = scalar_function_values(displacement_function)
    block_size = int(displacement_space.dofmap.index_map_bs)
    zero_h_uu = np.zeros((3 * block_size, 3 * block_size), dtype=float)
    zero_h_uphi = np.zeros((3 * block_size, 3), dtype=float)

    contributions: list[ContactLocalContribution] = []
    observations: list[ContactPointObservation] = []
    for facet in np.asarray(boundary_facets, dtype=np.int32):
        incident_cells = np.asarray(facet_to_cell.links(int(facet)), dtype=np.int32)
        if incident_cells.shape[0] == 0:
            continue
        cell = int(incident_cells[0])
        phi_dofs = np.asarray(phi_space.dofmap.cell_dofs(cell), dtype=np.int32)
        cell_coordinates = np.asarray(phi_coordinates[phi_dofs], dtype=float)
        cell_centroid = np.mean(cell_coordinates, axis=0)

        facet_vertices = np.asarray(facet_to_vertex.links(int(facet)), dtype=np.int32)
        facet_coordinates = np.asarray(geometry_coordinates[facet_vertices], dtype=float)
        midpoint, length, normal = facet_midpoint_length_and_outward_normal(
            facet_coordinates,
            cell_centroid,
        )

        shape_values = barycentric_coordinates(midpoint, cell_coordinates)
        phi_local = phi_values[phi_dofs]

        G_u = np.zeros(3 * block_size, dtype=float)
        for a in range(3):
            start = block_size * a
            G_u[start : start + block_size] = shape_values[a] * normal

        u_block_dofs = np.asarray(displacement_space.dofmap.cell_dofs(cell), dtype=np.int32)
        u_dofs = expand_block_dofs(u_block_dofs, block_size)
        u_local = displacement_values[u_dofs]
        g_n = float(float(gap_offset) - G_u @ u_local - shape_values @ phi_local)

        mapping = build_contact_surface_mapping(
            g_n=g_n,
            G_u=G_u,
            G_a=shape_values,
            H_uu_g=zero_h_uu,
            H_uphi_g=zero_h_uphi,
            weights=np.array([length], dtype=float),
        )
        local_result = execute_contact_surface_local_loop(mapping, law)
        point_result = local_result.point_results[0]

        contributions.append(
            ContactLocalContribution(
                u_dofs=u_dofs,
                phi_dofs=phi_dofs,
                R_u=local_result.local_residual_u,
                K_uu=local_result.local_K_uu,
                K_uphi=local_result.local_K_uphi,
            )
        )
        observations.append(
            ContactPointObservation(
                slave_facet=int(facet),
                slave_cell=cell,
                master_facet=None,
                master_cell=None,
                candidate_count=1,
                gap_n=g_n,
                lambda_n=point_result.lambda_n,
                weight=length,
                reaction_scalar=float(point_result.lambda_n * length),
                active=bool(point_result.lambda_n > 0.0),
                slave_point=midpoint,
                query_point=midpoint,
            )
        )

    return ContactAssemblyResult(
        contributions=tuple(contributions),
        summary=ContactAssemblySummary.from_observations("transition", tuple(observations)),
    )


__all__ = ["assemble_contact_local_contributions"]
