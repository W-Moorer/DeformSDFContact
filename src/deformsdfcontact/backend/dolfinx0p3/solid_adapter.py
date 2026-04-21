"""DOLFINx 0.3.0-compatible adapter for solid local contributions."""

from __future__ import annotations

import numpy as np

from ...assembly import SolidLocalContribution
from ...materials import IsotropicElasticParameters
from ...solid import build_solid_element_mapping, execute_solid_local_loop, triangle_p1_B_matrix
from .common import expand_block_dofs, scalar_function_values, triangle_area_and_shape_gradients


def assemble_solid_local_contributions(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    params: IsotropicElasticParameters,
) -> tuple[SolidLocalContribution, ...]:
    """Assemble all local solid contributions for the transition dry run.

    This is a DOLFINx 0.3.0-compatible path for 2D P1 triangles.
    """

    tdim = mesh.topology.dim
    num_cells = int(mesh.topology.index_map(tdim).size_local)
    phi_coordinates = phi_space.tabulate_dof_coordinates()[:, :tdim]
    displacement_values = scalar_function_values(displacement_function)
    block_size = int(displacement_space.dofmap.index_map_bs)

    contributions: list[SolidLocalContribution] = []
    for cell in range(num_cells):
        phi_dofs = np.asarray(phi_space.dofmap.cell_dofs(cell), dtype=np.int32)
        cell_coordinates = np.asarray(phi_coordinates[phi_dofs], dtype=float)
        area, shape_gradients = triangle_area_and_shape_gradients(cell_coordinates)
        B = triangle_p1_B_matrix(shape_gradients)

        u_block_dofs = np.asarray(displacement_space.dofmap.cell_dofs(cell), dtype=np.int32)
        u_dofs = expand_block_dofs(u_block_dofs, block_size)
        u_local = displacement_values[u_dofs]

        mapping = build_solid_element_mapping(
            u_local=u_local,
            B=B,
            weights=np.array([area], dtype=float),
        )
        local_result = execute_solid_local_loop(mapping, params)
        contributions.append(
            SolidLocalContribution(
                u_dofs=u_dofs,
                R_u=local_result.local_residual_u,
                K_uu=local_result.local_K_uu,
            )
        )

    return tuple(contributions)


__all__ = ["assemble_solid_local_contributions"]
