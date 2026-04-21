"""DOLFINx 0.3.0-compatible adapter for SDF local contributions."""

from __future__ import annotations

import numpy as np

from ...assembly import SDFLocalContribution
from ...sdf import (
    build_reinitialize_element_mapping,
    build_sdf_coupling_element_mapping,
    execute_reinitialize_local_loop,
    execute_sdf_coupling_local_loop,
    linearized_metric_sensitivity_from_shape_gradients,
)
from .common import expand_block_dofs, scalar_function_values, triangle_area_and_shape_gradients


def assemble_sdf_local_contributions(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    phi_function: object,
    *,
    beta: float = 0.0,
    phi_target: float = 0.0,
) -> tuple[SDFLocalContribution, ...]:
    """Assemble all local SDF contributions for the transition dry run.

    This is a DOLFINx 0.3.0-compatible path for 2D P1 triangles.
    """

    del displacement_function  # The current transition metric is linearized at the reference state.

    tdim = mesh.topology.dim
    num_cells = int(mesh.topology.index_map(tdim).size_local)
    phi_coordinates = phi_space.tabulate_dof_coordinates()[:, :tdim]
    phi_values = scalar_function_values(phi_function)
    block_size = int(displacement_space.dofmap.index_map_bs)

    contributions: list[SDFLocalContribution] = []
    for cell in range(num_cells):
        phi_dofs = np.asarray(phi_space.dofmap.cell_dofs(cell), dtype=np.int32)
        cell_coordinates = np.asarray(phi_coordinates[phi_dofs], dtype=float)
        area, shape_gradients = triangle_area_and_shape_gradients(cell_coordinates)
        phi_local = phi_values[phi_dofs]

        u_block_dofs = np.asarray(displacement_space.dofmap.cell_dofs(cell), dtype=np.int32)
        u_dofs = expand_block_dofs(u_block_dofs, block_size)
        A = np.eye(tdim, dtype=float)
        shape_values = np.full((1, phi_dofs.shape[0]), 1.0 / float(phi_dofs.shape[0]), dtype=float)
        shape_gradient_field = shape_gradients[None, :, :]

        reinitialize_mapping = build_reinitialize_element_mapping(
            phi_local=phi_local,
            shape_values=shape_values,
            shape_gradients=shape_gradient_field,
            A=A,
            weights=np.array([area], dtype=float),
            phi_target=phi_target,
            beta=beta,
        )
        residual_phi, K_phiphi = execute_reinitialize_local_loop(reinitialize_mapping)

        dA_du = linearized_metric_sensitivity_from_shape_gradients(shape_gradients)
        coupling_mapping = build_sdf_coupling_element_mapping(
            phi_local=phi_local,
            shape_gradients_phi=shape_gradients,
            A=A,
            dA_du=dA_du,
            weights=np.array([area], dtype=float),
        )
        K_phiu = execute_sdf_coupling_local_loop(coupling_mapping).local_K_phiu

        contributions.append(
            SDFLocalContribution(
                u_dofs=u_dofs,
                phi_dofs=phi_dofs,
                R_phi=residual_phi,
                K_phiu=K_phiu,
                K_phiphi=K_phiphi,
            )
        )

    return tuple(contributions)


__all__ = ["assemble_sdf_local_contributions"]
