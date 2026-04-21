"""Prototype query-point contact backend for DOLFINx 0.3.0.

This adapter is still backend-specific and prototype-scoped:

- it evaluates one fixed slave query point per boundary facet
- it reuses the backend-agnostic contact geometry and kernel layers
- it does not implement global contact search or ownership logic
"""

from __future__ import annotations

import numpy as np

from ...assembly import ContactLocalContribution
from ...contact import (
    AffineMasterMap2D,
    ContactLaw,
    build_contact_surface_mapping,
    execute_contact_surface_local_loop,
    query_point,
)
from .common import (
    barycentric_coordinates,
    expand_block_dofs,
    facet_midpoint_length_and_outward_normal,
    locate_boundary_facets,
    scalar_function_values,
)
from .contact_summary import ContactAssemblyResult, ContactAssemblySummary, ContactPointObservation


def _point_to_local(
    point: object,
    origin: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    point_array = np.asarray(point, dtype=float)
    delta = point_array - origin
    return np.array(
        [
            float(delta @ tangent),
            float(delta @ normal),
        ],
        dtype=float,
    )


def _point_to_global(
    local_point: object,
    origin: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    local_array = np.asarray(local_point, dtype=float)
    if local_array.shape != (2,):
        raise ValueError(f"local_point must have shape (2,), got {local_array.shape!r}")
    return origin + local_array[0] * tangent + local_array[1] * normal


def _facet_local_vertex_indices(
    cell_coordinates: np.ndarray,
    facet_coordinates: np.ndarray,
) -> tuple[int, int]:
    indices: list[int] = []
    for point in np.asarray(facet_coordinates, dtype=float):
        matches = np.where(np.all(np.isclose(cell_coordinates, point[None, :]), axis=1))[0]
        if matches.shape[0] != 1:
            raise ValueError("failed to locate one facet vertex in the local cell coordinates")
        indices.append(int(matches[0]))
    return int(indices[0]), int(indices[1])


def _component_step(value: float, scale: float = 1.0e-6) -> float:
    return scale * max(1.0, abs(float(value)))


def _query_gap_and_shape_values(
    *,
    cell_coordinates: np.ndarray,
    facet_local_vertices: tuple[int, int],
    reference_origin: np.ndarray,
    reference_tangent: np.ndarray,
    reference_normal: np.ndarray,
    reference_length: float,
    u_local: np.ndarray,
    phi_local: np.ndarray,
    gap_offset: float,
    query_point_fraction: float,
    query_point_normal_offset: float,
) -> tuple[float, np.ndarray]:
    vertex_a, vertex_b = facet_local_vertices
    x_a = cell_coordinates[vertex_a] + u_local[vertex_a]
    x_b = cell_coordinates[vertex_b] + u_local[vertex_b]
    origin_local = _point_to_local(x_a, reference_origin, reference_tangent, reference_normal)
    x_b_local = _point_to_local(x_b, reference_origin, reference_tangent, reference_normal)
    master_map = AffineMasterMap2D(
        origin=origin_local,
        tangent=x_b_local - origin_local,
    )

    x_slave_local = np.array(
        [
            float(query_point_fraction) * reference_length,
            float(query_point_normal_offset),
        ],
        dtype=float,
    )
    X_c_local = query_point(x_slave_local, master_map)
    X_c_global = _point_to_global(
        X_c_local,
        reference_origin,
        reference_tangent,
        reference_normal,
    )
    shape_values = barycentric_coordinates(X_c_global, cell_coordinates)
    phi_at_query = float(shape_values @ phi_local)
    return float(gap_offset - phi_at_query), shape_values


def _gap_gradient_wrt_u(
    gap_function: object,
    u_flat: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros_like(u_flat)
    for i in range(u_flat.shape[0]):
        step = _component_step(u_flat[i])
        perturbation = np.zeros_like(u_flat)
        perturbation[i] = step
        g_plus = float(gap_function(u_flat + perturbation))
        g_minus = float(gap_function(u_flat - perturbation))
        gradient[i] = (g_plus - g_minus) / (2.0 * step)
    return gradient


def _gap_hessian_wrt_u(
    gap_function: object,
    u_flat: np.ndarray,
) -> np.ndarray:
    size = u_flat.shape[0]
    hessian = np.zeros((size, size), dtype=float)
    base_gap = float(gap_function(u_flat))
    for i in range(size):
        step_i = _component_step(u_flat[i])
        perturb_i = np.zeros_like(u_flat)
        perturb_i[i] = step_i
        g_plus = float(gap_function(u_flat + perturb_i))
        g_minus = float(gap_function(u_flat - perturb_i))
        hessian[i, i] = (g_plus - 2.0 * base_gap + g_minus) / (step_i * step_i)
        for j in range(i + 1, size):
            step_j = _component_step(u_flat[j])
            perturb_j = np.zeros_like(u_flat)
            perturb_j[j] = step_j
            g_pp = float(gap_function(u_flat + perturb_i + perturb_j))
            g_pm = float(gap_function(u_flat + perturb_i - perturb_j))
            g_mp = float(gap_function(u_flat - perturb_i + perturb_j))
            g_mm = float(gap_function(u_flat - perturb_i - perturb_j))
            mixed = (g_pp - g_pm - g_mp + g_mm) / (4.0 * step_i * step_j)
            hessian[i, j] = mixed
            hessian[j, i] = mixed
    return hessian


def _gap_mixed_hessian_u_phi(
    gap_function: object,
    u_flat: np.ndarray,
    phi_local: np.ndarray,
) -> np.ndarray:
    mixed = np.zeros((u_flat.shape[0], phi_local.shape[0]), dtype=float)
    for i in range(u_flat.shape[0]):
        step_u = _component_step(u_flat[i])
        perturb_u = np.zeros_like(u_flat)
        perturb_u[i] = step_u
        for j in range(phi_local.shape[0]):
            step_phi = _component_step(phi_local[j])
            perturb_phi = np.zeros_like(phi_local)
            perturb_phi[j] = step_phi
            g_pp = float(gap_function(u_flat + perturb_u, phi_local + perturb_phi))
            g_pm = float(gap_function(u_flat + perturb_u, phi_local - perturb_phi))
            g_mp = float(gap_function(u_flat - perturb_u, phi_local + perturb_phi))
            g_mm = float(gap_function(u_flat - perturb_u, phi_local - perturb_phi))
            mixed[i, j] = (g_pp - g_pm - g_mp + g_mm) / (4.0 * step_u * step_phi)
    return mixed


def assemble_contact_query_local_contributions(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    phi_function: object,
    law: ContactLaw,
    *,
    gap_offset: float = 0.0,
    query_point_fraction: float = 0.35,
    query_point_normal_offset: float = 0.15,
    slave_boundary: str = "all",
) -> ContactAssemblyResult:
    """Assemble prototype query-point contact contributions on boundary facets.

    The query-point geometry is still intentionally small in scope:

    - one fixed slave point per boundary facet in the local facet frame
    - projection onto the deformed facet line
    - local P1 interpolation of `phi(X_c)` and its sensitivities
    """

    if not 0.0 <= float(query_point_fraction) <= 1.0:
        raise ValueError(
            "query_point_fraction must lie in [0, 1], "
            f"got {query_point_fraction!r}"
        )

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    boundary_facets = locate_boundary_facets(mesh, slave_boundary)

    geometry_coordinates = np.asarray(mesh.geometry.x[:, :tdim], dtype=float)
    phi_coordinates = np.asarray(phi_space.tabulate_dof_coordinates()[:, :tdim], dtype=float)
    phi_values = scalar_function_values(phi_function)
    displacement_values = scalar_function_values(displacement_function)
    block_size = int(displacement_space.dofmap.index_map_bs)
    if block_size != 2:
        raise ValueError(
            "assemble_contact_query_local_contributions currently supports only 2D vector spaces"
        )

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
        vertex_a, vertex_b = _facet_local_vertex_indices(cell_coordinates, facet_coordinates)

        midpoint, reference_length, reference_normal = facet_midpoint_length_and_outward_normal(
            facet_coordinates,
            cell_centroid,
        )
        reference_tangent = (facet_coordinates[1] - facet_coordinates[0]) / reference_length
        reference_origin = np.asarray(facet_coordinates[0], dtype=float)

        u_block_dofs = np.asarray(displacement_space.dofmap.cell_dofs(cell), dtype=np.int32)
        u_dofs = expand_block_dofs(u_block_dofs, block_size)
        u_local = np.asarray(displacement_values[u_dofs], dtype=float).reshape(-1, block_size)
        phi_local = np.asarray(phi_values[phi_dofs], dtype=float)
        u_flat = u_local.reshape(-1)

        def _gap_with_local_state(
            u_trial_flat: np.ndarray,
            phi_trial: np.ndarray | None = None,
        ) -> float:
            local_phi = phi_local if phi_trial is None else np.asarray(phi_trial, dtype=float)
            local_u = np.asarray(u_trial_flat, dtype=float).reshape(u_local.shape)
            gap_value, _ = _query_gap_and_shape_values(
                cell_coordinates=cell_coordinates,
                facet_local_vertices=(vertex_a, vertex_b),
                reference_origin=reference_origin,
                reference_tangent=reference_tangent,
                reference_normal=reference_normal,
                reference_length=reference_length,
                u_local=local_u,
                phi_local=local_phi,
                gap_offset=float(gap_offset),
                query_point_fraction=float(query_point_fraction),
                query_point_normal_offset=float(query_point_normal_offset),
            )
            return float(gap_value)

        g_n, shape_values = _query_gap_and_shape_values(
            cell_coordinates=cell_coordinates,
            facet_local_vertices=(vertex_a, vertex_b),
            reference_origin=reference_origin,
            reference_tangent=reference_tangent,
            reference_normal=reference_normal,
            reference_length=reference_length,
            u_local=u_local,
            phi_local=phi_local,
            gap_offset=float(gap_offset),
            query_point_fraction=float(query_point_fraction),
            query_point_normal_offset=float(query_point_normal_offset),
        )
        G_u = -_gap_gradient_wrt_u(_gap_with_local_state, u_flat)
        G_a = shape_values
        H_uu_g = -_gap_hessian_wrt_u(_gap_with_local_state, u_flat)
        H_uphi_g = -_gap_mixed_hessian_u_phi(_gap_with_local_state, u_flat, phi_local)

        mapping = build_contact_surface_mapping(
            g_n=g_n,
            G_u=G_u,
            G_a=G_a,
            H_uu_g=H_uu_g,
            H_uphi_g=H_uphi_g,
            weights=np.array([reference_length], dtype=float),
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
                weight=reference_length,
                reaction_scalar=float(point_result.lambda_n * reference_length),
                active=bool(point_result.lambda_n > 0.0),
                slave_point=_point_to_global(
                    np.array(
                        [
                            float(query_point_fraction) * reference_length,
                            float(query_point_normal_offset),
                        ],
                        dtype=float,
                    ),
                    reference_origin,
                    reference_tangent,
                    reference_normal,
                ),
                query_point=_point_to_global(
                    query_point(
                        np.array(
                            [
                                float(query_point_fraction) * reference_length,
                                float(query_point_normal_offset),
                            ],
                            dtype=float,
                        ),
                        AffineMasterMap2D(
                            origin=_point_to_local(
                                cell_coordinates[vertex_a] + u_local[vertex_a],
                                reference_origin,
                                reference_tangent,
                                reference_normal,
                            ),
                            tangent=_point_to_local(
                                cell_coordinates[vertex_b] + u_local[vertex_b],
                                reference_origin,
                                reference_tangent,
                                reference_normal,
                            )
                            - _point_to_local(
                                cell_coordinates[vertex_a] + u_local[vertex_a],
                                reference_origin,
                                reference_tangent,
                                reference_normal,
                            ),
                        ),
                    ),
                    reference_origin,
                    reference_tangent,
                    reference_normal,
                ),
            )
        )

    return ContactAssemblyResult(
        contributions=tuple(contributions),
        summary=ContactAssemblySummary.from_observations("query_point", tuple(observations)),
    )


__all__ = ["assemble_contact_query_local_contributions"]
