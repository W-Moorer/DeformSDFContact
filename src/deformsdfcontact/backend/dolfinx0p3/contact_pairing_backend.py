"""Prototype pairing / ownership contact backend for DOLFINx 0.3.0.

This backend is still prototype-scoped, but it is more realistic than the
midpoint-only and same-facet query-point adapters:

- slave and master boundary subsets are distinct
- each slave facet chooses one owned master facet from a small candidate set
- the query point is the closest projection of the current slave midpoint onto
  the owned current master segment
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...assembly import ContactLocalContribution
from ...contact import ContactLaw, build_contact_surface_mapping, execute_contact_surface_local_loop
from .common import (
    barycentric_coordinates,
    expand_block_dofs,
    locate_boundary_facets,
    scalar_function_values,
)
from .contact_query_backend import (
    _facet_local_vertex_indices,
    _gap_gradient_wrt_u,
    _gap_hessian_wrt_u,
    _gap_mixed_hessian_u_phi,
)
from .contact_summary import ContactAssemblyResult, ContactAssemblySummary, ContactPointObservation


@dataclass(frozen=True)
class _FacetCandidate:
    facet: int
    cell: int
    phi_dofs: np.ndarray
    u_dofs: np.ndarray
    cell_coordinates: np.ndarray
    vertex_a: int
    vertex_b: int
    reference_a: np.ndarray
    reference_b: np.ndarray
    current_a: np.ndarray
    current_b: np.ndarray
    weight: float


def _current_segment_endpoints(
    cell_coordinates: np.ndarray,
    u_local: np.ndarray,
    vertex_a: int,
    vertex_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(cell_coordinates[vertex_a] + u_local[vertex_a], dtype=float),
        np.asarray(cell_coordinates[vertex_b] + u_local[vertex_b], dtype=float),
    )


def _current_midpoint(
    cell_coordinates: np.ndarray,
    u_local: np.ndarray,
    vertex_a: int,
    vertex_b: int,
) -> np.ndarray:
    point_a, point_b = _current_segment_endpoints(cell_coordinates, u_local, vertex_a, vertex_b)
    return 0.5 * (point_a + point_b)


def _projection_onto_segment(
    point: np.ndarray,
    segment_a: np.ndarray,
    segment_b: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    tangent = np.asarray(segment_b - segment_a, dtype=float)
    denom = float(tangent @ tangent)
    if denom <= 1.0e-14:
        raise ValueError("master segment must have positive length")
    raw_t = float((point - segment_a) @ tangent / denom)
    t = min(1.0, max(0.0, raw_t))
    query_point = segment_a + t * tangent
    distance = float(np.linalg.norm(point - query_point))
    return t, query_point, distance


def _combine_u_dofs(
    slave_u_dofs: np.ndarray,
    master_u_dofs: np.ndarray,
    displacement_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined_u_dofs = np.unique(np.concatenate([slave_u_dofs, master_u_dofs])).astype(np.int32)
    dof_to_position = {int(dof): i for i, dof in enumerate(combined_u_dofs.tolist())}
    slave_positions = np.array([dof_to_position[int(dof)] for dof in slave_u_dofs], dtype=np.int32)
    master_positions = np.array([dof_to_position[int(dof)] for dof in master_u_dofs], dtype=np.int32)
    combined_u_values = np.asarray(displacement_values[combined_u_dofs], dtype=float)
    return combined_u_dofs, slave_positions, master_positions, combined_u_values


def _build_boundary_facet_candidates(
    *,
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_values: np.ndarray,
    phi_coordinates: np.ndarray,
    geometry_coordinates: np.ndarray,
    selector: str,
) -> tuple[_FacetCandidate, ...]:
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    block_size = int(displacement_space.dofmap.index_map_bs)

    candidates: list[_FacetCandidate] = []
    for facet in locate_boundary_facets(mesh, selector):
        incident_cells = np.asarray(facet_to_cell.links(int(facet)), dtype=np.int32)
        if incident_cells.shape[0] == 0:
            continue
        cell = int(incident_cells[0])
        phi_dofs = np.asarray(phi_space.dofmap.cell_dofs(cell), dtype=np.int32)
        cell_coordinates = np.asarray(phi_coordinates[phi_dofs], dtype=float)
        facet_vertices = np.asarray(facet_to_vertex.links(int(facet)), dtype=np.int32)
        facet_coordinates = np.asarray(geometry_coordinates[facet_vertices], dtype=float)
        vertex_a, vertex_b = _facet_local_vertex_indices(cell_coordinates, facet_coordinates)
        u_block_dofs = np.asarray(displacement_space.dofmap.cell_dofs(cell), dtype=np.int32)
        u_dofs = expand_block_dofs(u_block_dofs, block_size)
        u_local = np.asarray(displacement_values[u_dofs], dtype=float).reshape(-1, block_size)
        current_a, current_b = _current_segment_endpoints(cell_coordinates, u_local, vertex_a, vertex_b)
        weight = float(np.linalg.norm(cell_coordinates[vertex_b] - cell_coordinates[vertex_a]))
        candidates.append(
            _FacetCandidate(
                facet=int(facet),
                cell=cell,
                phi_dofs=phi_dofs,
                u_dofs=u_dofs,
                cell_coordinates=cell_coordinates,
                vertex_a=vertex_a,
                vertex_b=vertex_b,
                reference_a=np.asarray(cell_coordinates[vertex_a], dtype=float),
                reference_b=np.asarray(cell_coordinates[vertex_b], dtype=float),
                current_a=current_a,
                current_b=current_b,
                weight=weight,
            )
        )
    return tuple(candidates)


def _select_owned_master_candidate(
    slave_point: np.ndarray,
    master_candidates: tuple[_FacetCandidate, ...],
) -> tuple[_FacetCandidate, float, np.ndarray]:
    if not master_candidates:
        raise ValueError("master_candidates must be non-empty")

    owned = master_candidates[0]
    owned_t, owned_query, best_distance = _projection_onto_segment(
        slave_point,
        owned.current_a,
        owned.current_b,
    )
    for candidate in master_candidates[1:]:
        t, query_point, distance = _projection_onto_segment(
            slave_point,
            candidate.current_a,
            candidate.current_b,
        )
        if distance < best_distance:
            owned = candidate
            owned_t = t
            owned_query = query_point
            best_distance = distance
    return owned, owned_t, owned_query


def assemble_contact_pairing_local_contributions(
    mesh: object,
    displacement_space: object,
    phi_space: object,
    displacement_function: object,
    phi_function: object,
    law: ContactLaw,
    *,
    gap_offset: float = 0.0,
    slave_boundary: str = "top",
    master_boundary: str = "bottom",
) -> ContactAssemblyResult:
    """Assemble prototype pairing-based query-point contact contributions."""

    displacement_values = scalar_function_values(displacement_function)
    phi_values = scalar_function_values(phi_function)
    tdim = mesh.topology.dim
    geometry_coordinates = np.asarray(mesh.geometry.x[:, :tdim], dtype=float)
    phi_coordinates = np.asarray(phi_space.tabulate_dof_coordinates()[:, :tdim], dtype=float)
    block_size = int(displacement_space.dofmap.index_map_bs)
    if block_size != 2:
        raise ValueError(
            "assemble_contact_pairing_local_contributions currently supports only 2D vector spaces"
        )

    slave_candidates = _build_boundary_facet_candidates(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_values=displacement_values,
        phi_coordinates=phi_coordinates,
        geometry_coordinates=geometry_coordinates,
        selector=slave_boundary,
    )
    master_candidates = _build_boundary_facet_candidates(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_values=displacement_values,
        phi_coordinates=phi_coordinates,
        geometry_coordinates=geometry_coordinates,
        selector=master_boundary,
    )
    if not master_candidates:
        return ContactAssemblyResult(
            contributions=(),
            summary=ContactAssemblySummary.from_observations("pairing", ()),
        )

    contributions: list[ContactLocalContribution] = []
    observations: list[ContactPointObservation] = []
    for slave in slave_candidates:
        slave_u_local = np.asarray(displacement_values[slave.u_dofs], dtype=float).reshape(-1, block_size)
        slave_point = _current_midpoint(
            slave.cell_coordinates,
            slave_u_local,
            slave.vertex_a,
            slave.vertex_b,
        )
        owned_master, _, _ = _select_owned_master_candidate(slave_point, master_candidates)
        master_phi_local = np.asarray(phi_values[owned_master.phi_dofs], dtype=float)
        combined_u_dofs, slave_positions, master_positions, combined_u_values = _combine_u_dofs(
            slave.u_dofs,
            owned_master.u_dofs,
            displacement_values,
        )

        def _closure_from_local_state(
            combined_u_trial: np.ndarray,
            phi_trial: np.ndarray | None = None,
        ) -> float:
            phi_local = master_phi_local if phi_trial is None else np.asarray(phi_trial, dtype=float)
            slave_u_trial = np.asarray(combined_u_trial[slave_positions], dtype=float).reshape(-1, block_size)
            master_u_trial = np.asarray(combined_u_trial[master_positions], dtype=float).reshape(-1, block_size)
            current_slave_point = _current_midpoint(
                slave.cell_coordinates,
                slave_u_trial,
                slave.vertex_a,
                slave.vertex_b,
            )
            current_master_a, current_master_b = _current_segment_endpoints(
                owned_master.cell_coordinates,
                master_u_trial,
                owned_master.vertex_a,
                owned_master.vertex_b,
            )
            t, _, _ = _projection_onto_segment(current_slave_point, current_master_a, current_master_b)
            reference_query = (1.0 - t) * owned_master.reference_a + t * owned_master.reference_b
            shape_values = barycentric_coordinates(reference_query, owned_master.cell_coordinates)
            phi_at_query = float(shape_values @ phi_local)
            return float(phi_at_query - gap_offset)

        closure = float(_closure_from_local_state(combined_u_values))
        g_n = -closure
        t_base, current_query_point, _ = _projection_onto_segment(
            slave_point,
            owned_master.current_a,
            owned_master.current_b,
        )
        reference_query = (1.0 - t_base) * owned_master.reference_a + t_base * owned_master.reference_b
        shape_values = barycentric_coordinates(reference_query, owned_master.cell_coordinates)

        G_u = _gap_gradient_wrt_u(_closure_from_local_state, combined_u_values)
        G_a = shape_values
        H_uu_g = _gap_hessian_wrt_u(_closure_from_local_state, combined_u_values)
        H_uphi_g = _gap_mixed_hessian_u_phi(_closure_from_local_state, combined_u_values, master_phi_local)

        mapping = build_contact_surface_mapping(
            g_n=g_n,
            G_u=G_u,
            G_a=G_a,
            H_uu_g=H_uu_g,
            H_uphi_g=H_uphi_g,
            weights=np.array([owned_master.weight], dtype=float),
        )
        local_result = execute_contact_surface_local_loop(mapping, law)
        point_result = local_result.point_results[0]
        contributions.append(
            ContactLocalContribution(
                u_dofs=combined_u_dofs,
                phi_dofs=owned_master.phi_dofs,
                R_u=local_result.local_residual_u,
                K_uu=local_result.local_K_uu,
                K_uphi=local_result.local_K_uphi,
            )
        )
        observations.append(
            ContactPointObservation(
                slave_facet=slave.facet,
                slave_cell=slave.cell,
                master_facet=owned_master.facet,
                master_cell=owned_master.cell,
                candidate_count=len(master_candidates),
                gap_n=g_n,
                lambda_n=point_result.lambda_n,
                weight=owned_master.weight,
                reaction_scalar=float(point_result.lambda_n * owned_master.weight),
                active=bool(point_result.lambda_n > 0.0),
                slave_point=slave_point,
                query_point=current_query_point,
            )
        )

    return ContactAssemblyResult(
        contributions=tuple(contributions),
        summary=ContactAssemblySummary.from_observations("pairing", tuple(observations)),
    )


__all__ = ["assemble_contact_pairing_local_contributions"]
