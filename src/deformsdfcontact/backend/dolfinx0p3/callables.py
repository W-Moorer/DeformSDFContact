"""Solver-neutral residual and Jacobian callables for the transition backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc

from ...assembly import (
    BoundaryConditionContract,
    MonolithicBlockLayout,
    StructuralNodalLoad,
    apply_dirichlet_to_jacobian,
    apply_dirichlet_to_residual,
    apply_dirichlet_to_residual_and_jacobian,
    apply_dirichlet_values_to_state,
)
from ...contact import ContactLaw
from ...materials import IsotropicElasticParameters
from .assembly import Dolfinx0p3MonolithicDryRunResult, assemble_monolithic_dry_run
from .common import create_petsc_matrix, create_petsc_vector, set_function_values


def _as_state_array(state: object, length: int) -> np.ndarray:
    if hasattr(state, "getArray"):
        array = np.asarray(state.getArray(), dtype=float)
    else:
        array = np.asarray(state, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"state must have shape ({length},), got {array.shape!r}")
    return array.copy()


@dataclass
class Dolfinx0p3ResidualJacobianCallables:
    """Residual and Jacobian callables for the DOLFINx 0.3.0 transition backend."""

    mesh: object
    displacement_space: object
    phi_space: object
    displacement_function: object
    phi_function: object
    layout: MonolithicBlockLayout
    solid_params: IsotropicElasticParameters
    contact_law: ContactLaw
    reinitialize_beta: float = 0.0
    phi_target: object = 0.0
    contact_gap_offset: float = 0.0
    contact_backend: str = "transition"
    contact_query_point_fraction: float = 0.35
    contact_query_point_normal_offset: float = 0.15
    contact_slave_boundary: str = "all"
    contact_master_boundary: str = "all"
    external_loads: tuple[StructuralNodalLoad, ...] = ()
    boundary_conditions: BoundaryConditionContract = BoundaryConditionContract()

    @property
    def comm(self) -> object:
        return self.mesh.mpi_comm()

    def split_state(self, state: object) -> tuple[np.ndarray, np.ndarray]:
        """Return structural and SDF subvectors from a monolithic state."""

        state_array = _as_state_array(state, self.layout.total_dofs)
        return (
            state_array[: self.layout.ndof_u],
            state_array[self.layout.phi_offset :],
        )

    def create_state_vector(
        self,
        *,
        monolithic_values: object | None = None,
        u_values: object | None = None,
        phi_values: object | None = None,
        apply_dirichlet: bool = False,
    ) -> PETSc.Vec:
        """Create a monolithic PETSc state vector."""

        if monolithic_values is not None:
            state_array = _as_state_array(monolithic_values, self.layout.total_dofs)
        else:
            if u_values is None:
                u_array = np.zeros(self.layout.ndof_u, dtype=float)
            else:
                u_array = np.asarray(u_values, dtype=float)
            if phi_values is None:
                phi_array = np.zeros(self.layout.ndof_phi, dtype=float)
            else:
                phi_array = np.asarray(phi_values, dtype=float)
            if u_array.shape != (self.layout.ndof_u,):
                raise ValueError(
                    f"u_values must have shape ({self.layout.ndof_u},), got {u_array.shape!r}"
                )
            if phi_array.shape != (self.layout.ndof_phi,):
                raise ValueError(
                    f"phi_values must have shape ({self.layout.ndof_phi},), got {phi_array.shape!r}"
                )
            state_array = np.concatenate([u_array, phi_array])

        if apply_dirichlet:
            state_array = apply_dirichlet_values_to_state(
                self.layout,
                state_array,
                self.boundary_conditions,
            )

        state = create_petsc_vector(self.layout.total_dofs, self.comm)
        state.setValues(
            np.arange(self.layout.total_dofs, dtype=np.int32),
            state_array,
            addv=PETSc.InsertMode.INSERT_VALUES,
        )
        state.assemblyBegin()
        state.assemblyEnd()
        return state

    def create_residual_vector(self) -> PETSc.Vec:
        """Create a residual work vector for `SNES`."""

        return create_petsc_vector(self.layout.total_dofs, self.comm)

    def create_jacobian_matrix(self) -> PETSc.Mat:
        """Create a Jacobian work matrix for `SNES`."""

        return create_petsc_matrix(self.layout.total_dofs, self.layout.total_dofs, self.comm)

    def _update_transition_functions(self, state: object) -> np.ndarray:
        clamped_state = apply_dirichlet_values_to_state(
            self.layout,
            state,
            self.boundary_conditions,
        )
        u_values = clamped_state[: self.layout.ndof_u]
        phi_values = clamped_state[self.layout.phi_offset :]
        set_function_values(self.displacement_function, u_values)
        set_function_values(self.phi_function, phi_values)
        return clamped_state

    def assemble_system(self, state: object) -> Dolfinx0p3MonolithicDryRunResult:
        """Assemble the monolithic residual and Jacobian at one state."""

        state_array = _as_state_array(state, self.layout.total_dofs)
        self._update_transition_functions(state_array)
        system = assemble_monolithic_dry_run(
            self.mesh,
            self.displacement_space,
            self.phi_space,
            self.displacement_function,
            self.phi_function,
            solid_params=self.solid_params,
            contact_law=self.contact_law,
            reinitialize_beta=self.reinitialize_beta,
            phi_target=self.phi_target,
            contact_gap_offset=self.contact_gap_offset,
            contact_backend=self.contact_backend,
            contact_query_point_fraction=self.contact_query_point_fraction,
            contact_query_point_normal_offset=self.contact_query_point_normal_offset,
            contact_slave_boundary=self.contact_slave_boundary,
            contact_master_boundary=self.contact_master_boundary,
            external_loads=tuple(self.external_loads),
        )
        apply_dirichlet_to_residual_and_jacobian(
            self.layout,
            state_array,
            system.R,
            system.K,
            self.boundary_conditions,
        )
        return system

    def assemble_residual(self, state: object) -> PETSc.Vec:
        """Assemble only the monolithic residual `R(x)`."""

        state_array = _as_state_array(state, self.layout.total_dofs)
        self._update_transition_functions(state_array)
        system = assemble_monolithic_dry_run(
            self.mesh,
            self.displacement_space,
            self.phi_space,
            self.displacement_function,
            self.phi_function,
            solid_params=self.solid_params,
            contact_law=self.contact_law,
            reinitialize_beta=self.reinitialize_beta,
            phi_target=self.phi_target,
            contact_gap_offset=self.contact_gap_offset,
            contact_backend=self.contact_backend,
            contact_query_point_fraction=self.contact_query_point_fraction,
            contact_query_point_normal_offset=self.contact_query_point_normal_offset,
            contact_slave_boundary=self.contact_slave_boundary,
            contact_master_boundary=self.contact_master_boundary,
            external_loads=tuple(self.external_loads),
        )
        apply_dirichlet_to_residual(
            self.layout,
            state_array,
            system.R,
            self.boundary_conditions,
        )
        return system.R

    def assemble_jacobian(self, state: object) -> PETSc.Mat:
        """Assemble only the monolithic Jacobian `J(x)`."""

        state_array = _as_state_array(state, self.layout.total_dofs)
        self._update_transition_functions(state_array)
        system = assemble_monolithic_dry_run(
            self.mesh,
            self.displacement_space,
            self.phi_space,
            self.displacement_function,
            self.phi_function,
            solid_params=self.solid_params,
            contact_law=self.contact_law,
            reinitialize_beta=self.reinitialize_beta,
            phi_target=self.phi_target,
            contact_gap_offset=self.contact_gap_offset,
            contact_backend=self.contact_backend,
            contact_query_point_fraction=self.contact_query_point_fraction,
            contact_query_point_normal_offset=self.contact_query_point_normal_offset,
            contact_slave_boundary=self.contact_slave_boundary,
            contact_master_boundary=self.contact_master_boundary,
            external_loads=tuple(self.external_loads),
        )
        apply_dirichlet_to_jacobian(
            self.layout,
            system.K,
            self.boundary_conditions,
        )
        return system.K

    def assemble_residual_into(self, state: object, residual: PETSc.Vec) -> None:
        """Assemble the monolithic residual into an existing PETSc vector."""

        self.assemble_residual(state).copy(residual)

    def assemble_jacobian_into(self, state: object, jacobian: PETSc.Mat) -> None:
        """Assemble the monolithic Jacobian into an existing PETSc matrix."""

        self.assemble_jacobian(state).copy(jacobian)


__all__ = ["Dolfinx0p3ResidualJacobianCallables"]
