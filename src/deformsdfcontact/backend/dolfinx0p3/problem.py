"""Minimal transition monolithic problem wrapper and toy-problem builder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dolfinx import fem, generation
from mpi4py import MPI

from ...assembly import (
    BlockDirichletCondition,
    BoundaryConditionContract,
    MonolithicBlockLayout,
    StructuralNodalLoad,
)
from ...contact import PenaltyContactLaw
from ...materials import IsotropicElasticParameters
from .assembly import assemble_monolithic_dry_run
from .callables import Dolfinx0p3ResidualJacobianCallables
from .common import expand_block_dofs, function_space_dimension, set_function_values


@dataclass
class TransitionMonolithicProblem:
    """Transition-environment monolithic problem wrapper for the first solver prototype."""

    mesh: object
    displacement_space: object
    phi_space: object
    callables: Dolfinx0p3ResidualJacobianCallables
    layout: MonolithicBlockLayout
    boundary_conditions: BoundaryConditionContract
    external_loads: tuple[StructuralNodalLoad, ...]
    initial_state: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def comm(self) -> object:
        return self.mesh.mpi_comm()

    def create_state_vector(
        self,
        *,
        monolithic_values: object | None = None,
        u_values: object | None = None,
        phi_values: object | None = None,
        apply_dirichlet: bool = False,
    ):
        """Create a PETSc monolithic state vector."""

        return self.callables.create_state_vector(
            monolithic_values=monolithic_values,
            u_values=u_values,
            phi_values=phi_values,
            apply_dirichlet=apply_dirichlet,
        )

    def create_initial_guess(self):
        """Create the default prototype initial guess."""

        if self.initial_state is None:
            return self.create_state_vector(apply_dirichlet=True)
        return self.create_state_vector(monolithic_values=self.initial_state, apply_dirichlet=True)

    def create_reference_state(self):
        """Create the manufactured reference state."""

        if self.reference_state is None:
            raise ValueError("reference_state is not available for this problem")
        return self.create_state_vector(monolithic_values=self.reference_state, apply_dirichlet=True)

    def create_residual_vector(self):
        """Create a residual work vector."""

        return self.callables.create_residual_vector()

    def create_jacobian_matrix(self):
        """Create a Jacobian work matrix."""

        return self.callables.create_jacobian_matrix()

    def assemble_system(self, state: object):
        """Assemble residual and Jacobian at one state."""

        return self.callables.assemble_system(state)

    def assemble_residual(self, state: object):
        """Assemble residual `R(x)`."""

        return self.callables.assemble_residual(state)

    def assemble_jacobian(self, state: object):
        """Assemble Jacobian `J(x)`."""

        return self.callables.assemble_jacobian(state)

    def assemble_residual_into(self, state: object, residual: object) -> None:
        """Assemble residual into an existing vector."""

        self.callables.assemble_residual_into(state, residual)

    def assemble_jacobian_into(self, state: object, jacobian: object) -> None:
        """Assemble Jacobian into an existing matrix."""

        self.callables.assemble_jacobian_into(state, jacobian)


def build_unit_square_toy_problem(
    *,
    comm: object = MPI.COMM_WORLD,
    solid_params: IsotropicElasticParameters | None = None,
    penalty: float = 25.0,
    gap_offset: float = 0.75,
    reinitialize_beta: float = 5.0,
    phi_scale: float = 1.0,
    contact_backend: str = "transition",
    contact_query_point_fraction: float = 0.35,
    contact_query_point_normal_offset: float = 0.15,
    contact_slave_boundary: str = "all",
    contact_master_boundary: str = "all",
) -> TransitionMonolithicProblem:
    """Build the smallest solver-capable transition toy problem.

    The manufactured reference state is:

    - `u* = 0`
    - `phi*(x, y) = phi_scale * y`
    """

    if solid_params is None:
        solid_params = IsotropicElasticParameters(E=10.0, nu=0.25)

    mesh = generation.UnitSquareMesh(comm, 1, 1)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    displacement_function = fem.Function(displacement_space)
    phi_function = fem.Function(phi_space)

    layout = MonolithicBlockLayout(
        ndof_u=function_space_dimension(displacement_space),
        ndof_phi=function_space_dimension(phi_space),
    )

    u_block_coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates()[:, : mesh.topology.dim],
        dtype=float,
    )
    phi_coordinates = np.asarray(phi_space.tabulate_dof_coordinates()[:, : mesh.topology.dim], dtype=float)

    left_u_block_dofs = np.where(np.isclose(u_block_coordinates[:, 0], 0.0))[0].astype(np.int32)
    left_u_dofs = expand_block_dofs(left_u_block_dofs, int(displacement_space.dofmap.index_map_bs))

    phi_fixed_mask = np.isclose(phi_coordinates[:, 0], 0.0) | (
        np.isclose(phi_coordinates[:, 0], 1.0) & np.isclose(phi_coordinates[:, 1], 0.0)
    )
    phi_fixed_dofs = np.where(phi_fixed_mask)[0].astype(np.int32)

    phi_exact = float(phi_scale) * phi_coordinates[:, 1].copy()
    u_exact = np.zeros(layout.ndof_u, dtype=float)
    reference_state = np.concatenate([u_exact, phi_exact])

    boundary_conditions = BoundaryConditionContract(
        conditions=(
            BlockDirichletCondition(
                block="u",
                dofs=left_u_dofs,
                values=np.zeros(left_u_dofs.shape[0], dtype=float),
            ),
            BlockDirichletCondition(
                block="phi",
                dofs=phi_fixed_dofs,
                values=phi_exact[phi_fixed_dofs],
            ),
        )
    )

    set_function_values(displacement_function, u_exact)
    set_function_values(phi_function, phi_exact)
    exact_assembled = assemble_monolithic_dry_run(
        mesh,
        displacement_space,
        phi_space,
        displacement_function,
        phi_function,
        solid_params=solid_params,
        contact_law=PenaltyContactLaw(penalty=penalty),
        reinitialize_beta=reinitialize_beta,
        phi_target=phi_exact,
        contact_gap_offset=gap_offset,
        contact_backend=contact_backend,
        contact_query_point_fraction=contact_query_point_fraction,
        contact_query_point_normal_offset=contact_query_point_normal_offset,
        contact_slave_boundary=contact_slave_boundary,
        contact_master_boundary=contact_master_boundary,
    )
    external_loads = (
        StructuralNodalLoad(
            u_dofs=np.arange(layout.ndof_u, dtype=np.int32),
            values=np.asarray(exact_assembled.R_u.getArray(), dtype=float).copy(),
        ),
    )

    initial_state = reference_state.copy()
    free_phi_dofs = np.setdiff1d(np.arange(layout.ndof_phi, dtype=np.int32), phi_fixed_dofs)
    initial_state[layout.phi_offset + free_phi_dofs] = 0.35 * float(phi_scale)

    callables = Dolfinx0p3ResidualJacobianCallables(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_function=displacement_function,
        phi_function=phi_function,
        layout=layout,
        solid_params=solid_params,
        contact_law=PenaltyContactLaw(penalty=penalty),
        reinitialize_beta=reinitialize_beta,
        phi_target=phi_exact,
        contact_gap_offset=gap_offset,
        contact_backend=contact_backend,
        contact_query_point_fraction=contact_query_point_fraction,
        contact_query_point_normal_offset=contact_query_point_normal_offset,
        contact_slave_boundary=contact_slave_boundary,
        contact_master_boundary=contact_master_boundary,
        external_loads=external_loads,
        boundary_conditions=boundary_conditions,
    )
    return TransitionMonolithicProblem(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        callables=callables,
        layout=layout,
        boundary_conditions=boundary_conditions,
        external_loads=external_loads,
        initial_state=initial_state,
        reference_state=reference_state,
        metadata={
            "backend": "dolfinx0p3",
            "problem_kind": "unit_square_manufactured_contact_prototype",
            "contact_backend": contact_backend,
            "contact_slave_boundary": contact_slave_boundary,
            "contact_master_boundary": contact_master_boundary,
            "gap_offset": float(gap_offset),
            "penalty": float(penalty),
            "phi_scale": float(phi_scale),
        },
    )


def build_contact_strip_benchmark(
    *,
    comm: object = MPI.COMM_WORLD,
    solid_params: IsotropicElasticParameters | None = None,
    penalty: float = 25.0,
    gap_offset: float = 0.4,
    reinitialize_beta: float = 5.0,
    phi_bias: float = 0.35,
    contact_backend: str = "transition",
    contact_query_point_fraction: float = 0.35,
    contact_query_point_normal_offset: float = 0.15,
) -> TransitionMonolithicProblem:
    """Build a small contact-strip mini benchmark with explicit slave/master subsets.

    Geometry:

    - unit square with a `2 x 2` triangular mesh
    - slave contact boundary: top
    - master contact boundary: bottom

    The benchmark remains manufactured for stability, but the contact geometry
    is more realistic than the original all-boundary toy problem.
    """

    if solid_params is None:
        solid_params = IsotropicElasticParameters(E=10.0, nu=0.25)

    mesh = generation.UnitSquareMesh(comm, 2, 2)
    displacement_space = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    phi_space = fem.FunctionSpace(mesh, ("Lagrange", 1))
    displacement_function = fem.Function(displacement_space)
    phi_function = fem.Function(phi_space)

    layout = MonolithicBlockLayout(
        ndof_u=function_space_dimension(displacement_space),
        ndof_phi=function_space_dimension(phi_space),
    )

    u_block_coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates()[:, : mesh.topology.dim],
        dtype=float,
    )
    phi_coordinates = np.asarray(phi_space.tabulate_dof_coordinates()[:, : mesh.topology.dim], dtype=float)

    left_u_block_dofs = np.where(np.isclose(u_block_coordinates[:, 0], 0.0))[0].astype(np.int32)
    left_u_dofs = expand_block_dofs(left_u_block_dofs, int(displacement_space.dofmap.index_map_bs))

    phi_fixed_mask = np.isclose(phi_coordinates[:, 0], 0.0) | (
        np.isclose(phi_coordinates[:, 0], 1.0) & np.isclose(phi_coordinates[:, 1], 0.0)
    )
    phi_fixed_dofs = np.where(phi_fixed_mask)[0].astype(np.int32)

    x_coords = phi_coordinates[:, 0]
    y_coords = phi_coordinates[:, 1]
    phi_exact = float(phi_bias) + 0.22 * y_coords + 0.05 * x_coords * (1.0 - x_coords)
    u_exact = np.zeros(layout.ndof_u, dtype=float)
    reference_state = np.concatenate([u_exact, phi_exact])

    boundary_conditions = BoundaryConditionContract(
        conditions=(
            BlockDirichletCondition(
                block="u",
                dofs=left_u_dofs,
                values=np.zeros(left_u_dofs.shape[0], dtype=float),
            ),
            BlockDirichletCondition(
                block="phi",
                dofs=phi_fixed_dofs,
                values=phi_exact[phi_fixed_dofs],
            ),
        )
    )

    set_function_values(displacement_function, u_exact)
    set_function_values(phi_function, phi_exact)
    exact_assembled = assemble_monolithic_dry_run(
        mesh,
        displacement_space,
        phi_space,
        displacement_function,
        phi_function,
        solid_params=solid_params,
        contact_law=PenaltyContactLaw(penalty=penalty),
        reinitialize_beta=reinitialize_beta,
        phi_target=phi_exact,
        contact_gap_offset=gap_offset,
        contact_backend=contact_backend,
        contact_query_point_fraction=contact_query_point_fraction,
        contact_query_point_normal_offset=contact_query_point_normal_offset,
        contact_slave_boundary="top",
        contact_master_boundary="bottom",
    )
    external_loads = (
        StructuralNodalLoad(
            u_dofs=np.arange(layout.ndof_u, dtype=np.int32),
            values=np.asarray(exact_assembled.R_u.getArray(), dtype=float).copy(),
        ),
    )

    initial_state = reference_state.copy()
    free_phi_dofs = np.setdiff1d(np.arange(layout.ndof_phi, dtype=np.int32), phi_fixed_dofs)
    initial_state[layout.phi_offset + free_phi_dofs] = phi_exact[free_phi_dofs] - 0.12

    callables = Dolfinx0p3ResidualJacobianCallables(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        displacement_function=displacement_function,
        phi_function=phi_function,
        layout=layout,
        solid_params=solid_params,
        contact_law=PenaltyContactLaw(penalty=penalty),
        reinitialize_beta=reinitialize_beta,
        phi_target=phi_exact,
        contact_gap_offset=gap_offset,
        contact_backend=contact_backend,
        contact_query_point_fraction=contact_query_point_fraction,
        contact_query_point_normal_offset=contact_query_point_normal_offset,
        contact_slave_boundary="top",
        contact_master_boundary="bottom",
        external_loads=external_loads,
        boundary_conditions=boundary_conditions,
    )
    return TransitionMonolithicProblem(
        mesh=mesh,
        displacement_space=displacement_space,
        phi_space=phi_space,
        callables=callables,
        layout=layout,
        boundary_conditions=boundary_conditions,
        external_loads=external_loads,
        initial_state=initial_state,
        reference_state=reference_state,
        metadata={
            "backend": "dolfinx0p3",
            "problem_kind": "contact_strip_mini_benchmark",
            "contact_backend": contact_backend,
            "contact_slave_boundary": "top",
            "contact_master_boundary": "bottom",
            "gap_offset": float(gap_offset),
            "penalty": float(penalty),
            "phi_bias": float(phi_bias),
        },
    )


__all__ = [
    "TransitionMonolithicProblem",
    "build_contact_strip_benchmark",
    "build_unit_square_toy_problem",
]
