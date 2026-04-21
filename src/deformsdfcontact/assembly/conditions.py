"""Prototype external-load and boundary-condition contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from petsc4py import PETSc

from .contracts import MonolithicBlockLayout


def _as_index_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    return array


def _as_vector(values: object, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _normalized_block_name(block: object) -> str:
    name = str(block)
    if name not in {"u", "phi"}:
        raise ValueError(f"block must be 'u' or 'phi', got {name!r}")
    return name


def _state_array(state: object, length: int) -> np.ndarray:
    if hasattr(state, "getArray"):
        array = np.asarray(state.getArray(), dtype=float)
    else:
        array = np.asarray(state, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"state must have shape ({length},), got {array.shape!r}")
    return array


@dataclass(frozen=True)
class StructuralNodalLoad:
    """Prototype structural nodal load on block-local `u` dofs."""

    u_dofs: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        u_dofs = _as_index_vector(self.u_dofs, "u_dofs")
        values = _as_vector(self.values, "values", u_dofs.shape[0])
        object.__setattr__(self, "u_dofs", u_dofs)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class BlockDirichletCondition:
    """Prototype block-aware Dirichlet condition on `u` or `phi` dofs."""

    block: str
    dofs: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        block = _normalized_block_name(self.block)
        dofs = _as_index_vector(self.dofs, "dofs")
        values = _as_vector(self.values, "values", dofs.shape[0])
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "values", values)

    def global_dofs(self, layout: MonolithicBlockLayout) -> np.ndarray:
        """Return constrained dofs in monolithic global numbering."""

        if self.block == "u":
            return self.dofs.copy()
        return layout.lift_phi_dofs(self.dofs)


@dataclass(frozen=True)
class BoundaryConditionContract:
    """Prototype collection of strong Dirichlet conditions."""

    conditions: tuple[BlockDirichletCondition, ...] = ()

    def __iter__(self):
        return iter(self.conditions)

    def __len__(self) -> int:
        return len(self.conditions)


def accumulate_structural_nodal_loads(
    ndof_u: int,
    loads: object = (),
) -> np.ndarray:
    """Accumulate prototype structural nodal loads into one block vector."""

    vector = np.zeros(int(ndof_u), dtype=float)
    for load in tuple(loads):
        if np.any(load.u_dofs < 0) or np.any(load.u_dofs >= ndof_u):
            raise ValueError("structural load dofs must lie inside the structural block")
        vector[load.u_dofs] += load.values
    return vector


def dirichlet_global_dofs_and_values(
    layout: MonolithicBlockLayout,
    boundary_conditions: BoundaryConditionContract | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique global constrained dofs and values."""

    if boundary_conditions is None or len(boundary_conditions) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=float)

    mapping: dict[int, float] = {}
    for condition in boundary_conditions:
        global_dofs = condition.global_dofs(layout)
        for dof, value in zip(global_dofs, condition.values):
            key = int(dof)
            if key in mapping and not np.isclose(mapping[key], value):
                raise ValueError(f"conflicting Dirichlet values prescribed on dof {key}")
            mapping[key] = float(value)

    ordered_dofs = np.array(sorted(mapping.keys()), dtype=np.int32)
    ordered_values = np.array([mapping[int(dof)] for dof in ordered_dofs], dtype=float)
    return ordered_dofs, ordered_values


def apply_dirichlet_values_to_state(
    layout: MonolithicBlockLayout,
    state: object,
    boundary_conditions: BoundaryConditionContract | None,
) -> np.ndarray:
    """Return a clamped state array honoring the prototype Dirichlet data."""

    clamped = _state_array(state, layout.total_dofs).copy()
    global_dofs, values = dirichlet_global_dofs_and_values(layout, boundary_conditions)
    if global_dofs.size > 0:
        clamped[global_dofs] = values
    return clamped


def apply_dirichlet_to_residual(
    layout: MonolithicBlockLayout,
    state: object,
    residual: PETSc.Vec,
    boundary_conditions: BoundaryConditionContract | None,
) -> None:
    """Overwrite constrained residual rows with the prototype strong form."""

    global_dofs, values = dirichlet_global_dofs_and_values(layout, boundary_conditions)
    if global_dofs.size == 0:
        return
    state_array = _state_array(state, layout.total_dofs)
    residual.setValues(global_dofs, state_array[global_dofs] - values, addv=PETSc.InsertMode.INSERT_VALUES)
    residual.assemblyBegin()
    residual.assemblyEnd()


def apply_dirichlet_to_jacobian(
    layout: MonolithicBlockLayout,
    jacobian: PETSc.Mat,
    boundary_conditions: BoundaryConditionContract | None,
) -> None:
    """Zero constrained rows and columns and insert unit diagonals."""

    global_dofs, _ = dirichlet_global_dofs_and_values(layout, boundary_conditions)
    if global_dofs.size == 0:
        return
    jacobian.zeroRowsColumns(global_dofs, diag=1.0)
    jacobian.assemblyBegin()
    jacobian.assemblyEnd()


def apply_dirichlet_to_residual_and_jacobian(
    layout: MonolithicBlockLayout,
    state: object,
    residual: PETSc.Vec,
    jacobian: PETSc.Mat,
    boundary_conditions: BoundaryConditionContract | None,
) -> None:
    """Apply the prototype strong Dirichlet strategy to one residual/Jacobian pair."""

    apply_dirichlet_to_residual(layout, state, residual, boundary_conditions)
    apply_dirichlet_to_jacobian(layout, jacobian, boundary_conditions)


__all__ = [
    "BlockDirichletCondition",
    "BoundaryConditionContract",
    "StructuralNodalLoad",
    "accumulate_structural_nodal_loads",
    "apply_dirichlet_to_jacobian",
    "apply_dirichlet_to_residual",
    "apply_dirichlet_to_residual_and_jacobian",
    "apply_dirichlet_values_to_state",
    "dirichlet_global_dofs_and_values",
]
