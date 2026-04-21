"""Assembly-neutral contracts and block-layout semantics."""

from .conditions import (
    BlockDirichletCondition,
    BoundaryConditionContract,
    StructuralNodalLoad,
    accumulate_structural_nodal_loads,
    apply_dirichlet_to_jacobian,
    apply_dirichlet_to_residual,
    apply_dirichlet_to_residual_and_jacobian,
    apply_dirichlet_values_to_state,
    dirichlet_global_dofs_and_values,
)
from .contracts import (
    AssemblyPlan,
    ContactLocalContribution,
    MonolithicBlockLayout,
    SDFLocalContribution,
    SolidLocalContribution,
)

__all__ = [
    "AssemblyPlan",
    "BlockDirichletCondition",
    "BoundaryConditionContract",
    "ContactLocalContribution",
    "MonolithicBlockLayout",
    "SDFLocalContribution",
    "SolidLocalContribution",
    "StructuralNodalLoad",
    "accumulate_structural_nodal_loads",
    "apply_dirichlet_to_jacobian",
    "apply_dirichlet_to_residual",
    "apply_dirichlet_to_residual_and_jacobian",
    "apply_dirichlet_values_to_state",
    "dirichlet_global_dofs_and_values",
]
