# Prototype External Loads And Boundary Conditions

## Scope

This document defines the minimal external-load and boundary-condition
contracts used by the first solver-capable monolithic prototype.

The current layer is intentionally narrow. It only freezes:

- how structural external loads enter `R_u`
- how prototype Dirichlet conditions are represented on the `u` and `phi`
  blocks
- how strong constraints are imposed on the assembled residual and Jacobian

This is a prototype-level contract, not a production boundary-condition system.

## Structural External Load Contract

The current prototype supports only nodal loads on the structural block.

`StructuralNodalLoad` stores:

- block-local structural dofs `u_dofs`
- load values on those dofs

The assembled structural residual keeps the intended sign convention:

`R_u = R_u_int - R_u_ext + R_u_c`

The current implementation accumulates a global structural load vector and
subtracts it from:

- the standalone structural residual block
- the monolithic residual

The external load does not contribute to the Jacobian in the current prototype.

## Dirichlet Contract

The prototype uses block-aware Dirichlet conditions:

- `block = "u"` for structural dofs
- `block = "phi"` for SDF dofs

Each `BlockDirichletCondition` stores:

- block-local dof ids
- prescribed values on those dofs

The block-local numbering is important:

- `u` dofs are numbered inside the structural block
- `phi` dofs are numbered inside the SDF block

Global monolithic indices are obtained only through the frozen
`MonolithicBlockLayout`.

## Strong Imposition Strategy

The prototype uses a simple, solver-friendly strong imposition strategy:

1. clamp the assembled state used for local evaluation to the prescribed values
2. assemble the unconstrained residual and Jacobian using that clamped state
3. overwrite constrained residual rows with `x_i - x_i^D`
4. zero constrained Jacobian rows and columns and insert unit diagonals

This is intentionally minimal. It is sufficient for:

- the first solver-capable monolithic prototype
- residual/Jacobian callables
- PETSc SNES integration

It is not meant to be the final production boundary-condition path.

## Current Prototype Policy

For the current toy problem:

- structural Dirichlet conditions fix part of the displacement field to remove
  rigid-body modes
- SDF Dirichlet conditions fix only a small subset of scalar dofs to remove the
  prototype nullspace
- the remaining free dofs are solved monolithically

The current `phi` constraints are therefore prototype stabilization choices,
not the final research boundary treatment.
