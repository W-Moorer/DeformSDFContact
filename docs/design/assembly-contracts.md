# Monolithic Assembly Contracts

## Scope

This document defines the assembly-neutral contracts and block-layout semantics
used to move from the existing local chains toward an assembled monolithic dry
run.

The current layer is intentionally limited to:

- local contribution contracts
- monolithic block layout semantics
- assembly planning metadata

No backend-specific mesh API, form API, or solver logic belongs here.

## Monolithic Unknowns

The global unknown remains block-structured:

`R = [R_u, R_phi]^T`

`K = [[K_uu, K_uphi], [K_phiu, K_phiphi]]`

with:

- `u`: structural displacement unknowns
- `phi`: pull-back SDF unknowns

## Block Layout

`MonolithicBlockLayout` freezes the global ordering semantics:

- all `u` dofs come first
- all `phi` dofs come second

The layout object provides:

- `ndof_u`
- `ndof_phi`
- `total_dofs`
- global block shapes
- the global offset of the `phi` block

This semantics is backend-independent. A backend may assemble into a flat AIJ
matrix, a nested matrix, or another PETSc-compatible structure, but it should
not change the block meaning.

## Local Contribution Contracts

The local chains now produce three assembly-neutral contribution types:

- `SolidLocalContribution`
- `SDFLocalContribution`
- `ContactLocalContribution`

These objects freeze which local blocks each subsystem is allowed to write.

### Solid

The solid chain writes only:

- `R_u`
- `K_uu`

### SDF

The SDF chain writes only:

- `R_phi`
- `K_phiu`
- `K_phiphi`

### Contact

The contact chain writes only:

- `R_u`
- `K_uu`
- `K_uphi`

This makes the monolithic block semantics explicit without mixing subsystem
formulas together.

## Assembly Plan

`AssemblyPlan` is minimal metadata that ties:

- a `MonolithicBlockLayout`
- a backend identifier
- a short note about the current adapter

The current backend-specific adapter is a transition implementation targeting
DOLFINx 0.3.0 + PETSc. The contracts themselves are not tied to that version.

## What This Layer Does Not Do

This layer does **not** implement:

- local kernel formulas
- quadrature logic
- DOLFINx forms
- PETSc solver control
- global sparse insertion rules

It only freezes the data contracts used by later adapters.
