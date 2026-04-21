# PETSc SNES Prototype Path

## Scope

This document defines the first solver-capable monolithic prototype path.

The implementation lives in:

- `src/deformsdfcontact/solvers/petsc_snes.py`

## Goal

The current goal is deliberately narrow:

- connect the monolithic residual callable to `PETSc.SNES`
- connect the monolithic Jacobian callable to `PETSc.SNES`
- solve one very small transition toy problem

This stage stops once the solver path is verified.

## Current Solver Policy

The prototype uses a minimal PETSc configuration:

- `SNES` type: `newtonls`
- `KSP` type: `preonly`
- `PC` type: `lu`

No attempt is made yet to optimize:

- fieldsplit
- Schur complements
- Krylov behavior
- preconditioning strategy

## Callback Semantics

The solver callback layer only:

- asks the transition problem for `R(x)`
- asks the transition problem for `J(x)`
- copies the assembled objects into the PETSc work vectors/matrices

The callback layer does **not** redefine the residual or Jacobian semantics.

## Current Status

This is a prototype solver path for the current transition environment:

- PETSc 3.15.5
- petsc4py 3.15.1
- DOLFINx 0.3.0

It is not the final solver architecture and it should not be mistaken for the
production monolithic path.
