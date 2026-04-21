# Monolithic Residual And Jacobian Callables

## Scope

This document defines the solver-neutral residual and Jacobian callables used
between the assembled dry run and the first solver-capable monolithic
prototype.

The current implementation lives under:

- `src/deformsdfcontact/backend/dolfinx0p3/callables.py`
- `src/deformsdfcontact/backend/dolfinx0p3/problem.py`

## Input State

The callable interface consumes the monolithic state:

`x = [u, phi]^T`

with the global block order frozen by `MonolithicBlockLayout`:

- structural scalar dofs first
- SDF scalar dofs second

## Callable Semantics

The current transition callables provide:

- `assemble_residual(x) -> PETSc.Vec`
- `assemble_jacobian(x) -> PETSc.Mat`
- `assemble_system(x) -> assembled residual/Jacobian pair`

The callable layer is responsible for:

- splitting the monolithic state into `u` and `phi`
- clamping the working state to the prototype Dirichlet values
- updating the DOLFINx 0.3.0 transition functions
- calling the existing local assembly chain
- applying strong Dirichlet enforcement to the assembled monolithic objects

The callable layer is **not** responsible for:

- SNES iteration control
- nonlinear solve policy
- performance tuning

## Transition Contact Assumption

The current residual and Jacobian callables still use the transition contact
adapter. The adapter remains midpoint-based and does not implement production
contact search or ownership logic.

For the solver-capable prototype, it is only required that:

- the residual depends on the current state in a way consistent with the local
  transition Jacobian blocks
- the block positions remain correct
- the toy problem can be solved monolithically

The current callable path therefore enforces consistency first and realism
second. The midpoint contact closure is still a transition approximation.

This is a solver path for the transition backend, not the final contact
backend.
