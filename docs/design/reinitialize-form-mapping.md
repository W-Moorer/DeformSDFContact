# Reinitialize Form Mapping

## Scope

This document defines the minimal assembly-neutral adapter contract under
`src/deformsdfcontact/sdf/form_mapping.py`.

The purpose of this layer is to sit between:

- the local reinitialization kernels in `sdf/reinitialize.py`
- a future FE backend that knows about element tables, quadrature points, and
  field coefficients

This layer remains completely independent from DOLFINx, UFL, PETSc, contact,
and global assembly.

## Role In The Stack

The current stack is:

1. `sdf/reference`
2. `sdf/predictor`
3. `sdf/reinitialize`
4. `sdf/form_mapping`  ← this document

The `reinitialize` layer defines the local residual and tangent algebra.
The `form_mapping` layer defines how a backend must package element-local data
so those kernels can be evaluated consistently.

## What This Layer Does

Implemented now:

- a backend-neutral quadrature-point contract
- a backend-neutral element mapping contract
- a small array-based reference builder that validates shapes and normalizes
  scalar-vs-batched inputs

This is not a form compiler. It is a contract and validation layer.

## What This Layer Does Not Do

This layer does **not** implement:

- DOLFINx forms
- UFL expressions
- global residual or Jacobian assembly
- quadrature loops over a mesh
- contact terms
- solver hooks
- SNES/KSP business logic

## Contract Objects

At one quadrature point, the future FE backend must be able to provide:

- `phi`
- `grad_phi`
- `phi_target`
- `shape_values`
- `shape_gradients`
- `A`
- `weight`
- `beta`

The reference contract represents this with:

- `ReinitializeQuadraturePointData`
- `ReinitializeElementMapping`

`ReinitializeElementMapping` is simply a collection of quadrature-point records
for one scalar element.

## Supported Input Shapes

The reference builder accepts:

- `phi_local`: `(nshape,)`
- `shape_values`: `(nqp, nshape)`
- `shape_gradients`: `(nqp, nshape, dim)`
- `A`: `(dim, dim)` or `(nqp, dim, dim)`
- `weights`: `(nqp,)`
- `phi_target`: scalar, `(nshape,)`, or `(nqp,)`
- `beta`: scalar or `(nqp,)`

This is intentionally explicit. Hidden broadcasting is avoided except for the
two cases that are useful and unambiguous:

- constant `A` broadcast to all quadrature points
- constant `beta` broadcast to all quadrature points

## Beta Contract

`beta` is the anchoring weight for the regularization term.

The contract is:

- `beta = 0` is valid and means the adapter feeds a pure eikonal kernel
- `beta > 0` is valid
- `beta < 0` is invalid
- scalar `beta` is broadcast to every quadrature point
- `(nqp,)` beta is interpreted pointwise

The adapter contract does not decide how `beta` is chosen. It only defines how
that choice is transported into the local kernel inputs.

## Phi-Target Contract

`phi_target` is intentionally flexible because future backends may source it
from:

- a constant anchor
- a local FE coefficient vector
- a pre-evaluated quadrature field

The contract therefore accepts:

- scalar `phi_target`
- local `(nshape,)` coefficients, which are interpolated by `shape_values`
- quadrature values `(nqp,)`

## Reference Builder

`build_reinitialize_element_mapping(...)` is a small array-based reference
adapter.

Its purpose is to:

- validate shapes
- normalize scalar-vs-batched parameters
- compute `phi` and `grad_phi` at each quadrature point
- package the result into contract objects

It exists so this contract can be tested before any FE backend is attached.

## Future FE Backends

A future backend-specific adapter should satisfy the `ReinitializeFormAdapter`
protocol and produce the same contract objects.

For example, a DOLFINx-specific implementation could later:

- read local DoF vectors
- tabulate basis values and gradients at quadrature points
- evaluate or fetch `A`
- emit `ReinitializeElementMapping`

without changing the pure local kernel algebra in `sdf/reinitialize.py`.

## Test Strategy

This layer is tested only for:

- `beta` broadcasting and pointwise behavior
- rejection of negative `beta`
- shape-contract enforcement
- correct local interpolation of `phi`, `grad_phi`, and `phi_target`

Nothing in these tests depends on DOLFINx, PETSc, or any global assembly path.
