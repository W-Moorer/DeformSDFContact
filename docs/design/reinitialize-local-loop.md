# Reinitialize Local Loop

## Scope

This document defines the minimal backend-agnostic quadrature executor under
`src/deformsdfcontact/sdf/local_loop.py`.

The executor is intentionally restricted to:

- consuming `ReinitializeElementMapping`
- iterating over element-local quadrature points
- calling the existing point residual and tangent kernels
- accumulating one element residual vector and one element tangent matrix

This layer does not introduce any FE backend objects.

## Position In The Stack

The current reinitialization stack is:

1. `sdf/reinitialize.py`: point kernels and one-point element formula
2. `sdf/form_mapping.py`: assembly-neutral mapping contract
3. `sdf/local_loop.py`: quadrature executor defined here

The `local_loop` layer is the first place where multiple quadrature points are
accumulated, but it is still entirely local to one scalar element.

## Purpose

This executor gives the project a backend-neutral reference for how an FE
backend should drive the local reinitialization kernels.

It exists so the following responsibilities stay separated:

- backend-specific tabulation and field extraction
- local kernel algebra
- global residual/Jacobian assembly

## Input Contract

The executor accepts exactly one `ReinitializeElementMapping`.

Each quadrature-point record inside that mapping must provide:

- `phi`
- `grad_phi`
- `phi_target`
- `shape_values`
- `shape_gradients`
- `A`
- `weight`
- `beta`

The executor does not compute these quantities itself.

## Loop Semantics

For every quadrature point `q` and every local basis pair `(a, b)`, the
executor evaluates:

- residual entry `r_a += R_q(N_a)`
- tangent entry `K_ab += K_q(N_a, N_b)`

by calling the existing point kernels:

- `reinitialize_point_residual(...)`
- `reinitialize_point_tangent(...)`

This is intentional. The loop layer should compose existing kernels, not
re-derive or duplicate their formulas.

## Output

The executor returns:

- element residual with shape `(nshape,)`
- element tangent with shape `(nshape, nshape)`

These are purely local arrays and carry no global indexing information.

## Validation

Although `build_reinitialize_element_mapping(...)` already validates the common
array path, the executor performs a small amount of defensive checking because
future backend-specific adapters may construct `ReinitializeElementMapping`
directly.

The executor checks:

- non-empty quadrature data
- consistent `nshape`
- consistent spatial dimension
- shape agreement for `shape_values`, `shape_gradients`, `grad_phi`, and `A`
- nonnegative `beta`

## What This Layer Does Not Do

This layer does **not** implement:

- DOLFINx forms
- UFL expressions
- global assembly
- PETSc vectors or matrices
- contact terms
- solver control flow

It is only a local quadrature accumulator.

## Test Strategy

The local loop is tested for:

- equivalence to the one-point element kernel for a single quadrature point
- correct summation over multiple quadrature points
- explicit use of the point kernels
- rejection of inconsistent mapping data

These tests remain pure `numpy` and backend-independent.
