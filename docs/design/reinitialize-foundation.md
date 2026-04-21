# Reinitialize Foundation

## Scope

This document defines the minimal `sdf/reinitialize` foundation under
`src/deformsdfcontact/sdf/reinitialize.py`.

The current layer is intentionally restricted to:

- pointwise residual evaluation
- pointwise tangent evaluation
- element-local residual/tangent kernels for one scalar field

Everything here is pure `numpy` logic. No FE framework objects appear in this
layer.

## Purpose

The reinitialization layer supplies the local algebra needed by a later FE
assembly layer. It does not decide how quadrature points are chosen, how forms
are compiled, or how nonlinear solves are driven.

Its job is to make the core pull-back SDF reinitialization kernel explicit and
testable before any DOLFINx or PETSc business code is attached.

## Governing Local Formula

At one quadrature point, define:

- `g = grad(phi)`
- `A = C^{-1}` from the kinematics layer
- `q = g^T A g - 1`

The local residual against a scalar test function `eta` is:

`R(eta) = q * grad(eta)^T A g + beta * (phi - phi_target) * eta`

where:

- `phi_target` is an externally supplied anchor value
- `phi_target` may later come from `phi0`, `phi_pred`, or another upstream
  warm start
- `beta >= 0` is the local anchoring weight

For a trial variation `dphi`, the local tangent is:

`K(eta, dphi) = 2 * (grad(eta)^T A g) * (grad(dphi)^T A g)
              + q * grad(eta)^T A grad(dphi)
              + beta * eta * dphi`

This expression assumes the physical metric tensor `A = C^{-1}` is symmetric.

## Element-Local Kernel

For one scalar element with local basis values `N_a` and local basis gradients
`grad(N_a)`, the layer evaluates:

- `phi_h = sum_a N_a * phi_a`
- `grad(phi_h) = sum_a grad(N_a) * phi_a`

At one quadrature point, the local residual contribution is:

`r_a = w * [ q * s_a + beta * (phi_h - phi_target) * N_a ]`

with:

`s_a = grad(N_a)^T A grad(phi_h)`

The local tangent contribution is:

`K_ab = w * [ 2 * s_a * s_b + q * grad(N_a)^T A grad(N_b) + beta * N_a * N_b ]`

where `w` already includes the quadrature weight and geometric Jacobian factor.

## What This Layer Does

Implemented now:

- `eikonal_defect`
- `reinitialize_point_residual`
- `reinitialize_point_tangent`
- `reinitialize_element_residual_tangent`

These functions accept only pointwise tensors, local element vectors, basis
values, basis gradients, and scalar parameters.

## What This Layer Does Not Do

This layer does **not** implement:

- DOLFINx forms
- quadrature loops over a mesh
- contact
- predictor construction
- query-point search or ownership
- global residual/Jacobian assembly
- SNES/KSP/solver logic
- reinitialization stepping strategy

## Interface Boundary With Predictor

The predictor layer provides a theory-consistent warm start such as `phi_pred`.
The reinitialization layer does not build that predictor itself.

Instead, it accepts a generic `phi_target` input, which lets a later FE layer
choose whether to anchor toward:

- the reference distance `phi0`
- the metric-stretch predictor `phi_pred`
- some other externally prepared target field

## Test Strategy

The first tests remain independent from DOLFINx and PETSc.

They cover:

- zero defect / zero residual for an exact signed-distance state
- pointwise tangent consistency against finite differences
- zero element residual for an affine exact distance field
- element tangent consistency against finite differences

This keeps the local algebra stable before any form or solver integration
begins.
