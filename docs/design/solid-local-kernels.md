# Solid Local Kernels

## Scope

This document defines the minimal backend-agnostic solid local chain under
`src/deformsdfcontact/solid`.

The current implementation is intentionally restricted to:

- 2D small-strain linear elasticity
- point-level internal-force and tangent kernels
- an element-local form-mapping layer
- an element-local quadrature loop

This is sufficient for the first assembled monolithic dry run.

## Model Choice

The solid chain currently uses isotropic small-strain linear elasticity in
plane strain.

This choice is deliberate:

- it is easy to verify by hand
- the tangent is constant
- it closes the structure-side assembly chain quickly

The tensor-level `materials` foundation remains available and unchanged. The
current solid local chain is a minimal numeric assembly chain, not a replacement
for the broader constitutive foundation.

## Point Kernel

At one quadrature point, the solid point kernel consumes:

- engineering strain vector `eps` with shape `(3,)`
- strain-displacement matrix `B` with shape `(3, ndof_u_local)`
- quadrature weight

and returns:

- `r_u_int = B^T sigma * weight`
- `K_uu_int = B^T C B * weight`

with:

- `sigma = C eps`
- `C` the plane-strain isotropic tangent

## Local Mapping And Loop

The form-mapping layer normalizes:

- `u_local`
- one or more `B` operators
- one or more quadrature weights

The local loop:

- evaluates the point kernel at each quadrature point
- sums the residual contributions
- sums the tangent contributions

## What This Layer Does Not Do

This layer does **not** implement:

- nonlinear constitutive updates
- body-force or traction assembly contracts
- DOLFINx forms
- global assembly
- solver logic

For the current monolithic dry run, `R_u_ext` may simply be zero.
