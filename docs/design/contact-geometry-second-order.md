# Contact Geometry Second Order

## Scope

This document defines the minimal backend-agnostic second-order contact
geometry layer under `src/deformsdfcontact/contact/geometry_second_order.py`.

The current layer freezes the geometry-only second-order objects that later
contact local kernels may consume:

- `E = dX_c / dd`
- `H_uu^(g)` for the gap geometry
- `H_uphi^(g)` for mixed geometry-SDF interpolation effects

Everything here is pure `numpy` pointwise logic.

## Minimal Analytic Model

The layer intentionally reuses the same 2D analytic master geometry as the
first-order foundation:

- reference interface: `X = [xi, 0]`
- current master line: `x_m(xi) = b + xi * t`
- geometry parameters: `d = [b_x, b_y, t_x, t_y]`

The reference query point remains:

`X_c(d) = [xi_c(d), 0]`

with:

`xi_c = ((x_s - b) . t) / (t . t)`

This model is simple enough to differentiate by hand while still retaining a
nontrivial second derivative because `xi_c(d)` depends rationally on `t`.

## Why A Quadratic Phi Field Is Needed

If `phi` is only affine, then its Hessian vanishes and the core curvature term
cannot be exercised.

To avoid freezing a misleadingly trivial second-order interface, this layer
adds a minimal quadratic field:

`phi(X) = offset + linear . X + 0.5 * X^T H X`

with:

- `gradient(phi)(X) = linear + H X`
- `hessian(phi)(X) = H`

An affine field remains useful as a degenerate test case with zero Hessian.

## Second-Order Objects

### Query Sensitivity

The first-order query sensitivity is:

`E = dX_c / dd`

In the current model, `E` has shape `(2, 4)`.

The second derivative of the query map is also exposed:

`d2Xc_dd2`

with shape `(2, 4, 4)`.

Only the first reference coordinate varies in the current model, so the second
row of both `E` and `d2Xc_dd2` is structurally zero.

### Gap Geometry Hessian

The full gap Hessian in the current model is split into:

`H_uu^(g) = H_uu,curvature^(g) + H_uu,query^(g)`

where:

`H_uu,curvature^(g) = E^T H_phi E`

and:

`H_uu,query^(g) = sum_i grad(phi)_i * d2Xc_i/dd2`

The curvature term is the minimum required second-order geometry object.

The query-acceleration term is also retained because, even with an affine
master line, the projection map `X_c(d)` is nonlinear in `d`.

## Mixed Geometry-SDF Object

The minimal mixed second-order geometry object is frozen as:

`H_uphi^(g) = E^T B_phi(X_c)`

where:

- `B_phi(X_c)` is a local shape-gradient matrix
- in the current interface, `B_phi(X_c)` has shape `(2, nphi_local)`
- `H_uphi^(g)` therefore has shape `(4, nphi_local)`

This object is purely local. It does not know anything about global DoF
numbering or assembly.

## What This Layer Does

Implemented now:

- `QuadraticPhiField2D`
- `query_sensitivity_second_order`
- `second_order_gap_geometry`
- `evaluate_contact_second_order_geometry`

The output freezes the second-order geometry semantics for later contact local
kernels.

## What This Layer Does Not Do

This layer does **not** implement:

- contact residual
- penalty or augmented Lagrangian laws
- `K^c_uu`
- `K^c_uphi`
- local contact force kernels
- DOLFINx forms
- global assembly
- solver logic

## Current Model Boundaries

The current model is intentionally minimal and therefore simplifies several
general-theory effects.

Simplified or absent due to the chosen model:

- no master-surface curvature beyond the projection nonlinearity of `xi_c(d)`
- no non-affine master mapping
- no geometry-induced variation of the supplied local shape-gradient matrix
  `B_phi(X_c)`; the current layer freezes `H_uphi^(g)` as `E^T B_phi(X_c)`
- no contact constitutive terms of any kind

These simplifications are properties of the current analytic model, not general
theoretical statements about contact.

## Relation To Later Layers

Later contact local kernels may consume:

- `X_c`
- `E`
- `H_uu^(g)`
- `H_uphi^(g)`

Later form or assembly layers may call this second-order geometry layer, but
they must not push backend-specific concepts back into it.
