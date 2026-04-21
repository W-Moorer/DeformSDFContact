# Contact Geometry Foundation

## Scope

This document defines the minimal backend-agnostic contact-geometry foundation
under `src/deformsdfcontact/contact/geometry.py`.

The current layer is restricted to the geometric quantities needed before any
contact force or contact tangent is introduced:

- reference query point `X_c`
- normal gap `g_n = phi(X_c)`
- local SDF sensitivity `G_a`
- local geometry sensitivity `G_u`

Everything in this layer is pure `numpy` pointwise logic.

## Minimal Analytic Setting

The foundation uses a deliberately simple 2D setup.

### Master Geometry

The reference master interface is the line:

`X = [xi, 0]`

The current master line is represented by an affine map:

`x_m(xi) = b + xi * t`

with:

- `b in R^2` the current line origin
- `t in R^2` the current line tangent direction

The local geometry parameter vector is:

`d = [b_x, b_y, t_x, t_y]`

### Query Point

For a given slave point `x_s`, the reference query point is defined by the
orthogonal projection onto the current master line:

`xi_c = ((x_s - b) . t) / (t . t)`

`X_c = [xi_c, 0]`

This makes `X_c` explicit, differentiable, and easy to verify by hand.

### Gap

The normal gap is defined exactly as:

`g_n(d, a) = phi(X_c(d))`

where `phi` is a scalar field evaluated in reference coordinates.

The current foundation uses an analytic pointwise field interface. A minimal
affine field implementation is provided for testing:

`phi(X) = c + g . X`

## Local First-Order Sensitivities

The linearized gap is written as:

`delta g_n = G_u delta d + G_a delta a`

### Local SDF Sensitivity

For this stage, the SDF sensitivity is frozen in the smallest useful form:

`G_a = N_phi(X_c)`

That is, `G_a` is simply the vector of local shape values used to interpolate
the SDF field at the query point.

This layer does not know anything about global DoF numbering or assembly.

### Local Geometry Sensitivity

For the affine master map, the geometry sensitivity is:

`G_u = grad(phi)(X_c)^T * dX_c/dd`

with:

`dX_c/dd in R^(2 x 4)`

Because `X_c = [xi_c, 0]`, only the derivative of `xi_c` is nonzero.

This is the smallest local geometry sensitivity that can be checked by finite
differences without introducing FE displacement fields or global vectors.

## What This Layer Does

Implemented now:

- a 2D affine master map
- reference query-point evaluation
- gap evaluation
- local `G_u`
- local `G_a`
- a structured result object for downstream contact kernels

## What This Layer Does Not Do

This layer does **not** implement:

- contact force
- penalty or augmented Lagrangian laws
- `K^c_{uu}`
- `K^c_{uphi}`
- FE assembly
- DOLFINx forms
- global residual/Jacobian insertion
- solver logic
- friction

## Relation To Later Layers

Later contact local kernels should consume the structured output of this layer:

- `X_c`
- `g_n`
- `G_u`
- `G_a`

Later forms or assembly layers may call this geometry layer, but they must not
push backend-specific concepts back into it.

This foundation is intended to freeze the semantics of the geometric quantities
before any contact constitutive choice or backend integration begins.
