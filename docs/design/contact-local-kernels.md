# Contact Local Kernels

## Scope

This document defines the minimal backend-agnostic contact local-kernel layer
under:

- `src/deformsdfcontact/contact/laws.py`
- `src/deformsdfcontact/contact/kernels.py`
- `src/deformsdfcontact/contact/local_loop.py`

The current layer is restricted to:

- a minimal normal-contact law abstraction
- a penalty normal-contact law
- point-level contact residual evaluation
- point-level consistent tangents
- a local loop that sums point contributions

Everything here is pure `numpy` local algebra.

## What This Layer Does

Implemented now:

- `ContactLaw` protocol
- `PenaltyContactLaw`
- point residual `r_u_c`
- point tangents `K_uu_c`, `K_uphi_c`
- local accumulation over multiple point inputs

This layer consumes already-frozen geometric quantities and turns them into
local algebraic contact contributions.

## What This Layer Does Not Do

This layer does **not** implement:

- query-point geometry
- contact search
- contact kinematics
- penalty or AL outer update loops
- DOLFINx forms
- global assembly
- PETSc objects
- solver logic
- friction

## Inputs From Geometry Layers

The contact point kernels consume only the previously frozen geometry objects:

- `g_n`
- `G_u`
- `G_a`
- `H_uu_g`
- `H_uphi_g`

The kernel layer must not recompute query points or geometric sensitivities.

## Penalty Law Semantics

The minimal penalty law is active only in penetration:

- if `g_n < 0`, contact is active
- if `g_n >= 0`, contact is inactive

The compressive multiplier is:

`lambda_n = penalty * (-g_n)` for `g_n < 0`

and zero otherwise.

The returned scalar `k_n` is the active-set penalty stiffness with respect to
gap closure `(-g_n)`, so:

- `k_n = penalty` in the active set
- `k_n = 0` otherwise

This keeps `lambda_n` and `k_n` both nonnegative.

## Point Kernel Formulas

The current point kernel uses the frozen local formulas:

- `r_u_c = lambda_n * G_u * weight`
- `K_uu_c = (k_n * outer(G_u, G_u) + lambda_n * H_uu_g) * weight`
- `K_uphi_c = (k_n * outer(G_u, G_a) + lambda_n * H_uphi_g) * weight`

Shape conventions:

- `G_u`: `(ndof_u_local,)`
- `G_a`: `(nphi_local,)`
- `H_uu_g`: `(ndof_u_local, ndof_u_local)`
- `H_uphi_g`: `(ndof_u_local, nphi_local)`

These are purely local objects. No global numbering appears here.

## Local Loop

The local loop is intentionally simple:

- evaluate every point input with the chosen law
- sum `r_u_c`
- sum `K_uu_c`
- sum `K_uphi_c`

No sparse insertion, form compilation, or solver hook exists in this layer.

## Relation To Previous Layer

This layer only consumes geometry results. It does not modify the meaning of:

- `g_n`
- `G_u`
- `G_a`
- `H_uu_g`
- `H_uphi_g`

Geometry semantics remain frozen in the dedicated contact geometry layers.

## Relation To Later Layers

Future form or assembly layers may call these local kernels, but they must not
rewrite the local formulas defined here.

This is the contract boundary between:

- pure geometry
- pure local contact algebra
- future backend-specific assembly code

## Test Strategy

The tests cover:

- active and inactive penalty law behavior
- exact point residual and tangent formulas
- explicit use of second-order geometry terms
- finite-difference checks of `K_uu_c` and `K_uphi_c`
- correct accumulation in the local loop

All tests stay independent from DOLFINx, PETSc, and any global assembly path.
