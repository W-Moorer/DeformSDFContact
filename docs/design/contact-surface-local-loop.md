# Contact Surface Local Loop

## Scope

This document defines the minimal backend-agnostic surface quadrature executor
under `src/deformsdfcontact/contact/surface_local_loop.py`.

The executor takes a normalized `ContactSurfaceMapping`, evaluates the existing
point kernels at every surface quadrature point, and accumulates a local
contact residual and local tangent blocks.

## What This Layer Does

Implemented now:

- iterate over surface quadrature points
- convert each point contract into `ContactPointKernelInput`
- call `evaluate_contact_point_kernel(...)`
- accumulate:
  - `local_residual_u`
  - `local_K_uu`
  - `local_K_uphi`

This layer is still purely local and backend-independent.

## What This Layer Does Not Do

This layer does **not** implement:

- global matrix or vector insertion
- DOLFINx forms
- PETSc hooks
- solver control flow
- geometry evaluation
- contact search

## Relation To Previous Layer

The surface local loop consumes the normalized output of
`build_contact_surface_mapping(...)`.

It does not re-interpret or recompute:

- `g_n`
- `G_u`
- `G_a`
- `H_uu_g`
- `H_uphi_g`

All of those semantics remain frozen in the geometry and point-kernel layers.

## Relation To Future Assembly Layers

Future backend-specific form or assembly code may call this local executor, but
they must not rewrite the point-kernel formulas defined in the contact kernel
layer.

This local loop exists precisely to keep:

- backend-neutral local algebra
- backend-specific global insertion

separated cleanly.
