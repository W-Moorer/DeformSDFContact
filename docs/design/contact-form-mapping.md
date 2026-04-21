# Contact Form Mapping

## Scope

This document defines the minimal backend-agnostic contract layer under
`src/deformsdfcontact/contact/form_mapping.py`.

The role of this layer is to normalize local surface quadrature data into a
stable contract that can be consumed by the contact point kernels and the
surface local quadrature executor.

## What This Layer Does

Implemented now:

- a normalized surface quadrature-point contract
- a normalized surface mapping object
- explicit broadcasting rules for constant local quantities
- shape validation for local contact arrays

This layer is purely local and purely array-based.

## What This Layer Does Not Do

This layer does **not** implement:

- query-point geometry
- contact law evaluation
- point kernel formulas
- DOLFINx forms
- global assembly
- solver logic

## Contract Objects

At one surface quadrature point, the contact kernel needs exactly:

- `g_n`
- `G_u`
- `G_a`
- `H_uu_g`
- `H_uphi_g`
- `weight`

The mapping layer packages these into:

- `ContactQuadraturePointData`
- `ContactSurfaceMapping`

The mapping object is just a collection of normalized quadrature-point records
for one local contact patch.

## Supported Input Shapes

The reference builder accepts:

- `g_n`: scalar or `(nqp,)`
- `G_u`: `(ndof_u_local,)` or `(nqp, ndof_u_local)`
- `G_a`: `(nphi_local,)` or `(nqp, nphi_local)`
- `H_uu_g`: `(ndof_u_local, ndof_u_local)` or `(nqp, ndof_u_local, ndof_u_local)`
- `H_uphi_g`: `(ndof_u_local, nphi_local)` or `(nqp, ndof_u_local, nphi_local)`
- `weights`: `(nqp,)`

The `weights` array defines `nqp`.

## Broadcasting Rules

The builder allows the following constant quantities to be broadcast to every
quadrature point:

- scalar `g_n`
- single `G_u`
- single `G_a`
- single `H_uu_g`
- single `H_uphi_g`

Per-point arrays with leading dimension `nqp` are preserved pointwise.

No hidden broadcasting beyond these cases is allowed.

## Shape Contract

The builder validates:

- every point has the same `ndof_u_local`
- every point has the same `nphi_local`
- `H_uu_g` is square in the `u` block
- `H_uphi_g` matches the `(u, phi)` local block size
- `weights` are one scalar per quadrature point

If any of these contracts fail, the builder raises an explicit `ValueError`.

## Relation To Neighboring Layers

This layer consumes only the frozen subset of geometry-layer outputs that the
contact point kernels need.

It exists strictly to serve:

- `evaluate_contact_point_kernel(...)`
- `execute_contact_surface_local_loop(...)`

It must not change the meaning of the point-kernel formulas.
