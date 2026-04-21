# Prototype Query-Point Contact Backend

## Scope

This document defines the current prototype contact backend that is more
geometrically meaningful than the existing transition midpoint adapter while
remaining compatible with the same monolithic solver path.

The implementation lives in:

- `src/deformsdfcontact/backend/dolfinx0p3/contact_query_backend.py`

This is still a transition-environment backend for DOLFINx 0.3.0. It is **not**
a production contact implementation.

## What This Backend Does

For each boundary facet, the backend currently:

- builds a local facet frame from the reference facet tangent/normal
- places one fixed prototype slave point in that local frame
- projects that slave point onto the deformed facet line to obtain a query point
- evaluates the local interpolated `phi(X_c)` at that projected reference point
- forms the local closure measure
  `c = phi(X_c) - gap_offset`
  so that the existing penalty/contact kernel contract is preserved
- computes local contact inputs
  - `g_n = gap_offset - phi(X_c)`
  - `G_u = d c / d u_local`
  - `G_a = d c / d a_local`
  - `H_uu^(g) = d^2 c / d u_local^2`
  - `H_uphi^(g) = d^2 c / d u_local d a_local`
- passes those inputs into the existing backend-agnostic contact local kernels

## Current Prototype Simplifications

The current query-point backend is intentionally limited:

- one fixed slave point per boundary facet
- no global contact search
- no ownership/distributed query management
- no segment clipping or active-set smoothing
- no friction
- no augmented-Lagrangian outer loop

The second-order geometry terms used by this backend are currently evaluated by
small local finite differences of the scalar gap function. This is a prototype
choice made to keep the assembled solver path consistent before introducing a
heavier production geometry backend.

## Relationship to the Baseline Transition Backend

The baseline transition backend:

- evaluates contact from a midpoint-only facet surrogate
- does not explicitly construct a projected query point

The query-point backend:

- uses a projected query-point geometry
- reuses the same contact laws, point kernels, and surface local loop
- plugs into the same monolithic residual/Jacobian callable path

This allows the current prototype solver path to compare:

- `contact_backend="transition"`
- `contact_backend="query_point"`

without changing the global solver interface.

## What This Backend Does Not Do

This backend does **not** implement:

- production query-point search
- robust contact pair ownership
- DOLFINx form generation for contact
- global contact assembly policies beyond the current transition adapter
- solver-level contact algorithms beyond the current monolithic prototype
