# Prototype Contact Pairing Backend

## Scope

This document defines the improved prototype contact backend added after the
first query-point prototype.

The implementation lives in:

- `src/deformsdfcontact/backend/dolfinx0p3/contact_pairing_backend.py`

This backend is still a transition-environment implementation for DOLFINx 0.3.0
and is **not** production-ready.

## Why This Backend Exists

The earlier query-point prototype already introduced:

- a projected query point
- a more meaningful local contact geometry than the pure midpoint surrogate

But it still hard-coded one slave point per facet on the same local patch and
did not include a distinct slave/master pairing concept.

This backend moves one step closer to the intended contact semantics by adding:

- explicit slave boundary subset
- explicit master boundary subset
- a minimal candidate set
- one owned master facet per slave facet

## Current Pairing / Ownership Logic

For each slave boundary facet:

1. compute the current slave midpoint
2. enumerate all master boundary facets in the current prototype candidate set
3. project the slave midpoint onto each current master segment
4. choose the owned master facet by minimum projected Euclidean distance
5. freeze that ownership for the current local linearization

This is intentionally small in scope. It is enough to support:

- explicit slave/master distinction
- query-point projection onto a distinct master facet
- benchmark-level comparison against the transition baseline

## Current Query Point

After ownership is selected:

- the current query point is the closest point on the owned current master
  segment
- the reference query point is reconstructed by the same segment parameter on
  the owned reference master segment
- `phi(X_c)` is interpolated in the owned master cell

## Current Local Sensitivities

The backend reuses the existing backend-agnostic contact local chain, but the
backend itself computes the local geometric inputs.

For the current prototype this means:

- `g_n = gap_offset - phi(X_c)`
- `G_u`, `H_uu^(g)`, and `H_uphi^(g)` are obtained by small local finite
  differences of the scalar closure measure
- `G_a` is the local P1 interpolation vector at the owned reference query point

The ownership choice is frozen while these local finite differences are taken.
This is a deliberate prototype simplification.

## What Is More Realistic Than Before

Compared with the earlier prototypes, this backend now has:

- distinct slave and master contact subsets
- explicit facet pairing / ownership
- projected query points on owned master facets
- contact diagnostics that can report owned pair count and active point count

## What Is Still Simplified

This backend still does **not** implement:

- production contact search
- robust distributed ownership
- contact pair updates inside line-search globalization
- production second-order contact geometry
- friction
- augmented-Lagrangian outer iterations

It is still a prototype backend. Its job is to be more realistic than the
transition/query-point baselines while staying small enough to remain solvable
and testable in the current transition environment.
