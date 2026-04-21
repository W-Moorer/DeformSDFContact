# Contact Strip Mini Benchmark

## Scope

This document defines the first mini benchmark used to compare the current
contact backends inside the unified monolithic solve path.

The implementation entry point lives in:

- `src/deformsdfcontact/backend/dolfinx0p3/problem.py`
  - `build_contact_strip_benchmark(...)`

This benchmark remains a transition-environment benchmark. It is intentionally
small and manufactured enough to stay stable on the current DOLFINx 0.3.0 +
PETSc prototype path.

## Geometry

The benchmark uses:

- a unit square domain
- a `2 x 2` triangular mesh
- top boundary as the slave contact boundary
- bottom boundary as the master contact boundary

This is still minimal, but it is more realistic than the earlier all-boundary
manufactured toy problem because contact is now restricted to a clear boundary
subset with an explicit slave/master split.

## Unknowns

The monolithic unknown remains:

- `x = [u, phi]^T`

with the same block structure:

- `R = [R_u, R_phi]^T`
- `K = [[K_uu, K_uphi], [K_phiu, K_phiphi]]`

## Reference State

The current reference state is kept simple:

- `u* = 0`
- `phi*(x, y) = phi_bias + 0.22 y + 0.05 x (1 - x)`

This reference field is not presented as a production signed-distance field. It
is a small manufactured field chosen so that:

- the transition baseline sees active contact on the slave top boundary
- the pairing backend sees a different, interpretable contact state on the
  master bottom boundary

## Boundary Conditions

The current benchmark uses the same minimal prototype boundary-condition
strategy as the earlier toy problem:

- structural Dirichlet on the left boundary
- scalar `phi` Dirichlet on the left boundary plus one anchor node

This is a prototype constraint strategy, not a production BC policy.

## Continuation Parameter

The current stepped benchmark uses:

- `phi_bias`

This keeps the continuation path small and stable while still changing the
contact state in a controlled way.

The default comparison schedule used in the smoke tests is:

- `phi_bias = [0.3, 0.35, 0.4]`

An intentionally harder schedule, such as `[0.3, 0.4, 0.5]`, is used to test
failure reporting for the pairing backend.

## Benchmark Observables

The current comparison records:

- per-step converged reason
- per-step nonlinear iteration count
- per-step final residual norm
- per-step `||R_u||`
- per-step `||R_phi||`
- per-step contact summary:
  - active point count
  - candidate count
  - owned pair count
  - gap min / mean / max
  - summed reaction-like scalar

This is enough to make backend behavior comparable without introducing a larger
benchmark harness.

## What This Benchmark Does Not Try To Be

This mini benchmark does **not** try to be:

- a production contact benchmark
- a multi-body benchmark
- a friction benchmark
- an augmented-Lagrangian benchmark
- a performance benchmark

It is the first small benchmark whose geometry and contact subsets are explicit
enough to compare the transition baseline and the improved pairing backend in a
repeatable way.
