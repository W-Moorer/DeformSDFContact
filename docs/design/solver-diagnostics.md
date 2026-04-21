# Prototype Solver Diagnostics

## Scope

This document defines the minimal diagnostics layer used by the current
monolithic prototype solver path.

The implementation lives in:

- `src/deformsdfcontact/solvers/diagnostics.py`

The current diagnostics layer is intentionally lightweight. It is meant to make
the prototype solver path observable, not to become a production monitoring
framework.

## Current Diagnostics

The prototype records:

- total residual norm
- split residual norms:
  - `||R_u||`
  - `||R_phi||`
- Jacobian block shapes
- Jacobian block nonzero counts
- final-step contact summary:
  - backend name
  - candidate count
  - owned pair count
  - active contact point count
  - gap min / mean / max
  - summed contact measure / reaction-like scalar
- PETSc SNES converged reason
- nonlinear iteration count
- linear solve iteration count
- linear solve failure count

## Iteration Monitoring

During a nonlinear solve, the prototype monitor records one diagnostics entry
per nonlinear iteration. Each entry stores:

- nonlinear iteration index
- total residual norm
- split residual norms

The current implementation computes split residual norms by reassembling the
current residual at the monitor state. This is acceptable at prototype scale.

## Matrix Description

`describe_block_matrix(...)` reports:

- global matrix shape
- block shapes implied by `MonolithicBlockLayout`
- per-block nonzero counts

The current implementation uses dense extraction for the block statistics. This
is intentional and acceptable for the tiny prototype problems currently in use.

## Contact Summary

For benchmark-scale comparison, the final solver diagnostics also store a small
contact summary extracted from the assembled contact backend.

This summary is intentionally small and backend-thin. It is only meant to
support:

- transition-vs-pairing backend comparison
- mini benchmark regression checks
- quick visibility into active contact status

It is not meant to become a production contact monitoring system.

## What This Layer Does Not Do

This layer does **not** implement:

- persistent logging infrastructure
- large-scale sparse diagnostics
- adaptive solver control
- automatic Jacobian verification in production solves

It is a prototype diagnostics layer for the current transition environment.
