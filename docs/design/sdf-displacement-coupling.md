# SDF Displacement Coupling

## Scope

This document defines the missing displacement-coupling local chain for the SDF
subproblem under:

- `src/deformsdfcontact/sdf/coupling.py`
- `src/deformsdfcontact/sdf/coupling_form_mapping.py`
- `src/deformsdfcontact/sdf/coupling_local_loop.py`

The goal is to freeze a backend-agnostic local chain for `K_phiu`.

## Current Modeling Choice

The current implementation uses a small transition model for the metric
sensitivity:

- the current pull-back metric is linearized about the reference state
- `A_0 = I`
- `delta A = -(delta H + delta H^T)`

with `H = grad(u)`.

This is sufficient for:

- a local `K_phiu` contract
- finite-difference verification
- the first assembled monolithic dry run

It is explicitly a transition kernel, not the final nonlinear metric
linearization for the full research code.

## Point Formula

For one quadrature point, define:

- `g = grad(phi)`
- `q = g^T A g - 1`
- `s_a = grad(N_a)^T A g`

The derivative with respect to one displacement dof `u_j` is:

`dR_a/du_j = [g^T dA_j g] * s_a + q * [grad(N_a)^T dA_j g]`

times the quadrature weight.

This yields a point-level `K_phiu` block with shape:

- `(nphi_local, ndof_u_local)`

## Mapping And Loop

The form-mapping layer normalizes:

- `phi_local`
- one or more scalar shape-gradient tables
- one or more metric tensors
- one or more metric sensitivities `dA/du`
- one or more quadrature weights

The local loop:

- computes one point-level `K_phiu`
- sums all quadrature-point contributions

## What This Layer Does Not Do

This layer does **not** implement:

- global assembly
- DOLFINx forms
- solver logic
- the full nonlinear metric derivative away from the current transition state

The current kernel is sufficient to close the monolithic block structure for
the assembled dry run.
