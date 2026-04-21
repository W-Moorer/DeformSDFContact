# DOLFINx 0.3.0 Transition Assembly Adapter

## Scope

This document defines the first backend-specific assembly adapter used to reach
the assembled monolithic dry run in the current transition environment:

- PETSc 3.15.5
- petsc4py 3.15.1
- DOLFINx 0.3.0

The adapter is intentionally narrow. It only:

- reads local mesh and dof data from DOLFINx 0.3.0
- calls the backend-agnostic solid, SDF, and contact local chains
- inserts local contributions into PETSc vectors and matrices

It does **not** implement:

- DOLFINx UFL forms
- global nonlinear solve control
- SNES/KSP business logic
- friction
- augmented-Lagrangian outer updates

## Monolithic Block Semantics

The adapter honors the assembly-neutral block layout:

- `u` dofs first
- `phi` dofs second

and assembles the monolithic system:

`R = [R_u, R_phi]^T`

`K = [[K_uu, K_uphi], [K_phiu, K_phiphi]]`

without changing the underlying local formulas.

## Solid Adapter

The solid adapter is 2D, P1-triangle, and DOLFINx 0.3.0 compatible.

For each cell it:

- extracts local displacement dofs
- reconstructs triangle geometry from scalar P1 dof coordinates
- computes constant P1 shape gradients
- calls the small-strain solid local loop

The current structural residual is:

`R_u = R_u_int + R_u_c`

because `R_u_ext` is intentionally zero in the dry run.

## SDF Adapter

The SDF adapter is also 2D, P1-triangle, and DOLFINx 0.3.0 compatible.

For each cell it:

- extracts local scalar `phi` dofs
- uses one centroid quadrature point
- assembles `R_phi` and `K_phiphi` through the existing reinitialize local loop
- assembles `K_phiu` through the transition displacement-coupling local loop

The current metric choice is the transition one already frozen in the SDF
coupling design:

- `A = I`
- `dA/du` from the reference-state linearized metric sensitivity

## Contact Adapter

The current contact adapter is deliberately a transition mapping.

It does **not** perform:

- closest-point search
- query-point ownership logic
- reinitialization
- contact search

Instead, it maps each boundary facet to one midpoint quadrature point and
builds contact point-kernel data directly:

- `g_n` from a transition midpoint closure model
  `gap_offset - G_u u_local - N_phi phi_local`
- `G_a` from midpoint P1 shape values
- `G_u` from midpoint shape values projected onto the outward facet normal
- `H_uu_g = 0`
- `H_uphi_g = 0`

This is enough to validate block placement and local-to-global insertion of the
existing contact local chain while staying consistent with the current local
kernel sign convention. It is a transition adapter only, not the final contact
geometry backend.

## What This Adapter Validates

The assembled dry run checks that:

- local solid contributions enter `R_u` and `K_uu`
- local SDF contributions enter `R_phi`, `K_phiu`, and `K_phiphi`
- local contact contributions enter `R_u`, `K_uu`, and `K_uphi`
- PETSc receives correctly sized global vectors and matrices

This is the stopping point for the current stage. Solving the monolithic system
is intentionally out of scope.
