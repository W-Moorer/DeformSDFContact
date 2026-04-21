# Materials Foundation

## Scope

This document defines the next extracted layer under `src/deformsdfcontact`:
the tensor-level material foundation.

The purpose is to establish a stable material-model interface that can be used
later by assembly code, while explicitly staying out of:

- element assembly
- contact
- predictor logic
- solver orchestration

## Reused Kinematics State

The material layer reuses the structured return of the kinematics layer:

- `FiniteStrainState`
- `finite_strain_kinematics(u)`
- `finite_strain_kinematics_from_F(F)`

The state provides:

- `dimension`
- `I`
- `grad_u`
- `F`
- `C`
- `C_inv`
- `J`
- `E`

This is sufficient for tensor-level constitutive evaluation.

## Interface Choice

The common constitutive interface is defined in terms of the deformation
gradient `F`:

- `strain_energy_density(state, params)` returns `W(F)`
- `stress_measure(state, params)` returns the first Piola stress `P = dW/dF`
- `consistent_tangent(state, params)` returns the rank-4 tensor `A = dP/dF`

This keeps the interface uniform across:

- a small-strain model expressed through `sym(F - I)`
- a finite-strain hyperelastic model expressed directly through `F`

## Parameters

The minimal foundation currently uses one parameter container:

- `IsotropicElasticParameters(E, nu)`

From these we derive:

- `mu`
- `lambda`

This is enough for both minimal models and makes the small-strain consistency
checks straightforward.

## Implemented Minimal Models

### LinearElasticSmallStrain

This model is intentionally small-strain, but is embedded in an `F`-based
interface through:

- `eps = sym(F - I)`
- `W = 0.5 * lambda * tr(eps)^2 + mu * eps:eps`

The resulting `P = dW/dF` coincides with the familiar linear elastic stress.

### CompressibleNeoHookean

The implemented compressible Neo-Hookean energy is:

- `W = 0.5 * mu * (tr(C) - d) - mu * ln(J) + 0.5 * lambda * ln(J)^2`

with:

- `C = F^T F`
- `J = det(F)`

This is a minimal but standard hyperelastic baseline for tensor-level tests.

## Why Stop Here

At this stage the repository needs:

1. a stable kinematics layer
2. a stable constitutive layer
3. only then a decision on how assembly and contact should consume them

If assembly, contact, or predictor logic is introduced now, the material API
will get shaped by solver details too early.

## Test Strategy

The first material tests are tensor-level only.

They cover:

- rigid-motion invariance checks
- small-strain consistency between the two models near `F = I`
- tangent agreement with directional finite differences

They do **not** cover:

- cell integration
- quadrature point loops
- DOLFINx form assembly
- contact traction laws
