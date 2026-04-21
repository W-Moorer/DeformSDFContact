# Kinematics Foundation

## Scope

This document defines the first extracted layer under `src/deformsdfcontact`:
the kinematics foundation.

The goal is to pull out the reusable finite-strain tensor algebra that already
appears in the benchmark code, while explicitly **not** starting predictor,
contact, or solver implementation.

## Existing Reference In This Repository

Two existing locations drove this extraction:

- `pullback_sdf_contact/metric/tensors.py`
  - currently exposes `kinematics(u)` and returns `(F, C, inv(C))`
- `pullback_sdf_contact/contact_geometry/sensitivities.py`
  - builds contact-geometric quantities from `F` and `F^{-1}`

The new module keeps the same tensor vocabulary, but moves it into a package
surface that can be unit-tested independently.

## Design Goals

- keep the layer UFL-first and side-effect free
- support both direct tensor assembly from `grad(u)` and convenience assembly
  from a displacement field `u`
- preserve a compatibility helper matching the current tuple-style
  `kinematics(u) -> (F, C, C_inv)`
- expose named functions for the individual tensor measures so later modules do
  not have to unpack positional tuples

## Non-Goals

This layer does **not** include:

- predictor logic
- contact gap, normals, tangents, or pull-back maps such as `E = -F^{-1}L`
- quadrature-point ownership or geometry queries
- material laws
- residual/Jacobian assembly

Those stay outside the package surface for now.

## Proposed Module Surface

Implemented now:

- `deformsdfcontact.kinematics`
  - `displacement_gradient(u)`
  - `deformation_gradient_from_grad_u(grad_u)`
  - `deformation_gradient(u)`
  - `right_cauchy_green(F)`
  - `left_cauchy_green(F)`
  - `inverse_deformation_gradient(F)`
  - `inverse_right_cauchy_green(C)`
  - `jacobian(F)`
  - `green_lagrange_strain(F)`
  - `small_strain(u)`
  - `finite_strain_kinematics(u)`
  - `kinematics(u)` compatibility helper

Reserved but intentionally empty:

- `deformsdfcontact.predictor`
- `deformsdfcontact.contact`

## API Shape

The primary structured entry point is `finite_strain_kinematics(u)`, which
returns a dataclass with:

- `dimension`
- `I`
- `grad_u`
- `F`
- `C`
- `C_inv`
- `J`
- `E`

This keeps later code readable and avoids repeated local recomputation.

The tuple helper `kinematics(u)` remains available for code that currently only
needs `(F, C, C_inv)`.

## Why This Is Enough For Now

This repository already needs a clean separation between:

1. reusable tensor algebra
2. constitutive laws
3. predictor/load-step logic
4. contact geometry and contact mechanics
5. PETSc/DOLFINx assembly

Only item 1 is stable enough to extract without prematurely freezing the rest
of the architecture.

## Test Strategy

The first unit tests stay independent from DOLFINx and use UFL directly.

They verify:

- tensor construction from a constant displacement gradient
- `C`, `C^{-1}`, `J`, and `E` consistency
- the compatibility helper built from an affine displacement field

This keeps the tests fast and isolates tensor algebra from FE discretization.
