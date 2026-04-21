# SDF Reference And Predictor

## Scope

This document defines the minimal `sdf/reference + predictor` layer under
`src/deformsdfcontact/sdf`.

The goal is to introduce:

- a reference signed-distance geometry
- a narrow-band and sign convention around that reference geometry
- a metric-stretch warm-start predictor

This layer is intentionally restricted to pointwise and tensor-level logic.

## What This Layer Does

Implemented now:

- a minimal reference geometry: `ReferencePlane`
- reference signed distance `phi0(X)`
- reference nearest point `Pi0(X)`
- reference unit normal `N0(Pi0(X))`
- narrow-band test `|phi0(X)| < delta`
- sign queries
- the metric-stretch warm-start predictor

The predictor follows the foundation formula:

`phi_pred(X) = phi0(X) / sqrt(N0(Pi0(X))^T A(Pi0(X); d) N0(Pi0(X)))`

with:

- `A = C^{-1}`
- `A` supplied as a pointwise tensor input
- `phi_pred` used only as a warm start

## What This Layer Does Not Do

This layer does **not** implement:

- reinitialization
- contact
- query-point ownership
- query-point search on meshes
- assembly
- DOLFINx forms
- PETSc solver logic
- Taylor or inverse predictors as formal predictor paths

If other predictor variants are ever explored, they should remain optional and
must not displace the metric-stretch predictor as the primary interface.

## Reference Geometry Choice

The current reference layer uses the simplest analytically controlled geometry:
an infinite plane.

This is enough to validate:

- signed-distance conventions
- nearest-point projection
- unit normals
- narrow-band logic
- metric-based scaling of the distance predictor

It is deliberately not a general surface representation.

## Shape Conventions

The layer supports:

- single points with shape `(dim,)`
- point batches with shape `(N, dim)`

Similarly, the predictor supports:

- a single metric tensor with shape `(dim, dim)`
- a batch of metric tensors with shape `(N, dim, dim)`

Broadcasting is intentionally limited and explicit.

## Predictor Interpretation

The metric-stretch predictor is a geometric warm start.

It is **not**:

- a reinitialized signed-distance field
- a guarantee of exact post-deformation distance
- a replacement for later PDE or variational correction

Its intended role is to provide a theory-consistent initial guess before any
future transport/reinitialization stage exists.

## Public Surface

Implemented now:

- `deformsdfcontact.sdf.reference.ReferencePlane`
- `deformsdfcontact.sdf.predictor.metric_stretch_factor`
- `deformsdfcontact.sdf.predictor.predict_pullback_distance`
- `deformsdfcontact.sdf.predictor.predict_from_reference_geometry`
- `deformsdfcontact.sdf.predictor.ReferencePredictorResult`

## Test Strategy

The first tests stay completely independent from DOLFINx and PETSc.

They cover:

- plane reference distance and projection
- unit normal and sign conventions
- narrow-band classification
- rigid-motion invariance under `A = I`
- scaling under uniaxial stretch
- zero-level preservation and sign preservation under shear
- batch input handling

This keeps the layer stable before any FE or solver integration begins.
