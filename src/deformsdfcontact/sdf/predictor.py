"""Pure pointwise warm-start predictors for reference SDF data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vectors(values: object, dimension: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != dimension:
            raise ValueError(f"{name} must have shape ({dimension},), got {array.shape!r}")
        return array.reshape(1, dimension), True
    if array.ndim == 2:
        if array.shape[1] != dimension:
            raise ValueError(f"{name} must have shape (N, {dimension}), got {array.shape!r}")
        return array, False
    raise ValueError(f"{name} must have shape ({dimension},) or (N, {dimension}), got {array.shape!r}")


def _as_scalar_field(values: object, count: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(count, float(array), dtype=float), True
    if array.ndim == 1 and array.shape[0] == count:
        return array.astype(float, copy=False), False
    raise ValueError(f"{name} must be a scalar or have shape ({count},), got {array.shape!r}")


def _as_metric_tensors(values: object, dimension: int, count: int) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (dimension, dimension):
            raise ValueError(f"A must have shape ({dimension}, {dimension}), got {array.shape!r}")
        return np.repeat(array[None, :, :], count, axis=0), True
    if array.ndim == 3:
        if array.shape != (count, dimension, dimension):
            raise ValueError(
                f"A must have shape ({count}, {dimension}, {dimension}), got {array.shape!r}"
            )
        return array, False
    raise ValueError(
        f"A must have shape ({dimension}, {dimension}) or ({count}, {dimension}, {dimension}), "
        f"got {array.shape!r}"
    )


def _restore_scalar_like(values: np.ndarray, single: bool):
    return float(values[0]) if single else values


def _restore_vector_like(values: np.ndarray, single: bool):
    return values[0] if single else values


@dataclass(frozen=True)
class ReferencePredictorResult:
    """Structured output of the reference-geometry warm-start predictor."""

    X: object
    phi0: object
    nearest_point: object
    normal: object
    stretch_factor: object
    phi_pred: object


def metric_stretch_factor(normal: object, A: object):
    """Return `sqrt(n^T A n)` for a unit normal `n`.

    Supported shapes:

    - `normal`: `(dim,)` or `(N, dim)`
    - `A`: `(dim, dim)` or `(N, dim, dim)`
    """

    normal_array = np.asarray(normal, dtype=float)
    if normal_array.ndim == 1:
        dimension = int(normal_array.shape[0])
    elif normal_array.ndim == 2:
        dimension = int(normal_array.shape[1])
    else:
        raise ValueError(
            f"normal must have shape (dim,) or (N, dim), got {normal_array.shape!r}"
        )

    normals, single = _as_vectors(normal, dimension, "normal")
    metric_tensors, _ = _as_metric_tensors(A, dimension, normals.shape[0])
    stretch_squared = np.einsum("ni,nij,nj->n", normals, metric_tensors, normals)
    if np.any(stretch_squared <= 0.0):
        raise ValueError("n^T A n must be strictly positive for every sample")
    stretch = np.sqrt(stretch_squared)
    return _restore_scalar_like(stretch, single)


def predict_pullback_distance(phi0: object, normal: object, A: object):
    """Return the metric-stretch warm start.

    This predictor is a warm start only. It is not a reinitialized SDF and does
    not perform any PDE or variational correction.
    """

    normal_array = np.asarray(normal, dtype=float)
    if normal_array.ndim == 1:
        dimension = int(normal_array.shape[0])
    elif normal_array.ndim == 2:
        dimension = int(normal_array.shape[1])
    else:
        raise ValueError(
            f"normal must have shape (dim,) or (N, dim), got {normal_array.shape!r}"
        )

    normals, single = _as_vectors(normal, dimension, "normal")
    stretch = np.asarray(metric_stretch_factor(normals, A), dtype=float)
    phi0_array, phi0_single = _as_scalar_field(phi0, normals.shape[0], "phi0")
    predicted = phi0_array / stretch
    return _restore_scalar_like(predicted, single and phi0_single)


def predict_from_reference_geometry(X: object, reference_geometry: object, A_at_reference: object) -> ReferencePredictorResult:
    """Evaluate the metric-stretch predictor from a reference geometry object.

    `A_at_reference` may be:

    - a constant metric tensor with shape `(dim, dim)`
    - a batch of metric tensors with shape `(N, dim, dim)`
    - a callable `A_at_reference(P0)` returning one of the two forms above
    """

    phi0 = reference_geometry.phi0(X)
    nearest_point = reference_geometry.nearest_point(X)
    normal = reference_geometry.normal_at_nearest_point(X)
    metric = A_at_reference(nearest_point) if callable(A_at_reference) else A_at_reference
    stretch = metric_stretch_factor(normal, metric)
    phi_pred = predict_pullback_distance(phi0, normal, metric)
    return ReferencePredictorResult(
        X=X,
        phi0=phi0,
        nearest_point=nearest_point,
        normal=normal,
        stretch_factor=stretch,
        phi_pred=phi_pred,
    )


__all__ = [
    "ReferencePredictorResult",
    "metric_stretch_factor",
    "predict_from_reference_geometry",
    "predict_pullback_distance",
]
