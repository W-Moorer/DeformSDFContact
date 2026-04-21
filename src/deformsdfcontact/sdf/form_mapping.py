"""Assembly-neutral adapter contracts for reinitialization kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _as_vector(values: object, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _as_shape_values(values: object, nshape: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != nshape:
        raise ValueError(
            f"shape_values must have shape (nqp, {nshape}), got {array.shape!r}"
        )
    return array


def _as_shape_gradients(values: object, nqp: int, nshape: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[0] != nqp or array.shape[1] != nshape:
        raise ValueError(
            "shape_gradients must have shape "
            f"({nqp}, {nshape}, dim), got {array.shape!r}"
        )
    return array


def _as_weights(values: object, nqp: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != nqp:
        raise ValueError(f"weights must have shape ({nqp},), got {array.shape!r}")
    return array


def _as_metric_tensors(values: object, nqp: int, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (dimension, dimension):
            raise ValueError(
                f"A must have shape ({dimension}, {dimension}), got {array.shape!r}"
            )
        return np.repeat((0.5 * (array + array.T))[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape == (nqp, dimension, dimension):
        return 0.5 * (array + np.swapaxes(array, 1, 2))
    raise ValueError(
        "A must have shape "
        f"({dimension}, {dimension}) or ({nqp}, {dimension}, {dimension}), "
        f"got {array.shape!r}"
    )


def _as_nonnegative_scalars(values: object, nqp: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if scalar < 0.0:
            raise ValueError(f"{name} must be nonnegative, got {scalar!r}")
        return np.full(nqp, scalar, dtype=float)
    if array.ndim == 1 and array.shape[0] == nqp:
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative at every quadrature point")
        return array.astype(float, copy=False)
    raise ValueError(f"{name} must be a scalar or have shape ({nqp},), got {array.shape!r}")


def _as_point_targets(
    values: object,
    nqp: int,
    nshape: int,
    shape_values: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(nqp, float(array), dtype=float)
    if array.ndim == 1 and array.shape[0] == nshape:
        return shape_values @ array
    if array.ndim == 1 and array.shape[0] == nqp:
        return array.astype(float, copy=False)
    raise ValueError(
        "phi_target must be a scalar or have shape "
        f"({nshape},) or ({nqp},), got {array.shape!r}"
    )


@dataclass(frozen=True)
class ReinitializeQuadraturePointData:
    """Assembly-neutral local kernel data for one quadrature point."""

    phi: float
    grad_phi: np.ndarray
    phi_target: float
    shape_values: np.ndarray
    shape_gradients: np.ndarray
    A: np.ndarray
    weight: float
    beta: float

    @property
    def nshape(self) -> int:
        return int(self.shape_values.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.grad_phi.shape[0])


@dataclass(frozen=True)
class ReinitializeElementMapping:
    """Collection of quadrature-point data for one scalar element."""

    phi_local: np.ndarray
    quadrature_points: tuple[ReinitializeQuadraturePointData, ...]

    @property
    def nshape(self) -> int:
        return int(self.phi_local.shape[0])

    @property
    def nqp(self) -> int:
        return len(self.quadrature_points)

    @property
    def dimension(self) -> int:
        if not self.quadrature_points:
            raise ValueError("quadrature_points must be non-empty")
        return self.quadrature_points[0].dimension


class ReinitializeFormAdapter(Protocol):
    """Protocol for backend-specific element-to-kernel mapping adapters."""

    def map_element(
        self,
        phi_local: object,
        shape_values: object,
        shape_gradients: object,
        A: object,
        weights: object,
        *,
        phi_target: object = 0.0,
        beta: object = 0.0,
    ) -> ReinitializeElementMapping:
        """Return assembly-neutral quadrature data for one scalar element."""


def build_reinitialize_element_mapping(
    phi_local: object,
    shape_values: object,
    shape_gradients: object,
    A: object,
    weights: object,
    *,
    phi_target: object = 0.0,
    beta: object = 0.0,
) -> ReinitializeElementMapping:
    """Build a validated assembly-neutral element mapping from array data."""

    phi_local_array = _as_vector(phi_local, "phi_local")
    nshape = phi_local_array.shape[0]
    shape_values_array = _as_shape_values(shape_values, nshape)
    nqp = shape_values_array.shape[0]
    shape_gradients_array = _as_shape_gradients(shape_gradients, nqp, nshape)
    dimension = shape_gradients_array.shape[2]
    metric_tensors = _as_metric_tensors(A, nqp, dimension)
    weights_array = _as_weights(weights, nqp)
    beta_array = _as_nonnegative_scalars(beta, nqp, "beta")
    phi_target_array = _as_point_targets(phi_target, nqp, nshape, shape_values_array)

    phi_values = shape_values_array @ phi_local_array
    grad_values = np.einsum("qnd,n->qd", shape_gradients_array, phi_local_array)

    quadrature_points = tuple(
        ReinitializeQuadraturePointData(
            phi=float(phi_values[q]),
            grad_phi=grad_values[q].copy(),
            phi_target=float(phi_target_array[q]),
            shape_values=shape_values_array[q].copy(),
            shape_gradients=shape_gradients_array[q].copy(),
            A=metric_tensors[q].copy(),
            weight=float(weights_array[q]),
            beta=float(beta_array[q]),
        )
        for q in range(nqp)
    )
    return ReinitializeElementMapping(
        phi_local=phi_local_array.copy(),
        quadrature_points=quadrature_points,
    )


__all__ = [
    "ReinitializeElementMapping",
    "ReinitializeFormAdapter",
    "ReinitializeQuadraturePointData",
    "build_reinitialize_element_mapping",
]
