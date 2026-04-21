"""Assembly-neutral form mapping for local SDF displacement coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    return array


def _as_weights(values: object) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"weights must have shape (nqp,), got {array.shape!r}")
    if array.shape[0] == 0:
        raise ValueError("weights must be non-empty")
    return array, int(array.shape[0])


def _as_shape_gradient_field(values: object, nqp: int, nphi_local: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape[0] != nphi_local:
            raise ValueError(
                f"shape_gradients_phi must have shape ({nphi_local}, dim), got {array.shape!r}"
            )
        return np.repeat(array[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape[0] == nqp and array.shape[1] == nphi_local:
        return array.astype(float, copy=False)
    raise ValueError(
        "shape_gradients_phi must have shape "
        f"({nphi_local}, dim) or ({nqp}, {nphi_local}, dim), got {array.shape!r}"
    )


def _as_metric_field(values: object, nqp: int, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape == (dimension, dimension):
        return np.repeat(array[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape == (nqp, dimension, dimension):
        return array.astype(float, copy=False)
    raise ValueError(
        f"A must have shape ({dimension}, {dimension}) or ({nqp}, {dimension}, {dimension}), got {array.shape!r}"
    )


def _as_metric_sensitivity_field(values: object, nqp: int, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[1:] == (dimension, dimension):
        return np.repeat(array[None, :, :, :], nqp, axis=0)
    if array.ndim == 4 and array.shape[0] == nqp and array.shape[2:] == (dimension, dimension):
        return array.astype(float, copy=False)
    raise ValueError(
        "dA_du must have shape "
        f"(ndof_u_local, {dimension}, {dimension}) or (nqp, ndof_u_local, {dimension}, {dimension}), got {array.shape!r}"
    )


@dataclass(frozen=True)
class SDFCouplingQuadraturePointData:
    """Normalized quadrature-point data for the local `K_phiu` chain."""

    grad_phi: np.ndarray
    shape_gradients_phi: np.ndarray
    A: np.ndarray
    dA_du: np.ndarray
    weight: float


@dataclass(frozen=True)
class SDFCouplingElementMapping:
    """Collection of quadrature-point data for local SDF displacement coupling."""

    phi_local: np.ndarray
    quadrature_points: tuple[SDFCouplingQuadraturePointData, ...]
    nphi_local: int
    ndof_u_local: int
    nqp: int


def build_sdf_coupling_element_mapping(
    phi_local: object,
    shape_gradients_phi: object,
    A: object,
    dA_du: object,
    weights: object,
) -> SDFCouplingElementMapping:
    """Normalize element-local arrays for the `K_phiu` local loop."""

    phi_local_array = _as_vector(phi_local, "phi_local")
    nphi_local = int(phi_local_array.shape[0])
    weights_array, nqp = _as_weights(weights)
    shape_gradients_array = _as_shape_gradient_field(shape_gradients_phi, nqp, nphi_local)
    dimension = int(shape_gradients_array.shape[2])
    A_array = _as_metric_field(A, nqp, dimension)
    dA_du_array = _as_metric_sensitivity_field(dA_du, nqp, dimension)
    ndof_u_local = int(dA_du_array.shape[1])
    grad_phi_array = np.einsum("qad,a->qd", shape_gradients_array, phi_local_array)

    quadrature_points = tuple(
        SDFCouplingQuadraturePointData(
            grad_phi=grad_phi_array[q].copy(),
            shape_gradients_phi=shape_gradients_array[q].copy(),
            A=A_array[q].copy(),
            dA_du=dA_du_array[q].copy(),
            weight=float(weights_array[q]),
        )
        for q in range(nqp)
    )
    return SDFCouplingElementMapping(
        phi_local=phi_local_array.copy(),
        quadrature_points=quadrature_points,
        nphi_local=nphi_local,
        ndof_u_local=ndof_u_local,
        nqp=nqp,
    )


__all__ = [
    "SDFCouplingElementMapping",
    "SDFCouplingQuadraturePointData",
    "build_sdf_coupling_element_mapping",
]
