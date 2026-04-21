"""Backend-agnostic normalization of local surface quadrature data for contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_weights(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"weights must have shape (nqp,), got {array.shape!r}")
    if array.shape[0] == 0:
        raise ValueError("weights must be non-empty")
    return array


def _as_scalar_or_point_array(values: object, nqp: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(nqp, float(array), dtype=float)
    if array.ndim == 1 and array.shape[0] == nqp:
        return array.astype(float, copy=False)
    raise ValueError(f"{name} must be a scalar or have shape ({nqp},), got {array.shape!r}")


def _as_vector_field(values: object, nqp: int, name: str) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        size = array.shape[0]
        return np.repeat(array[None, :], nqp, axis=0), size
    if array.ndim == 2 and array.shape[0] == nqp:
        return array.astype(float, copy=False), int(array.shape[1])
    raise ValueError(
        f"{name} must have shape (n,) or ({nqp}, n), got {array.shape!r}"
    )


def _as_square_matrix_field(values: object, nqp: int, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape!r}")
        return np.repeat(array[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape == (nqp, size, size):
        return array.astype(float, copy=False)
    raise ValueError(
        f"{name} must have shape ({size}, {size}) or ({nqp}, {size}, {size}), got {array.shape!r}"
    )


def _as_rect_matrix_field(
    values: object,
    nqp: int,
    nrow: int,
    ncol: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (nrow, ncol):
            raise ValueError(f"{name} must have shape ({nrow}, {ncol}), got {array.shape!r}")
        return np.repeat(array[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape == (nqp, nrow, ncol):
        return array.astype(float, copy=False)
    raise ValueError(
        f"{name} must have shape ({nrow}, {ncol}) or ({nqp}, {nrow}, {ncol}), got {array.shape!r}"
    )


@dataclass(frozen=True)
class ContactQuadraturePointData:
    """Normalized local contact data for one surface quadrature point."""

    g_n: float
    G_u: np.ndarray
    G_a: np.ndarray
    H_uu_g: np.ndarray
    H_uphi_g: np.ndarray
    weight: float


@dataclass(frozen=True)
class ContactSurfaceMapping:
    """Collection of normalized surface quadrature-point data for one local patch."""

    point_data: tuple[ContactQuadraturePointData, ...]
    ndof_u_local: int
    nphi_local: int
    nqp: int


def build_contact_surface_mapping(
    *,
    g_n: object,
    G_u: object,
    G_a: object,
    H_uu_g: object,
    H_uphi_g: object,
    weights: object,
) -> ContactSurfaceMapping:
    """Normalize local surface quadrature arrays into a pointwise contract."""

    weights_array = _as_weights(weights)
    nqp = int(weights_array.shape[0])
    g_n_array = _as_scalar_or_point_array(g_n, nqp, "g_n")
    G_u_array, ndof_u_local = _as_vector_field(G_u, nqp, "G_u")
    G_a_array, nphi_local = _as_vector_field(G_a, nqp, "G_a")
    H_uu_array = _as_square_matrix_field(H_uu_g, nqp, ndof_u_local, "H_uu_g")
    H_uphi_array = _as_rect_matrix_field(
        H_uphi_g,
        nqp,
        ndof_u_local,
        nphi_local,
        "H_uphi_g",
    )

    point_data = tuple(
        ContactQuadraturePointData(
            g_n=float(g_n_array[q]),
            G_u=G_u_array[q].copy(),
            G_a=G_a_array[q].copy(),
            H_uu_g=H_uu_array[q].copy(),
            H_uphi_g=H_uphi_array[q].copy(),
            weight=float(weights_array[q]),
        )
        for q in range(nqp)
    )
    return ContactSurfaceMapping(
        point_data=point_data,
        ndof_u_local=ndof_u_local,
        nphi_local=nphi_local,
        nqp=nqp,
    )


__all__ = [
    "ContactQuadraturePointData",
    "ContactSurfaceMapping",
    "build_contact_surface_mapping",
]
