"""Assembly-neutral form mapping for 2D small-strain solid local kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    return array


def _as_B_field(values: object, nqp: int, ndof_u_local: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape != (3, ndof_u_local):
            raise ValueError(f"B must have shape (3, {ndof_u_local}), got {array.shape!r}")
        return np.repeat(array[None, :, :], nqp, axis=0)
    if array.ndim == 3 and array.shape == (nqp, 3, ndof_u_local):
        return array.astype(float, copy=False)
    raise ValueError(
        f"B must have shape (3, {ndof_u_local}) or ({nqp}, 3, {ndof_u_local}), got {array.shape!r}"
    )


def _as_weights(values: object, nqp: int | None = None) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"weights must have shape (nqp,), got {array.shape!r}")
    if array.shape[0] == 0:
        raise ValueError("weights must be non-empty")
    if nqp is not None and array.shape[0] != nqp:
        raise ValueError(f"weights must have shape ({nqp},), got {array.shape!r}")
    return array, int(array.shape[0])


@dataclass(frozen=True)
class SolidQuadraturePointData:
    """Local solid quadrature data for one point."""

    strain: np.ndarray
    B: np.ndarray
    weight: float


@dataclass(frozen=True)
class SolidElementMapping:
    """Collection of quadrature-point data for one solid element."""

    u_local: np.ndarray
    quadrature_points: tuple[SolidQuadraturePointData, ...]
    ndof_u_local: int
    nqp: int


def build_solid_element_mapping(
    u_local: object,
    B: object,
    weights: object,
) -> SolidElementMapping:
    """Build a validated solid element mapping from local arrays."""

    local_values = _as_vector(u_local, "u_local")
    ndof_u_local = int(local_values.shape[0])
    weights_array, nqp = _as_weights(weights)
    B_array = _as_B_field(B, nqp, ndof_u_local)
    strains = np.einsum("qij,j->qi", B_array, local_values)
    quadrature_points = tuple(
        SolidQuadraturePointData(
            strain=strains[q].copy(),
            B=B_array[q].copy(),
            weight=float(weights_array[q]),
        )
        for q in range(nqp)
    )
    return SolidElementMapping(
        u_local=local_values.copy(),
        quadrature_points=quadrature_points,
        ndof_u_local=ndof_u_local,
        nqp=nqp,
    )


__all__ = [
    "SolidElementMapping",
    "SolidQuadraturePointData",
    "build_solid_element_mapping",
]
