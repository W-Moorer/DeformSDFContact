"""Backend-agnostic local quadrature executor for reinitialization kernels."""

from __future__ import annotations

import numpy as np

from .form_mapping import ReinitializeElementMapping, ReinitializeQuadraturePointData
from .reinitialize import reinitialize_point_residual, reinitialize_point_tangent


def _validate_quadrature_point(
    quadrature_point: ReinitializeQuadraturePointData,
    *,
    nshape: int,
    dimension: int,
) -> None:
    if quadrature_point.shape_values.shape != (nshape,):
        raise ValueError(
            f"shape_values must have shape ({nshape},), got {quadrature_point.shape_values.shape!r}"
        )
    if quadrature_point.shape_gradients.shape != (nshape, dimension):
        raise ValueError(
            "shape_gradients must have shape "
            f"({nshape}, {dimension}), got {quadrature_point.shape_gradients.shape!r}"
        )
    if quadrature_point.grad_phi.shape != (dimension,):
        raise ValueError(
            f"grad_phi must have shape ({dimension},), got {quadrature_point.grad_phi.shape!r}"
        )
    if quadrature_point.A.shape != (dimension, dimension):
        raise ValueError(
            f"A must have shape ({dimension}, {dimension}), got {quadrature_point.A.shape!r}"
        )
    if quadrature_point.beta < 0.0:
        raise ValueError(f"beta must be nonnegative, got {quadrature_point.beta!r}")


def execute_reinitialize_local_loop(
    mapping: ReinitializeElementMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate one element residual and tangent from quadrature-point data."""

    nshape = int(mapping.phi_local.shape[0])
    if not mapping.quadrature_points:
        raise ValueError("quadrature_points must be non-empty")

    first_qp = mapping.quadrature_points[0]
    dimension = int(first_qp.grad_phi.shape[0])
    residual = np.zeros(nshape, dtype=float)
    tangent = np.zeros((nshape, nshape), dtype=float)

    for quadrature_point in mapping.quadrature_points:
        _validate_quadrature_point(
            quadrature_point,
            nshape=nshape,
            dimension=dimension,
        )
        for a in range(nshape):
            residual[a] += reinitialize_point_residual(
                quadrature_point.grad_phi,
                quadrature_point.shape_gradients[a],
                quadrature_point.A,
                phi=quadrature_point.phi,
                phi_target=quadrature_point.phi_target,
                test_value=quadrature_point.shape_values[a],
                beta=quadrature_point.beta,
                weight=quadrature_point.weight,
            )
            for b in range(nshape):
                tangent[a, b] += reinitialize_point_tangent(
                    quadrature_point.grad_phi,
                    quadrature_point.shape_gradients[a],
                    quadrature_point.shape_gradients[b],
                    quadrature_point.A,
                    test_value=quadrature_point.shape_values[a],
                    trial_value=quadrature_point.shape_values[b],
                    beta=quadrature_point.beta,
                    weight=quadrature_point.weight,
                )

    return residual, tangent


__all__ = ["execute_reinitialize_local_loop"]
