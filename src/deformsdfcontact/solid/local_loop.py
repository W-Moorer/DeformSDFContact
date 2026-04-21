"""Backend-agnostic local quadrature loop for solid internal-force kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import IsotropicElasticParameters
from .form_mapping import SolidElementMapping
from .kernels import SolidPointKernelInput, evaluate_solid_point_kernel


@dataclass(frozen=True)
class SolidLocalLoopResult:
    """Accumulated local internal-force residual and tangent for one element."""

    local_residual_u: np.ndarray
    local_K_uu: np.ndarray


def execute_solid_local_loop(
    mapping: SolidElementMapping,
    params: IsotropicElasticParameters,
) -> SolidLocalLoopResult:
    """Accumulate local solid contributions over all quadrature points."""

    if mapping.nqp <= 0 or len(mapping.quadrature_points) == 0:
        raise ValueError("mapping.quadrature_points must be non-empty")

    local_residual_u = np.zeros(mapping.ndof_u_local, dtype=float)
    local_K_uu = np.zeros((mapping.ndof_u_local, mapping.ndof_u_local), dtype=float)

    for point in mapping.quadrature_points:
        point_result = evaluate_solid_point_kernel(
            SolidPointKernelInput(
                strain=point.strain,
                B=point.B,
                weight=point.weight,
            ),
            params,
        )
        local_residual_u += point_result.r_u_int
        local_K_uu += point_result.K_uu_int

    return SolidLocalLoopResult(
        local_residual_u=local_residual_u,
        local_K_uu=local_K_uu,
    )


__all__ = ["SolidLocalLoopResult", "execute_solid_local_loop"]
