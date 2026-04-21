"""Backend-agnostic surface quadrature executor for local contact kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .form_mapping import ContactQuadraturePointData, ContactSurfaceMapping
from .kernels import ContactPointKernelInput, ContactPointKernelResult, evaluate_contact_point_kernel
from .laws import ContactLaw


@dataclass(frozen=True)
class ContactSurfaceLocalResult:
    """Accumulated local contact contributions over one surface quadrature patch."""

    local_residual_u: np.ndarray
    local_K_uu: np.ndarray
    local_K_uphi: np.ndarray
    point_results: tuple[ContactPointKernelResult, ...] | None = None


def _as_kernel_input(point_data: ContactQuadraturePointData) -> ContactPointKernelInput:
    return ContactPointKernelInput(
        g_n=point_data.g_n,
        G_u=point_data.G_u,
        G_a=point_data.G_a,
        H_uu_g=point_data.H_uu_g,
        H_uphi_g=point_data.H_uphi_g,
        weight=point_data.weight,
    )


def execute_contact_surface_local_loop(
    mapping: ContactSurfaceMapping,
    law: ContactLaw,
) -> ContactSurfaceLocalResult:
    """Evaluate and accumulate local contact contributions over a surface mapping."""

    if mapping.nqp <= 0 or len(mapping.point_data) == 0:
        raise ValueError("mapping.point_data must be non-empty")
    if len(mapping.point_data) != mapping.nqp:
        raise ValueError(
            f"mapping.nqp={mapping.nqp} does not match len(point_data)={len(mapping.point_data)}"
        )

    point_results = tuple(
        evaluate_contact_point_kernel(_as_kernel_input(point_data), law)
        for point_data in mapping.point_data
    )
    first = point_results[0]
    local_residual_u = np.zeros_like(first.r_u_c)
    local_K_uu = np.zeros_like(first.K_uu_c)
    local_K_uphi = np.zeros_like(first.K_uphi_c)

    for point_result in point_results:
        if point_result.r_u_c.shape != local_residual_u.shape:
            raise ValueError("all point results must share the same residual shape")
        if point_result.K_uu_c.shape != local_K_uu.shape:
            raise ValueError("all point results must share the same K_uu shape")
        if point_result.K_uphi_c.shape != local_K_uphi.shape:
            raise ValueError("all point results must share the same K_uphi shape")
        local_residual_u += point_result.r_u_c
        local_K_uu += point_result.K_uu_c
        local_K_uphi += point_result.K_uphi_c

    return ContactSurfaceLocalResult(
        local_residual_u=local_residual_u,
        local_K_uu=local_K_uu,
        local_K_uphi=local_K_uphi,
        point_results=point_results,
    )


__all__ = [
    "ContactSurfaceLocalResult",
    "execute_contact_surface_local_loop",
]
