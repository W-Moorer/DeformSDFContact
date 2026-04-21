"""Backend-agnostic local quadrature loop for the SDF `K_phiu` chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupling import SDFDisplacementCouplingPointInput, evaluate_sdf_displacement_coupling_point
from .coupling_form_mapping import SDFCouplingElementMapping


@dataclass(frozen=True)
class SDFCouplingLocalLoopResult:
    """Accumulated local `K_phiu` contribution for one scalar element."""

    local_K_phiu: np.ndarray


def execute_sdf_coupling_local_loop(
    mapping: SDFCouplingElementMapping,
) -> SDFCouplingLocalLoopResult:
    """Accumulate local `K_phiu` contributions over all quadrature points."""

    if mapping.nqp <= 0 or len(mapping.quadrature_points) == 0:
        raise ValueError("mapping.quadrature_points must be non-empty")

    local_K_phiu = np.zeros((mapping.nphi_local, mapping.ndof_u_local), dtype=float)
    for point in mapping.quadrature_points:
        point_result = evaluate_sdf_displacement_coupling_point(
            SDFDisplacementCouplingPointInput(
                grad_phi=point.grad_phi,
                shape_gradients_phi=point.shape_gradients_phi,
                A=point.A,
                dA_du=point.dA_du,
                weight=point.weight,
            )
        )
        local_K_phiu += point_result.K_phiu_point
    return SDFCouplingLocalLoopResult(local_K_phiu=local_K_phiu)


__all__ = ["SDFCouplingLocalLoopResult", "execute_sdf_coupling_local_loop"]
