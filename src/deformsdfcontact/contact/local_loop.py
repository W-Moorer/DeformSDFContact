"""Backend-agnostic local accumulation of contact point-kernel contributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .kernels import ContactPointKernelInput, evaluate_contact_point_kernel
from .laws import ContactLaw


@dataclass(frozen=True)
class ContactLocalLoopResult:
    """Accumulated local contact residual and tangents for one local patch."""

    local_residual: np.ndarray
    local_K_uu: np.ndarray
    local_K_uphi: np.ndarray


def execute_contact_local_loop(
    point_inputs: Sequence[ContactPointKernelInput],
    law: ContactLaw,
) -> ContactLocalLoopResult:
    """Accumulate local contact contributions over multiple point inputs."""

    if len(point_inputs) == 0:
        raise ValueError("point_inputs must be non-empty")

    first_result = evaluate_contact_point_kernel(point_inputs[0], law)
    local_residual = np.zeros_like(first_result.r_u_c)
    local_K_uu = np.zeros_like(first_result.K_uu_c)
    local_K_uphi = np.zeros_like(first_result.K_uphi_c)

    for point_input in point_inputs:
        point_result = evaluate_contact_point_kernel(point_input, law)
        if point_result.r_u_c.shape != local_residual.shape:
            raise ValueError(
                "all point inputs must produce the same residual shape, got "
                f"{point_result.r_u_c.shape!r} and {local_residual.shape!r}"
            )
        if point_result.K_uu_c.shape != local_K_uu.shape:
            raise ValueError(
                "all point inputs must produce the same K_uu shape, got "
                f"{point_result.K_uu_c.shape!r} and {local_K_uu.shape!r}"
            )
        if point_result.K_uphi_c.shape != local_K_uphi.shape:
            raise ValueError(
                "all point inputs must produce the same K_uphi shape, got "
                f"{point_result.K_uphi_c.shape!r} and {local_K_uphi.shape!r}"
            )
        local_residual += point_result.r_u_c
        local_K_uu += point_result.K_uu_c
        local_K_uphi += point_result.K_uphi_c

    return ContactLocalLoopResult(
        local_residual=local_residual,
        local_K_uu=local_K_uu,
        local_K_uphi=local_K_uphi,
    )


__all__ = ["ContactLocalLoopResult", "execute_contact_local_loop"]
