"""Point-level internal-force and tangent kernels for 2D small-strain elasticity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..materials import IsotropicElasticParameters


def _as_vector(values: object, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _as_matrix(values: object, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (m, n), got {array.shape!r}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape!r}, got {array.shape!r}")
    return array


def plane_strain_constitutive_matrix(params: IsotropicElasticParameters) -> np.ndarray:
    """Return the 2D plane-strain isotropic constitutive matrix."""

    mu = float(params.mu)
    lame_lambda = float(params.lame_lambda)
    return np.array(
        [
            [lame_lambda + 2.0 * mu, lame_lambda, 0.0],
            [lame_lambda, lame_lambda + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=float,
    )


def triangle_p1_B_matrix(shape_gradients: object) -> np.ndarray:
    """Return the constant CST `B` matrix from P1 triangle shape gradients.

    Input shape:

    - `shape_gradients`: `(3, 2)`

    Output shape:

    - `B`: `(3, 6)`
    """

    gradients = _as_matrix(shape_gradients, "shape_gradients", (3, 2))
    B = np.zeros((3, 6), dtype=float)
    for a in range(3):
        dNdx, dNdy = gradients[a]
        col = 2 * a
        B[0, col] = dNdx
        B[1, col + 1] = dNdy
        B[2, col] = dNdy
        B[2, col + 1] = dNdx
    return B


@dataclass(frozen=True)
class SolidPointKernelInput:
    """Inputs for one small-strain linear-elastic point kernel."""

    strain: object
    B: object
    weight: object = 1.0


@dataclass(frozen=True)
class SolidPointKernelResult:
    """Output of one small-strain linear-elastic point kernel."""

    stress: np.ndarray
    constitutive_matrix: np.ndarray
    r_u_int: np.ndarray
    K_uu_int: np.ndarray


def evaluate_solid_point_kernel(
    kernel_input: SolidPointKernelInput,
    params: IsotropicElasticParameters,
) -> SolidPointKernelResult:
    """Evaluate one 2D plane-strain linear-elastic point kernel."""

    strain = _as_vector(kernel_input.strain, "strain", length=3)
    B = _as_matrix(kernel_input.B, "B")
    if B.shape[0] != 3:
        raise ValueError(f"B must have shape (3, ndof_u_local), got {B.shape!r}")
    weight = float(np.asarray(kernel_input.weight, dtype=float))
    C = plane_strain_constitutive_matrix(params)
    stress = C @ strain
    r_u_int = weight * (B.T @ stress)
    K_uu_int = weight * (B.T @ C @ B)
    return SolidPointKernelResult(
        stress=stress,
        constitutive_matrix=C,
        r_u_int=r_u_int,
        K_uu_int=K_uu_int,
    )


__all__ = [
    "SolidPointKernelInput",
    "SolidPointKernelResult",
    "evaluate_solid_point_kernel",
    "plane_strain_constitutive_matrix",
    "triangle_p1_B_matrix",
]
