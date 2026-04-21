"""Backend-agnostic point kernels for local normal-contact contributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .laws import ContactLaw


def _as_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    return array


def _as_matrix(values: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape!r}, got {array.shape!r}")
    return array


def _as_scalar(value: object, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got {array.shape!r}")
    return float(array)


@dataclass(frozen=True)
class ContactPointKernelInput:
    """Local point-kernel inputs assembled from precomputed geometry objects.

    Shape conventions:

    - `G_u`: `(ndof_u_local,)`
    - `G_a`: `(nphi_local,)`
    - `H_uu_g`: `(ndof_u_local, ndof_u_local)`
    - `H_uphi_g`: `(ndof_u_local, nphi_local)`
    """

    g_n: object
    G_u: object
    G_a: object
    H_uu_g: object
    H_uphi_g: object
    weight: object = 1.0


@dataclass(frozen=True)
class ContactPointKernelResult:
    """Structured output of one local contact point-kernel evaluation."""

    lambda_n: float
    k_n: float
    r_u_c: np.ndarray
    K_uu_c: np.ndarray
    K_uphi_c: np.ndarray


def evaluate_contact_point_kernel(
    kernel_input: ContactPointKernelInput,
    law: ContactLaw,
) -> ContactPointKernelResult:
    """Evaluate one penalty-based local contact point kernel."""

    G_u = _as_vector(kernel_input.G_u, "G_u")
    G_a = _as_vector(kernel_input.G_a, "G_a")
    H_uu_g = _as_matrix(kernel_input.H_uu_g, "H_uu_g", (G_u.shape[0], G_u.shape[0]))
    H_uphi_g = _as_matrix(
        kernel_input.H_uphi_g,
        "H_uphi_g",
        (G_u.shape[0], G_a.shape[0]),
    )
    weight = _as_scalar(kernel_input.weight, "weight")
    g_n = _as_scalar(kernel_input.g_n, "g_n")

    lambda_n, k_n = law.evaluate(g_n)
    r_u_c = weight * lambda_n * G_u
    K_uu_c = weight * (k_n * np.outer(G_u, G_u) + lambda_n * H_uu_g)
    K_uphi_c = weight * (k_n * np.outer(G_u, G_a) + lambda_n * H_uphi_g)

    return ContactPointKernelResult(
        lambda_n=float(lambda_n),
        k_n=float(k_n),
        r_u_c=r_u_c,
        K_uu_c=K_uu_c,
        K_uphi_c=K_uphi_c,
    )


__all__ = [
    "ContactPointKernelInput",
    "ContactPointKernelResult",
    "evaluate_contact_point_kernel",
]
