"""Local SDF displacement-coupling kernels for the monolithic dry run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


def _as_tensor(values: object, name: str, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (n, m, p), got {array.shape!r}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape!r}, got {array.shape!r}")
    return array


def linearized_metric_sensitivity_from_shape_gradients(
    shape_gradients_u: object,
) -> np.ndarray:
    """Return the reference-state metric sensitivity `dA/du` for vector P1 dofs.

    Input shape:

    - `shape_gradients_u`: `(nnode, dim)`

    Output shape:

    - `(ndof_u_local, dim, dim)` with `ndof_u_local = nnode * dim`
    """

    gradients = _as_matrix(shape_gradients_u, "shape_gradients_u")
    nnode, dimension = gradients.shape
    ndof_u_local = nnode * dimension
    dA_du = np.zeros((ndof_u_local, dimension, dimension), dtype=float)

    for a in range(nnode):
        gradN = gradients[a]
        for component in range(dimension):
            dof = dimension * a + component
            dH = np.zeros((dimension, dimension), dtype=float)
            dH[component, :] = gradN
            dA_du[dof] = -(dH + dH.T)
    return dA_du


@dataclass(frozen=True)
class SDFDisplacementCouplingPointInput:
    """Point-level inputs for the SDF displacement-coupling kernel."""

    grad_phi: object
    shape_gradients_phi: object
    A: object
    dA_du: object
    weight: object = 1.0


@dataclass(frozen=True)
class SDFDisplacementCouplingPointResult:
    """Point-level `K_phiu` contribution."""

    K_phiu_point: np.ndarray


def evaluate_sdf_displacement_coupling_point(
    kernel_input: SDFDisplacementCouplingPointInput,
) -> SDFDisplacementCouplingPointResult:
    """Evaluate one point-level `K_phiu` contribution."""

    grad_phi = _as_vector(kernel_input.grad_phi, "grad_phi")
    dimension = int(grad_phi.shape[0])
    shape_gradients_phi = _as_matrix(kernel_input.shape_gradients_phi, "shape_gradients_phi")
    if shape_gradients_phi.shape[1] != dimension:
        raise ValueError(
            "shape_gradients_phi must have shape (nphi_local, dim), "
            f"got {shape_gradients_phi.shape!r}"
        )
    A = _as_matrix(kernel_input.A, "A", (dimension, dimension))
    ndof_u_local = int(np.asarray(kernel_input.dA_du, dtype=float).shape[0])
    dA_du = _as_tensor(kernel_input.dA_du, "dA_du", (ndof_u_local, dimension, dimension))
    weight = float(np.asarray(kernel_input.weight, dtype=float))

    metric_grad_phi = A @ grad_phi
    defect = float(grad_phi @ metric_grad_phi - 1.0)
    fluxes = shape_gradients_phi @ metric_grad_phi

    K_phiu = np.zeros((shape_gradients_phi.shape[0], ndof_u_local), dtype=float)
    for j in range(ndof_u_local):
        delta_metric_grad_phi = dA_du[j] @ grad_phi
        delta_q = float(grad_phi @ delta_metric_grad_phi)
        delta_flux = shape_gradients_phi @ delta_metric_grad_phi
        K_phiu[:, j] = weight * (delta_q * fluxes + defect * delta_flux)

    return SDFDisplacementCouplingPointResult(K_phiu_point=K_phiu)


__all__ = [
    "SDFDisplacementCouplingPointInput",
    "SDFDisplacementCouplingPointResult",
    "evaluate_sdf_displacement_coupling_point",
    "linearized_metric_sensitivity_from_shape_gradients",
]
