"""Pure pointwise and element-local kernels for SDF reinitialization."""

from __future__ import annotations

import numpy as np


def _as_vector(values: object, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    if length is not None and array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _as_shape_gradients(values: object, nshape: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(
            "shape_gradients must have shape (nshape, dim), "
            f"got {array.shape!r}"
        )
    if nshape is not None and array.shape[0] != nshape:
        raise ValueError(
            f"shape_gradients must have shape ({nshape}, dim), got {array.shape!r}"
        )
    return array


def _as_metric_tensor(values: object, dimension: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (dimension, dimension):
        raise ValueError(
            f"A must have shape ({dimension}, {dimension}), got {array.shape!r}"
        )
    return 0.5 * (array + array.T)


def _point_target_value(phi_target: object, shape_values: np.ndarray) -> float:
    target = np.asarray(phi_target, dtype=float)
    if target.ndim == 0:
        return float(target)
    if target.ndim == 1 and target.shape[0] == shape_values.shape[0]:
        return float(shape_values @ target)
    raise ValueError(
        "phi_target must be a scalar or have shape (nshape,), "
        f"got {target.shape!r}"
    )


def eikonal_defect(grad_phi: object, A: object) -> float:
    """Return the pointwise pull-back eikonal defect `grad(phi)^T A grad(phi) - 1`."""

    grad = _as_vector(grad_phi, "grad_phi")
    metric = _as_metric_tensor(A, grad.shape[0])
    return float(grad @ metric @ grad - 1.0)


def reinitialize_point_residual(
    grad_phi: object,
    test_grad: object,
    A: object,
    *,
    phi: float = 0.0,
    phi_target: float = 0.0,
    test_value: float = 0.0,
    beta: float = 0.0,
    weight: float = 1.0,
) -> float:
    """Return the pointwise residual contribution for one test function."""

    grad = _as_vector(grad_phi, "grad_phi")
    test_gradient = _as_vector(test_grad, "test_grad", length=grad.shape[0])
    metric = _as_metric_tensor(A, grad.shape[0])
    defect = float(grad @ metric @ grad - 1.0)
    directional_flux = float(test_gradient @ metric @ grad)
    mismatch = float(phi) - float(phi_target)
    return float(weight) * (
        defect * directional_flux + float(beta) * mismatch * float(test_value)
    )


def reinitialize_point_tangent(
    grad_phi: object,
    test_grad: object,
    trial_grad: object,
    A: object,
    *,
    test_value: float = 0.0,
    trial_value: float = 0.0,
    beta: float = 0.0,
    weight: float = 1.0,
) -> float:
    """Return the pointwise tangent contribution for one test/trial pair."""

    grad = _as_vector(grad_phi, "grad_phi")
    test_gradient = _as_vector(test_grad, "test_grad", length=grad.shape[0])
    trial_gradient = _as_vector(trial_grad, "trial_grad", length=grad.shape[0])
    metric = _as_metric_tensor(A, grad.shape[0])
    defect = float(grad @ metric @ grad - 1.0)
    test_flux = float(test_gradient @ metric @ grad)
    trial_flux = float(trial_gradient @ metric @ grad)
    metric_pair = float(test_gradient @ metric @ trial_gradient)
    return float(weight) * (
        2.0 * test_flux * trial_flux
        + defect * metric_pair
        + float(beta) * float(test_value) * float(trial_value)
    )


def reinitialize_element_residual_tangent(
    phi_local: object,
    shape_values: object,
    shape_gradients: object,
    A: object,
    *,
    phi_target: object = 0.0,
    beta: float = 0.0,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one quadrature-point element residual and tangent contribution.

    Shape conventions:

    - `phi_local`: `(nshape,)`
    - `shape_values`: `(nshape,)`
    - `shape_gradients`: `(nshape, dim)`
    - `A`: `(dim, dim)`
    - `phi_target`: scalar or `(nshape,)`
    """

    shape_values_array = _as_vector(shape_values, "shape_values")
    nshape = shape_values_array.shape[0]
    local_values = _as_vector(phi_local, "phi_local", length=nshape)
    gradients = _as_shape_gradients(shape_gradients, nshape=nshape)
    metric = _as_metric_tensor(A, gradients.shape[1])

    phi_h = float(shape_values_array @ local_values)
    phi_target_value = _point_target_value(phi_target, shape_values_array)
    grad_phi = gradients.T @ local_values

    metric_grad_phi = metric @ grad_phi
    directional_fluxes = gradients @ metric_grad_phi
    defect = float(grad_phi @ metric_grad_phi - 1.0)
    metric_stiffness = gradients @ metric @ gradients.T
    regularization = float(beta) * (phi_h - phi_target_value) * shape_values_array

    residual = float(weight) * (defect * directional_fluxes + regularization)
    tangent = float(weight) * (
        2.0 * np.outer(directional_fluxes, directional_fluxes)
        + defect * metric_stiffness
        + float(beta) * np.outer(shape_values_array, shape_values_array)
    )
    return residual, tangent


__all__ = [
    "eikonal_defect",
    "reinitialize_element_residual_tangent",
    "reinitialize_point_residual",
    "reinitialize_point_tangent",
]
