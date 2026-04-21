#!/usr/bin/env python3
"""Unit tests for the local reinitialization kernels."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.sdf.reinitialize import (
    eikonal_defect,
    reinitialize_element_residual_tangent,
    reinitialize_point_residual,
    reinitialize_point_tangent,
)


def test_point_defect_and_residual_vanish_for_exact_signed_distance_state() -> None:
    grad_phi = np.array([1.0, 0.0], dtype=float)
    test_grad = np.array([0.25, -0.5], dtype=float)
    A = np.eye(2)

    defect = eikonal_defect(grad_phi, A)
    residual = reinitialize_point_residual(
        grad_phi,
        test_grad,
        A,
        phi=0.3,
        phi_target=0.3,
        test_value=1.2,
        beta=4.0,
    )

    assert np.isclose(defect, 0.0)
    assert np.isclose(residual, 0.0)


def test_point_tangent_matches_finite_difference_directional_derivative() -> None:
    grad_phi = np.array([0.8, -0.2], dtype=float)
    test_grad = np.array([0.3, 0.7], dtype=float)
    trial_grad = np.array([-0.4, 0.2], dtype=float)
    A = np.diag([1.5, 0.75])
    phi = 0.6
    phi_target = 0.1
    test_value = -0.5
    trial_value = 0.4
    beta = 2.0
    eps = 1.0e-7

    tangent = reinitialize_point_tangent(
        grad_phi,
        test_grad,
        trial_grad,
        A,
        test_value=test_value,
        trial_value=trial_value,
        beta=beta,
    )

    def residual_at(step: float) -> float:
        return reinitialize_point_residual(
            grad_phi + step * trial_grad,
            test_grad,
            A,
            phi=phi + step * trial_value,
            phi_target=phi_target,
            test_value=test_value,
            beta=beta,
        )

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.isclose(tangent, fd, rtol=1.0e-6, atol=1.0e-8)


def test_element_kernel_has_zero_residual_for_exact_affine_distance_field() -> None:
    phi_local = np.array([0.0, 1.0], dtype=float)
    shape_values = np.array([0.5, 0.5], dtype=float)
    shape_gradients = np.array([[-1.0], [1.0]], dtype=float)
    A = np.array([[1.0]], dtype=float)

    residual, tangent = reinitialize_element_residual_tangent(
        phi_local,
        shape_values,
        shape_gradients,
        A,
        phi_target=0.5,
        beta=0.0,
    )

    assert np.allclose(residual, np.zeros(2))
    expected_tangent = 2.0 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    assert np.allclose(tangent, expected_tangent)


def test_element_tangent_matches_finite_difference_of_element_residual() -> None:
    phi_local = np.array([0.15, 0.45], dtype=float)
    shape_values = np.array([0.35, 0.65], dtype=float)
    shape_gradients = np.array([[-1.0], [1.0]], dtype=float)
    phi_target_local = np.array([0.05, 0.25], dtype=float)
    A = np.array([[1.7]], dtype=float)
    beta = 1.3
    weight = 0.75
    eps = 1.0e-7

    residual, tangent = reinitialize_element_residual_tangent(
        phi_local,
        shape_values,
        shape_gradients,
        A,
        phi_target=phi_target_local,
        beta=beta,
        weight=weight,
    )

    fd_columns = []
    for col in range(phi_local.shape[0]):
        direction = np.zeros_like(phi_local)
        direction[col] = 1.0
        residual_plus, _ = reinitialize_element_residual_tangent(
            phi_local + eps * direction,
            shape_values,
            shape_gradients,
            A,
            phi_target=phi_target_local,
            beta=beta,
            weight=weight,
        )
        residual_minus, _ = reinitialize_element_residual_tangent(
            phi_local - eps * direction,
            shape_values,
            shape_gradients,
            A,
            phi_target=phi_target_local,
            beta=beta,
            weight=weight,
        )
        fd_columns.append((residual_plus - residual_minus) / (2.0 * eps))

    fd_tangent = np.column_stack(fd_columns)
    assert np.allclose(tangent, fd_tangent, rtol=1.0e-6, atol=1.0e-8)
    assert residual.shape == (2,)
    assert tangent.shape == (2, 2)
