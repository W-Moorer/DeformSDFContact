#!/usr/bin/env python3
"""Unit tests for local SDF displacement-coupling kernels and loops."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.sdf import (
    SDFDisplacementCouplingPointInput,
    build_sdf_coupling_element_mapping,
    evaluate_sdf_displacement_coupling_point,
    execute_sdf_coupling_local_loop,
    linearized_metric_sensitivity_from_shape_gradients,
    reinitialize_element_residual_tangent,
)


def _shape_gradients_triangle() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def test_linearized_metric_sensitivity_has_expected_reference_state_structure() -> None:
    gradients = _shape_gradients_triangle()
    dA_du = linearized_metric_sensitivity_from_shape_gradients(gradients)

    assert dA_du.shape == (6, 2, 2)
    assert np.allclose(dA_du[0], [[2.0, 1.0], [1.0, 0.0]])
    assert np.allclose(dA_du[1], [[0.0, 1.0], [1.0, 2.0]])


def test_sdf_coupling_point_matches_finite_difference_of_point_residual() -> None:
    shape_gradients_phi = _shape_gradients_triangle()
    phi_local = np.array([0.2, -0.1, 0.5], dtype=float)
    grad_phi = shape_gradients_phi.T @ phi_local
    A0 = np.eye(2)
    dA_du = linearized_metric_sensitivity_from_shape_gradients(shape_gradients_phi)
    point_result = evaluate_sdf_displacement_coupling_point(
        SDFDisplacementCouplingPointInput(
            grad_phi=grad_phi,
            shape_gradients_phi=shape_gradients_phi,
            A=A0,
            dA_du=dA_du,
            weight=0.75,
        )
    )
    direction = np.array([0.3, -0.2, 0.4, 0.1, -0.5, 0.2], dtype=float)
    eps = 1.0e-7

    def residual_at(step: float) -> np.ndarray:
        A = A0 + np.tensordot(direction * step, dA_du, axes=(0, 0))
        residual, _ = reinitialize_element_residual_tangent(
            phi_local,
            np.ones(3, dtype=float) / 3.0,
            shape_gradients_phi,
            A,
            phi_target=0.0,
            beta=0.0,
            weight=0.75,
        )
        return residual

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.allclose(point_result.K_phiu_point @ direction, fd, rtol=1.0e-6, atol=1.0e-8)


def test_sdf_coupling_local_loop_matches_finite_difference_of_element_residual() -> None:
    shape_gradients_phi = _shape_gradients_triangle()
    phi_local = np.array([0.15, 0.45, -0.05], dtype=float)
    A0 = np.eye(2)
    dA_du = linearized_metric_sensitivity_from_shape_gradients(shape_gradients_phi)
    weights = np.array([0.4, 0.6], dtype=float)
    mapping = build_sdf_coupling_element_mapping(
        phi_local,
        np.stack([shape_gradients_phi, 1.1 * shape_gradients_phi], axis=0),
        A0,
        dA_du,
        weights,
    )
    loop_result = execute_sdf_coupling_local_loop(mapping)
    direction = np.array([0.2, -0.3, 0.1, 0.4, -0.2, 0.5], dtype=float)
    eps = 1.0e-7

    def residual_at(step: float) -> np.ndarray:
        residual_total = np.zeros(3, dtype=float)
        for q, scale in enumerate((1.0, 1.1)):
            grads = scale * shape_gradients_phi
            A = A0 + np.tensordot(direction * step, dA_du, axes=(0, 0))
            residual_q, _ = reinitialize_element_residual_tangent(
                phi_local,
                np.ones(3, dtype=float) / 3.0,
                grads,
                A,
                phi_target=0.0,
                beta=0.0,
                weight=weights[q],
            )
            residual_total += residual_q
        return residual_total

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.allclose(loop_result.local_K_phiu @ direction, fd, rtol=1.0e-6, atol=1.0e-8)


def test_sdf_coupling_local_loop_sums_multiple_quadrature_points() -> None:
    shape_gradients_phi = _shape_gradients_triangle()
    phi_local = np.array([0.1, -0.2, 0.4], dtype=float)
    A0 = np.eye(2)
    dA_du = linearized_metric_sensitivity_from_shape_gradients(shape_gradients_phi)
    mapping = build_sdf_coupling_element_mapping(
        phi_local,
        np.stack([shape_gradients_phi, 1.3 * shape_gradients_phi], axis=0),
        A0,
        dA_du,
        np.array([0.25, 0.75], dtype=float),
    )
    loop_result = execute_sdf_coupling_local_loop(mapping)
    point0 = evaluate_sdf_displacement_coupling_point(
        SDFDisplacementCouplingPointInput(
            grad_phi=shape_gradients_phi.T @ phi_local,
            shape_gradients_phi=shape_gradients_phi,
            A=A0,
            dA_du=dA_du,
            weight=0.25,
        )
    )
    point1 = evaluate_sdf_displacement_coupling_point(
        SDFDisplacementCouplingPointInput(
            grad_phi=(1.3 * shape_gradients_phi).T @ phi_local,
            shape_gradients_phi=1.3 * shape_gradients_phi,
            A=A0,
            dA_du=dA_du,
            weight=0.75,
        )
    )

    assert np.allclose(loop_result.local_K_phiu, point0.K_phiu_point + point1.K_phiu_point)
