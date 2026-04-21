#!/usr/bin/env python3
"""Unit tests for the minimal second-order contact geometry foundation."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.contact.geometry import AffineMasterMap2D, AffinePhiField2D
from deformsdfcontact.contact.geometry_second_order import (
    QuadraticPhiField2D,
    evaluate_contact_second_order_geometry,
)


def test_quadratic_phi_field_hessian_matches_analytic_definition() -> None:
    phi_field = QuadraticPhiField2D(
        offset=0.25,
        linear_vector=[1.0, -2.0],
        hessian_matrix=[[3.0, 1.5], [1.5, -4.0]],
    )
    point = np.array([0.2, -0.4], dtype=float)

    assert np.allclose(phi_field.hessian(point), [[3.0, 1.5], [1.5, -4.0]])
    assert np.allclose(phi_field.gradient(point), [1.0, -2.0] + phi_field.hessian(point) @ point)


def test_Huu_curvature_matches_gap_second_difference_when_grad_phi_vanishes() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = QuadraticPhiField2D(
        offset=0.0,
        linear_vector=[0.0, 0.0],
        hessian_matrix=[[2.5, 0.0], [0.0, 0.0]],
    )
    x_slave = np.array([0.0, 1.0], dtype=float)
    delta = np.array([0.7, -0.4, 0.3, 0.5], dtype=float)
    eps = 1.0e-7

    result = evaluate_contact_second_order_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_gradients_at_query=np.array([[1.0], [0.0]], dtype=float),
    )

    def gap_at(step: float) -> float:
        shifted = master_map.with_parameter_vector(master_map.parameter_vector + step * delta)
        return phi_field.value(shifted.inverse_query(x_slave))

    second_difference = (gap_at(eps) - 2.0 * gap_at(0.0) + gap_at(-eps)) / (eps * eps)

    assert np.allclose(result.grad_phi_at_query, [0.0, 0.0])
    assert np.allclose(result.H_uu_query_acceleration, np.zeros((4, 4)))
    assert np.isclose(delta @ result.H_uu_g @ delta, second_difference, rtol=1.0e-5, atol=1.0e-7)


def test_Huphi_matches_E_transpose_shape_gradient_definition() -> None:
    master_map = AffineMasterMap2D(origin=[0.1, -0.2], tangent=[1.2, 0.7])
    phi_field = QuadraticPhiField2D(
        offset=0.0,
        linear_vector=[0.5, -0.3],
        hessian_matrix=[[1.0, 0.2], [0.2, 0.5]],
    )
    x_slave = np.array([0.8, 0.4], dtype=float)
    B_phi = np.array([[1.0, -0.5, 0.25], [0.2, 0.4, -0.1]], dtype=float)

    result = evaluate_contact_second_order_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_gradients_at_query=B_phi,
    )

    assert result.H_uphi_g is not None
    assert np.allclose(result.H_uphi_g, result.E.T @ B_phi)


def test_Huu_applied_to_direction_matches_finite_difference_of_Gu() -> None:
    master_map = AffineMasterMap2D(origin=[0.2, -0.1], tangent=[1.4, 0.6])
    phi_field = QuadraticPhiField2D(
        offset=-0.3,
        linear_vector=[1.1, -0.7],
        hessian_matrix=[[0.8, 0.15], [0.15, -0.2]],
    )
    x_slave = np.array([1.0, 0.5], dtype=float)
    delta = np.array([0.3, -0.2, 0.4, 0.1], dtype=float)
    eps = 1.0e-7

    result = evaluate_contact_second_order_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_gradients_at_query=np.array([[1.0], [0.0]], dtype=float),
    )

    def Gu_at(step: float) -> np.ndarray:
        shifted = master_map.with_parameter_vector(master_map.parameter_vector + step * delta)
        return evaluate_contact_second_order_geometry(
            x_slave,
            shifted,
            phi_field,
            shape_gradients_at_query=np.array([[1.0], [0.0]], dtype=float),
        ).grad_phi_at_query @ evaluate_contact_second_order_geometry(
            x_slave,
            shifted,
            phi_field,
            shape_gradients_at_query=np.array([[1.0], [0.0]], dtype=float),
        ).E

    fd_directional_derivative = (Gu_at(eps) - Gu_at(-eps)) / (2.0 * eps)
    assert np.allclose(result.H_uu_g @ delta, fd_directional_derivative, rtol=1.0e-5, atol=1.0e-7)


def test_affine_phi_field_has_zero_curvature_term() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = AffinePhiField2D(offset=0.2, gradient_vector=[1.5, -0.4])
    x_slave = np.array([0.3, 0.8], dtype=float)

    result = evaluate_contact_second_order_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_gradients_at_query=np.array([[0.5, -0.1], [0.0, 0.2]], dtype=float),
    )

    assert np.allclose(result.hessian_phi_at_query, np.zeros((2, 2)))
    assert np.allclose(result.H_uu_curvature, np.zeros((4, 4)))
