#!/usr/bin/env python3
"""Unit tests for backend-agnostic local contact point kernels."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.contact.kernels import (
    ContactPointKernelInput,
    evaluate_contact_point_kernel,
)
from deformsdfcontact.contact.local_loop import execute_contact_local_loop
from deformsdfcontact.contact.laws import PenaltyContactLaw


def _active_kernel_input() -> ContactPointKernelInput:
    return ContactPointKernelInput(
        g_n=-0.2,
        G_u=np.array([1.0, -2.0], dtype=float),
        G_a=np.array([0.5, -1.0, 0.25], dtype=float),
        H_uu_g=np.array([[0.3, -0.1], [-0.1, 0.4]], dtype=float),
        H_uphi_g=np.array([[0.2, -0.3, 0.1], [0.05, 0.4, -0.2]], dtype=float),
        weight=1.5,
    )


def test_point_kernel_matches_closed_form_residual_and_tangents() -> None:
    law = PenaltyContactLaw(penalty=10.0)
    kernel_input = _active_kernel_input()

    result = evaluate_contact_point_kernel(kernel_input, law)

    lambda_n = 2.0
    k_n = 10.0
    expected_residual = 1.5 * lambda_n * np.array([1.0, -2.0], dtype=float)
    expected_K_uu = 1.5 * (
        k_n * np.outer([1.0, -2.0], [1.0, -2.0])
        + lambda_n * np.array([[0.3, -0.1], [-0.1, 0.4]], dtype=float)
    )
    expected_K_uphi = 1.5 * (
        k_n * np.outer([1.0, -2.0], [0.5, -1.0, 0.25])
        + lambda_n * np.array([[0.2, -0.3, 0.1], [0.05, 0.4, -0.2]], dtype=float)
    )

    assert result.lambda_n == lambda_n
    assert result.k_n == k_n
    assert np.allclose(result.r_u_c, expected_residual)
    assert np.allclose(result.K_uu_c, expected_K_uu)
    assert np.allclose(result.K_uphi_c, expected_K_uphi)


def test_inactive_contact_point_kernel_returns_zero_contributions() -> None:
    law = PenaltyContactLaw(penalty=10.0)
    kernel_input = ContactPointKernelInput(
        g_n=0.1,
        G_u=np.array([1.0, -2.0], dtype=float),
        G_a=np.array([0.5, -1.0], dtype=float),
        H_uu_g=np.array([[0.3, -0.1], [-0.1, 0.4]], dtype=float),
        H_uphi_g=np.array([[0.2, -0.3], [0.05, 0.4]], dtype=float),
        weight=2.0,
    )

    result = evaluate_contact_point_kernel(kernel_input, law)

    assert result.lambda_n == 0.0
    assert result.k_n == 0.0
    assert np.allclose(result.r_u_c, np.zeros(2))
    assert np.allclose(result.K_uu_c, np.zeros((2, 2)))
    assert np.allclose(result.K_uphi_c, np.zeros((2, 2)))


def test_geometry_second_order_terms_are_explicitly_used() -> None:
    law = PenaltyContactLaw(penalty=10.0)
    kernel_input = _active_kernel_input()

    result = evaluate_contact_point_kernel(kernel_input, law)
    lambda_n = result.lambda_n
    pure_penalty_uu = kernel_input.weight * result.k_n * np.outer(kernel_input.G_u, kernel_input.G_u)
    pure_penalty_uphi = kernel_input.weight * result.k_n * np.outer(kernel_input.G_u, kernel_input.G_a)

    assert not np.allclose(result.K_uu_c, pure_penalty_uu)
    assert not np.allclose(result.K_uphi_c, pure_penalty_uphi)
    assert np.allclose(result.K_uu_c - pure_penalty_uu, kernel_input.weight * lambda_n * np.asarray(kernel_input.H_uu_g))
    assert np.allclose(result.K_uphi_c - pure_penalty_uphi, kernel_input.weight * lambda_n * np.asarray(kernel_input.H_uphi_g))


def test_Kuu_matches_finite_difference_of_local_residual_along_geometry_direction() -> None:
    law = PenaltyContactLaw(penalty=8.0)
    kernel_input = ContactPointKernelInput(
        g_n=-0.3,
        G_u=np.array([0.8, -0.4], dtype=float),
        G_a=np.array([0.2, -0.1], dtype=float),
        H_uu_g=np.array([[0.5, 0.1], [0.1, -0.2]], dtype=float),
        H_uphi_g=np.zeros((2, 2), dtype=float),
        weight=1.2,
    )
    direction = np.array([0.4, -0.6], dtype=float)
    eps = 1.0e-7

    base_result = evaluate_contact_point_kernel(kernel_input, law)

    def residual_at(step: float) -> np.ndarray:
        g_step = (
            kernel_input.g_n
            - step * (np.asarray(kernel_input.G_u) @ direction)
            - 0.5 * step * step * (direction @ np.asarray(kernel_input.H_uu_g) @ direction)
        )
        G_u_step = np.asarray(kernel_input.G_u) + step * (np.asarray(kernel_input.H_uu_g) @ direction)
        stepped_input = ContactPointKernelInput(
            g_n=g_step,
            G_u=G_u_step,
            G_a=kernel_input.G_a,
            H_uu_g=kernel_input.H_uu_g,
            H_uphi_g=kernel_input.H_uphi_g,
            weight=kernel_input.weight,
        )
        return evaluate_contact_point_kernel(stepped_input, law).r_u_c

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.allclose(base_result.K_uu_c @ direction, fd, rtol=1.0e-6, atol=1.0e-8)


def test_Kuphi_matches_finite_difference_of_local_residual_along_phi_direction() -> None:
    law = PenaltyContactLaw(penalty=9.0)
    kernel_input = ContactPointKernelInput(
        g_n=-0.25,
        G_u=np.array([1.2, -0.7], dtype=float),
        G_a=np.array([0.6, -0.2, 0.3], dtype=float),
        H_uu_g=np.zeros((2, 2), dtype=float),
        H_uphi_g=np.array([[0.1, -0.4, 0.2], [0.05, 0.3, -0.1]], dtype=float),
        weight=0.8,
    )
    direction = np.array([0.2, -0.5, 0.4], dtype=float)
    eps = 1.0e-7

    base_result = evaluate_contact_point_kernel(kernel_input, law)

    def residual_at(step: float) -> np.ndarray:
        g_step = kernel_input.g_n - step * (np.asarray(kernel_input.G_a) @ direction)
        G_u_step = np.asarray(kernel_input.G_u) + step * (np.asarray(kernel_input.H_uphi_g) @ direction)
        stepped_input = ContactPointKernelInput(
            g_n=g_step,
            G_u=G_u_step,
            G_a=kernel_input.G_a,
            H_uu_g=kernel_input.H_uu_g,
            H_uphi_g=kernel_input.H_uphi_g,
            weight=kernel_input.weight,
        )
        return evaluate_contact_point_kernel(stepped_input, law).r_u_c

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.allclose(base_result.K_uphi_c @ direction, fd, rtol=1.0e-6, atol=1.0e-8)


def test_local_loop_accumulates_point_contributions_by_summation() -> None:
    law = PenaltyContactLaw(penalty=6.0)
    inputs = [
        ContactPointKernelInput(
            g_n=-0.1,
            G_u=np.array([1.0, 0.5], dtype=float),
            G_a=np.array([0.2, -0.3], dtype=float),
            H_uu_g=np.array([[0.1, 0.0], [0.0, 0.2]], dtype=float),
            H_uphi_g=np.array([[0.05, 0.1], [-0.02, 0.03]], dtype=float),
            weight=0.4,
        ),
        ContactPointKernelInput(
            g_n=-0.05,
            G_u=np.array([-0.4, 0.8], dtype=float),
            G_a=np.array([0.2, -0.3], dtype=float),
            H_uu_g=np.array([[0.0, -0.1], [-0.1, 0.3]], dtype=float),
            H_uphi_g=np.array([[0.02, -0.01], [0.04, 0.05]], dtype=float),
            weight=0.7,
        ),
    ]

    loop_result = execute_contact_local_loop(inputs, law)
    point_results = [evaluate_contact_point_kernel(point_input, law) for point_input in inputs]

    expected_residual = sum((result.r_u_c for result in point_results), start=np.zeros(2, dtype=float))
    expected_K_uu = sum((result.K_uu_c for result in point_results), start=np.zeros((2, 2), dtype=float))
    expected_K_uphi = sum((result.K_uphi_c for result in point_results), start=np.zeros((2, 2), dtype=float))

    assert np.allclose(loop_result.local_residual, expected_residual)
    assert np.allclose(loop_result.local_K_uu, expected_K_uu)
    assert np.allclose(loop_result.local_K_uphi, expected_K_uphi)
