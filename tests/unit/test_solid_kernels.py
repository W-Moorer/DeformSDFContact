#!/usr/bin/env python3
"""Unit tests for backend-agnostic solid local kernels."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.materials import IsotropicElasticParameters
from deformsdfcontact.solid import (
    SolidPointKernelInput,
    build_solid_element_mapping,
    evaluate_solid_point_kernel,
    execute_solid_local_loop,
    plane_strain_constitutive_matrix,
    triangle_p1_B_matrix,
)


def _triangle_shape_gradients() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def test_rigid_translation_and_infinitesimal_rotation_give_zero_internal_residual() -> None:
    params = IsotropicElasticParameters(E=10.0, nu=0.25)
    B = triangle_p1_B_matrix(_triangle_shape_gradients())
    translation = np.array([0.3, -0.4, 0.3, -0.4, 0.3, -0.4], dtype=float)
    rotation = np.array([0.0, 0.0, 0.0, 0.2, -0.2, 0.0], dtype=float)

    for u_local in (translation, rotation):
        mapping = build_solid_element_mapping(u_local, B, np.array([0.5], dtype=float))
        result = execute_solid_local_loop(mapping, params)
        assert np.allclose(result.local_residual_u, np.zeros(6))


def test_solid_point_kernel_matches_closed_form_linear_elasticity() -> None:
    params = IsotropicElasticParameters(E=12.0, nu=0.2)
    B = triangle_p1_B_matrix(_triangle_shape_gradients())
    strain = np.array([0.1, -0.05, 0.2], dtype=float)
    weight = 0.75

    result = evaluate_solid_point_kernel(
        SolidPointKernelInput(strain=strain, B=B, weight=weight),
        params,
    )
    C = plane_strain_constitutive_matrix(params)
    sigma = C @ strain

    assert np.allclose(result.stress, sigma)
    assert np.allclose(result.r_u_int, weight * (B.T @ sigma))
    assert np.allclose(result.K_uu_int, weight * (B.T @ C @ B))


def test_solid_tangent_matches_finite_difference_of_local_residual() -> None:
    params = IsotropicElasticParameters(E=15.0, nu=0.3)
    B = triangle_p1_B_matrix(_triangle_shape_gradients())
    u_local = np.array([0.1, -0.05, 0.08, 0.03, -0.02, 0.04], dtype=float)
    direction = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2], dtype=float)
    weight = np.array([0.5], dtype=float)
    eps = 1.0e-7

    mapping = build_solid_element_mapping(u_local, B, weight)
    base_result = execute_solid_local_loop(mapping, params)

    def residual_at(step: float) -> np.ndarray:
        stepped = build_solid_element_mapping(u_local + step * direction, B, weight)
        return execute_solid_local_loop(stepped, params).local_residual_u

    fd = (residual_at(eps) - residual_at(-eps)) / (2.0 * eps)
    assert np.allclose(base_result.local_K_uu @ direction, fd, rtol=1.0e-6, atol=1.0e-8)


def test_solid_local_loop_sums_multiple_quadrature_points_correctly() -> None:
    params = IsotropicElasticParameters(E=20.0, nu=0.25)
    B0 = triangle_p1_B_matrix(_triangle_shape_gradients())
    B1 = 1.2 * B0
    u_local = np.array([0.05, -0.02, 0.04, 0.01, -0.01, 0.03], dtype=float)
    mapping = build_solid_element_mapping(
        u_local,
        np.stack([B0, B1], axis=0),
        np.array([0.3, 0.7], dtype=float),
    )

    loop_result = execute_solid_local_loop(mapping, params)
    point0 = evaluate_solid_point_kernel(
        SolidPointKernelInput(strain=B0 @ u_local, B=B0, weight=0.3),
        params,
    )
    point1 = evaluate_solid_point_kernel(
        SolidPointKernelInput(strain=B1 @ u_local, B=B1, weight=0.7),
        params,
    )

    assert np.allclose(loop_result.local_residual_u, point0.r_u_int + point1.r_u_int)
    assert np.allclose(loop_result.local_K_uu, point0.K_uu_int + point1.K_uu_int)
