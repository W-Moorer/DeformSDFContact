#!/usr/bin/env python3
"""Unit tests for backend-agnostic contact surface local quadrature execution."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.contact.form_mapping import build_contact_surface_mapping
from deformsdfcontact.contact.kernels import ContactPointKernelInput, evaluate_contact_point_kernel
from deformsdfcontact.contact.laws import PenaltyContactLaw
from deformsdfcontact.contact.surface_local_loop import execute_contact_surface_local_loop


def test_single_point_surface_loop_matches_point_kernel() -> None:
    law = PenaltyContactLaw(penalty=10.0)
    mapping = build_contact_surface_mapping(
        g_n=-0.2,
        G_u=np.array([1.0, -2.0], dtype=float),
        G_a=np.array([0.5, -1.0, 0.25], dtype=float),
        H_uu_g=np.array([[0.3, -0.1], [-0.1, 0.4]], dtype=float),
        H_uphi_g=np.array([[0.2, -0.3, 0.1], [0.05, 0.4, -0.2]], dtype=float),
        weights=np.array([1.5], dtype=float),
    )

    loop_result = execute_contact_surface_local_loop(mapping, law)
    point_result = evaluate_contact_point_kernel(
        ContactPointKernelInput(
            g_n=-0.2,
            G_u=np.array([1.0, -2.0], dtype=float),
            G_a=np.array([0.5, -1.0, 0.25], dtype=float),
            H_uu_g=np.array([[0.3, -0.1], [-0.1, 0.4]], dtype=float),
            H_uphi_g=np.array([[0.2, -0.3, 0.1], [0.05, 0.4, -0.2]], dtype=float),
            weight=1.5,
        ),
        law,
    )

    assert np.allclose(loop_result.local_residual_u, point_result.r_u_c)
    assert np.allclose(loop_result.local_K_uu, point_result.K_uu_c)
    assert np.allclose(loop_result.local_K_uphi, point_result.K_uphi_c)


def test_multi_point_surface_loop_matches_explicit_pointwise_sum() -> None:
    law = PenaltyContactLaw(penalty=6.0)
    mapping = build_contact_surface_mapping(
        g_n=np.array([-0.1, -0.05], dtype=float),
        G_u=np.array([[1.0, 0.5], [-0.4, 0.8]], dtype=float),
        G_a=np.array([[0.2, -0.3], [0.2, -0.3]], dtype=float),
        H_uu_g=np.array(
            [
                [[0.1, 0.0], [0.0, 0.2]],
                [[0.0, -0.1], [-0.1, 0.3]],
            ],
            dtype=float,
        ),
        H_uphi_g=np.array(
            [
                [[0.05, 0.1], [-0.02, 0.03]],
                [[0.02, -0.01], [0.04, 0.05]],
            ],
            dtype=float,
        ),
        weights=np.array([0.4, 0.7], dtype=float),
    )

    loop_result = execute_contact_surface_local_loop(mapping, law)
    point_results = [
        evaluate_contact_point_kernel(
            ContactPointKernelInput(
                g_n=point.g_n,
                G_u=point.G_u,
                G_a=point.G_a,
                H_uu_g=point.H_uu_g,
                H_uphi_g=point.H_uphi_g,
                weight=point.weight,
            ),
            law,
        )
        for point in mapping.point_data
    ]

    expected_residual = sum((result.r_u_c for result in point_results), start=np.zeros(2, dtype=float))
    expected_K_uu = sum((result.K_uu_c for result in point_results), start=np.zeros((2, 2), dtype=float))
    expected_K_uphi = sum((result.K_uphi_c for result in point_results), start=np.zeros((2, 2), dtype=float))

    assert np.allclose(loop_result.local_residual_u, expected_residual)
    assert np.allclose(loop_result.local_K_uu, expected_K_uu)
    assert np.allclose(loop_result.local_K_uphi, expected_K_uphi)


def test_inactive_points_do_not_pollute_surface_local_result() -> None:
    law = PenaltyContactLaw(penalty=12.0)
    mapping = build_contact_surface_mapping(
        g_n=np.array([-0.2, 0.1], dtype=float),
        G_u=np.array([[1.0, -2.0], [3.0, 4.0]], dtype=float),
        G_a=np.array([[0.5, -0.25], [1.0, 2.0]], dtype=float),
        H_uu_g=np.array(
            [
                [[0.3, 0.1], [0.1, 0.4]],
                [[9.0, 9.0], [9.0, 9.0]],
            ],
            dtype=float,
        ),
        H_uphi_g=np.array(
            [
                [[0.2, -0.1], [0.05, 0.3]],
                [[7.0, 7.0], [7.0, 7.0]],
            ],
            dtype=float,
        ),
        weights=np.array([1.0, 2.0], dtype=float),
    )

    loop_result = execute_contact_surface_local_loop(mapping, law)
    active_only = evaluate_contact_point_kernel(
        ContactPointKernelInput(
            g_n=-0.2,
            G_u=np.array([1.0, -2.0], dtype=float),
            G_a=np.array([0.5, -0.25], dtype=float),
            H_uu_g=np.array([[0.3, 0.1], [0.1, 0.4]], dtype=float),
            H_uphi_g=np.array([[0.2, -0.1], [0.05, 0.3]], dtype=float),
            weight=1.0,
        ),
        law,
    )

    assert np.allclose(loop_result.local_residual_u, active_only.r_u_c)
    assert np.allclose(loop_result.local_K_uu, active_only.K_uu_c)
    assert np.allclose(loop_result.local_K_uphi, active_only.K_uphi_c)


def test_second_order_geometry_terms_enter_surface_local_result() -> None:
    law = PenaltyContactLaw(penalty=10.0)
    mapping = build_contact_surface_mapping(
        g_n=np.array([-0.2, -0.1], dtype=float),
        G_u=np.array([[1.0, -2.0], [0.5, 1.5]], dtype=float),
        G_a=np.array([[0.5, -1.0, 0.25], [0.2, 0.1, -0.3]], dtype=float),
        H_uu_g=np.array(
            [
                [[0.3, -0.1], [-0.1, 0.4]],
                [[-0.2, 0.05], [0.05, 0.1]],
            ],
            dtype=float,
        ),
        H_uphi_g=np.array(
            [
                [[0.2, -0.3, 0.1], [0.05, 0.4, -0.2]],
                [[0.3, 0.1, -0.2], [-0.1, 0.2, 0.05]],
            ],
            dtype=float,
        ),
        weights=np.array([1.5, 0.75], dtype=float),
    )

    loop_result = execute_contact_surface_local_loop(mapping, law)
    pure_penalty_uu = np.zeros((2, 2), dtype=float)
    pure_penalty_uphi = np.zeros((2, 3), dtype=float)
    for point in mapping.point_data:
        lambda_n, k_n = law.evaluate(point.g_n)
        pure_penalty_uu += point.weight * k_n * np.outer(point.G_u, point.G_u)
        pure_penalty_uphi += point.weight * k_n * np.outer(point.G_u, point.G_a)

    assert not np.allclose(loop_result.local_K_uu, pure_penalty_uu)
    assert not np.allclose(loop_result.local_K_uphi, pure_penalty_uphi)


def test_quadrature_weights_scale_surface_local_result_correctly() -> None:
    law = PenaltyContactLaw(penalty=5.0)
    base_mapping = build_contact_surface_mapping(
        g_n=np.array([-0.1, -0.1], dtype=float),
        G_u=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        G_a=np.array([[0.2], [0.2]], dtype=float),
        H_uu_g=np.array([[[0.1, 0.0], [0.0, 0.0]], [[0.1, 0.0], [0.0, 0.0]]], dtype=float),
        H_uphi_g=np.array([[[0.05], [0.0]], [[0.05], [0.0]]], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )
    scaled_mapping = build_contact_surface_mapping(
        g_n=np.array([-0.1, -0.1], dtype=float),
        G_u=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        G_a=np.array([[0.2], [0.2]], dtype=float),
        H_uu_g=np.array([[[0.1, 0.0], [0.0, 0.0]], [[0.1, 0.0], [0.0, 0.0]]], dtype=float),
        H_uphi_g=np.array([[[0.05], [0.0]], [[0.05], [0.0]]], dtype=float),
        weights=np.array([2.0, 0.5], dtype=float),
    )

    base_result = execute_contact_surface_local_loop(base_mapping, law)
    scaled_result = execute_contact_surface_local_loop(scaled_mapping, law)

    assert not np.allclose(base_result.local_residual_u, scaled_result.local_residual_u)
    assert not np.allclose(base_result.local_K_uu, scaled_result.local_K_uu)
    assert not np.allclose(base_result.local_K_uphi, scaled_result.local_K_uphi)
