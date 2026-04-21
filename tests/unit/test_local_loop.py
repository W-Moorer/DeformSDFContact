#!/usr/bin/env python3
"""Unit tests for the backend-agnostic reinitialize local loop."""

from __future__ import annotations

import numpy as np
import pytest

import deformsdfcontact.sdf.local_loop as local_loop
from deformsdfcontact.sdf.form_mapping import (
    ReinitializeElementMapping,
    ReinitializeQuadraturePointData,
    build_reinitialize_element_mapping,
)
from deformsdfcontact.sdf.reinitialize import reinitialize_element_residual_tangent


def test_local_loop_matches_single_quadrature_point_element_kernel() -> None:
    phi_local = np.array([0.15, 0.45], dtype=float)
    shape_values = np.array([[0.35, 0.65]], dtype=float)
    shape_gradients = np.array([[[-1.0], [1.0]]], dtype=float)
    phi_target_local = np.array([0.05, 0.25], dtype=float)
    A = np.array([[1.7]], dtype=float)
    beta = 1.3
    weight = np.array([0.75], dtype=float)

    mapping = build_reinitialize_element_mapping(
        phi_local,
        shape_values,
        shape_gradients,
        A,
        weight,
        phi_target=phi_target_local,
        beta=beta,
    )
    residual, tangent = local_loop.execute_reinitialize_local_loop(mapping)

    expected_residual, expected_tangent = reinitialize_element_residual_tangent(
        phi_local,
        shape_values[0],
        shape_gradients[0],
        A,
        phi_target=phi_target_local,
        beta=beta,
        weight=weight[0],
    )

    assert np.allclose(residual, expected_residual)
    assert np.allclose(tangent, expected_tangent)


def test_local_loop_sums_multiple_quadrature_points_correctly() -> None:
    phi_local = np.array([0.0, 1.0], dtype=float)
    shape_values = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=float)
    shape_gradients = np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float)
    A = np.array([[[1.0]], [[1.5]]], dtype=float)
    weights = np.array([0.4, 0.6], dtype=float)
    beta = np.array([0.0, 2.5], dtype=float)
    phi_target = np.array([0.0, 2.0], dtype=float)

    mapping = build_reinitialize_element_mapping(
        phi_local,
        shape_values,
        shape_gradients,
        A,
        weights,
        phi_target=phi_target,
        beta=beta,
    )
    residual, tangent = local_loop.execute_reinitialize_local_loop(mapping)

    expected_residual = np.zeros(2, dtype=float)
    expected_tangent = np.zeros((2, 2), dtype=float)
    for quadrature_point in mapping.quadrature_points:
        point_residual, point_tangent = reinitialize_element_residual_tangent(
            phi_local,
            quadrature_point.shape_values,
            quadrature_point.shape_gradients,
            quadrature_point.A,
            phi_target=quadrature_point.phi_target,
            beta=quadrature_point.beta,
            weight=quadrature_point.weight,
        )
        expected_residual += point_residual
        expected_tangent += point_tangent

    assert np.allclose(residual, expected_residual)
    assert np.allclose(tangent, expected_tangent)


def test_local_loop_calls_point_kernels_for_each_test_and_trial_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = build_reinitialize_element_mapping(
        phi_local=np.array([0.0, 1.0], dtype=float),
        shape_values=np.array([[0.5, 0.5], [0.25, 0.75]], dtype=float),
        shape_gradients=np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float),
        A=np.array([[1.0]], dtype=float),
        weights=np.array([0.5, 0.5], dtype=float),
        beta=1.0,
    )

    residual_calls = 0
    tangent_calls = 0
    original_residual = local_loop.reinitialize_point_residual
    original_tangent = local_loop.reinitialize_point_tangent

    def counted_residual(*args, **kwargs):
        nonlocal residual_calls
        residual_calls += 1
        return original_residual(*args, **kwargs)

    def counted_tangent(*args, **kwargs):
        nonlocal tangent_calls
        tangent_calls += 1
        return original_tangent(*args, **kwargs)

    monkeypatch.setattr(local_loop, "reinitialize_point_residual", counted_residual)
    monkeypatch.setattr(local_loop, "reinitialize_point_tangent", counted_tangent)

    local_loop.execute_reinitialize_local_loop(mapping)

    assert residual_calls == mapping.nqp * mapping.nshape
    assert tangent_calls == mapping.nqp * mapping.nshape * mapping.nshape


def test_local_loop_rejects_inconsistent_mapping_shapes() -> None:
    bad_mapping = ReinitializeElementMapping(
        phi_local=np.array([0.0, 1.0], dtype=float),
        quadrature_points=(
            ReinitializeQuadraturePointData(
                phi=0.5,
                grad_phi=np.array([1.0], dtype=float),
                phi_target=0.0,
                shape_values=np.array([0.5, 0.5, 0.0], dtype=float),
                shape_gradients=np.array([[-1.0], [1.0], [0.0]], dtype=float),
                A=np.array([[1.0]], dtype=float),
                weight=1.0,
                beta=0.0,
            ),
        ),
    )

    with pytest.raises(ValueError, match="shape_values"):
        local_loop.execute_reinitialize_local_loop(bad_mapping)
