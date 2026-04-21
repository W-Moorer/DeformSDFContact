#!/usr/bin/env python3
"""Unit tests for the metric-stretch SDF warm-start predictor."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.sdf.predictor import (
    metric_stretch_factor,
    predict_from_reference_geometry,
    predict_pullback_distance,
)
from deformsdfcontact.sdf.reference import ReferencePlane


def _metric_from_F(F: np.ndarray) -> np.ndarray:
    C = F.T @ F
    return np.linalg.inv(C)


def test_rigid_motion_identity_metric_returns_reference_distance() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 0.0, 1.0])
    point = np.array([1.0, -2.0, 0.4], dtype=float)

    result = predict_from_reference_geometry(point, plane, np.eye(3))

    assert np.isclose(result.phi0, 0.4)
    assert np.isclose(result.stretch_factor, 1.0)
    assert np.isclose(result.phi_pred, result.phi0)


def test_uniaxial_stretch_scales_distance_by_theoretical_factor() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[1.0, 0.0, 0.0])
    point = np.array([0.25, 0.0, 0.0], dtype=float)
    stretch = 1.4
    F = np.diag([stretch, 1.0, 1.0])
    A = _metric_from_F(F)

    factor = metric_stretch_factor(np.array([1.0, 0.0, 0.0]), A)
    predicted = predict_from_reference_geometry(point, plane, A)

    assert np.isclose(factor, 1.0 / stretch)
    assert np.isclose(predicted.phi_pred, stretch * predicted.phi0)


def test_pure_shear_preserves_zero_level_set_and_sign() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 1.0, 0.0])
    gamma = 0.6
    F = np.array(
        [
            [1.0, gamma, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    A = _metric_from_F(F)
    points = np.array(
        [
            [0.5, -0.2, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.3, -0.1],
        ],
        dtype=float,
    )

    result = predict_from_reference_geometry(points, plane, A)

    assert np.isclose(result.phi_pred[1], 0.0)
    assert np.array_equal(np.sign(result.phi_pred), np.sign(result.phi0))


def test_batch_inputs_work_for_points_normals_and_metric_tensors() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 0.0, 1.0])
    points = np.array(
        [
            [0.0, 0.0, -0.5],
            [1.0, 2.0, 0.0],
            [-1.0, 0.5, 0.75],
        ],
        dtype=float,
    )
    stretch = 1.25
    F = np.diag([1.0, 1.0, stretch])
    A = _metric_from_F(F)
    expected = plane.phi0(points) * stretch

    predicted = predict_from_reference_geometry(points, plane, A)
    direct = predict_pullback_distance(plane.phi0(points), plane.normal_at_nearest_point(points), A)

    assert predicted.nearest_point.shape == points.shape
    assert predicted.normal.shape == points.shape
    assert predicted.phi0.shape == (3,)
    assert np.allclose(predicted.phi_pred, expected)
    assert np.allclose(direct, expected)
