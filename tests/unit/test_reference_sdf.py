#!/usr/bin/env python3
"""Unit tests for the reference SDF plane helper."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.sdf.reference import ReferencePlane


def test_plane_interface_has_zero_signed_distance() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 1.0], unit_normal=[0.0, 0.0, 2.0])
    point = np.array([1.5, -3.0, 1.0], dtype=float)

    assert plane.phi0(point) == 0.0
    assert plane.sign(point) == 0


def test_nearest_point_lies_on_plane() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 0.0, 1.0])
    point = np.array([2.0, -1.0, 3.0], dtype=float)
    projected = plane.nearest_point(point)

    assert np.allclose(projected, [2.0, -1.0, 0.0])
    assert np.isclose(plane.phi0(projected), 0.0)


def test_normals_are_unit_length_for_single_and_batch_inputs() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 3.0, 4.0])
    points = np.array([[0.0, 1.0, 0.0], [1.0, -2.0, 2.0]], dtype=float)

    normal_single = plane.normal_at_nearest_point(points[0])
    normal_batch = plane.normal_at_nearest_point(points)

    assert np.isclose(np.linalg.norm(normal_single), 1.0)
    assert np.allclose(np.linalg.norm(normal_batch, axis=1), 1.0)


def test_narrow_band_classification_matches_strict_distance_threshold() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[1.0, 0.0, 0.0])
    points = np.array([[-0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=float)

    inside = plane.in_narrow_band(points, delta=0.5)

    assert np.array_equal(inside, np.array([True, False, False]))


def test_sign_convention_is_consistent_for_positive_negative_and_zero_distance() -> None:
    plane = ReferencePlane(point_on_interface=[0.0, 0.0, 0.0], unit_normal=[0.0, 0.0, 1.0])
    points = np.array(
        [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.5],
        ],
        dtype=float,
    )

    phi = plane.phi0(points)
    sign = plane.sign(points)

    assert np.array_equal(sign, np.array([-1, 0, 1]))
    assert np.all(sign[phi < 0.0] == -1)
    assert np.all(sign[phi > 0.0] == 1)
