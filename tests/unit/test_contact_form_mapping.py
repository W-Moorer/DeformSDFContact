#!/usr/bin/env python3
"""Unit tests for backend-agnostic contact surface form mapping."""

from __future__ import annotations

import numpy as np
import pytest

from deformsdfcontact.contact.form_mapping import build_contact_surface_mapping


def test_single_point_input_maps_to_one_quadrature_point() -> None:
    mapping = build_contact_surface_mapping(
        g_n=-0.2,
        G_u=np.array([1.0, -2.0], dtype=float),
        G_a=np.array([0.5, -0.25, 0.1], dtype=float),
        H_uu_g=np.array([[0.3, 0.1], [0.1, 0.4]], dtype=float),
        H_uphi_g=np.array([[0.2, -0.1, 0.0], [0.05, 0.3, -0.2]], dtype=float),
        weights=np.array([1.25], dtype=float),
    )

    assert mapping.nqp == 1
    assert mapping.ndof_u_local == 2
    assert mapping.nphi_local == 3
    point = mapping.point_data[0]
    assert np.isscalar(point.g_n)
    assert point.G_u.shape == (2,)
    assert point.G_a.shape == (3,)
    assert point.H_uu_g.shape == (2, 2)
    assert point.H_uphi_g.shape == (2, 3)
    assert point.weight == 1.25


def test_multi_point_input_preserves_per_point_values() -> None:
    mapping = build_contact_surface_mapping(
        g_n=np.array([-0.2, 0.1], dtype=float),
        G_u=np.array([[1.0, -2.0], [0.5, 0.25]], dtype=float),
        G_a=np.array([[0.5, -0.25], [0.1, 0.3]], dtype=float),
        H_uu_g=np.array(
            [
                [[0.3, 0.1], [0.1, 0.4]],
                [[-0.2, 0.0], [0.0, 0.6]],
            ],
            dtype=float,
        ),
        H_uphi_g=np.array(
            [
                [[0.2, -0.1], [0.05, 0.3]],
                [[-0.1, 0.0], [0.4, -0.2]],
            ],
            dtype=float,
        ),
        weights=np.array([0.5, 1.5], dtype=float),
    )

    assert mapping.nqp == 2
    assert np.isclose(mapping.point_data[0].g_n, -0.2)
    assert np.isclose(mapping.point_data[1].g_n, 0.1)
    assert np.allclose(mapping.point_data[1].G_u, [0.5, 0.25])
    assert np.allclose(mapping.point_data[1].G_a, [0.1, 0.3])
    assert np.allclose(mapping.point_data[1].H_uu_g, [[-0.2, 0.0], [0.0, 0.6]])
    assert np.allclose(mapping.point_data[1].H_uphi_g, [[-0.1, 0.0], [0.4, -0.2]])


def test_constant_terms_are_broadcast_to_all_quadrature_points() -> None:
    mapping = build_contact_surface_mapping(
        g_n=-0.15,
        G_u=np.array([1.0, -2.0], dtype=float),
        G_a=np.array([0.5, -0.25], dtype=float),
        H_uu_g=np.array([[0.3, 0.1], [0.1, 0.4]], dtype=float),
        H_uphi_g=np.array([[0.2, -0.1], [0.05, 0.3]], dtype=float),
        weights=np.array([0.3, 0.7, 1.1], dtype=float),
    )

    assert mapping.nqp == 3
    for point in mapping.point_data:
        assert np.isclose(point.g_n, -0.15)
        assert np.allclose(point.G_u, [1.0, -2.0])
        assert np.allclose(point.G_a, [0.5, -0.25])
        assert np.allclose(point.H_uu_g, [[0.3, 0.1], [0.1, 0.4]])
        assert np.allclose(point.H_uphi_g, [[0.2, -0.1], [0.05, 0.3]])


def test_shape_mismatch_raises_explicit_error() -> None:
    with pytest.raises(ValueError, match="H_uphi_g"):
        build_contact_surface_mapping(
            g_n=np.array([-0.2, -0.1], dtype=float),
            G_u=np.array([[1.0, -2.0], [0.5, 0.25]], dtype=float),
            G_a=np.array([0.5, -0.25], dtype=float),
            H_uu_g=np.array([[0.3, 0.1], [0.1, 0.4]], dtype=float),
            H_uphi_g=np.array([[0.2, -0.1, 0.0], [0.05, 0.3, -0.2]], dtype=float),
            weights=np.array([0.5, 1.5], dtype=float),
        )


def test_weights_are_preserved_pointwise() -> None:
    weights = np.array([0.2, 0.8, 1.4], dtype=float)
    mapping = build_contact_surface_mapping(
        g_n=np.array([-0.1, -0.2, 0.3], dtype=float),
        G_u=np.array([1.0, 0.5], dtype=float),
        G_a=np.array([0.2, -0.4], dtype=float),
        H_uu_g=np.array([[0.0, 0.1], [0.1, 0.2]], dtype=float),
        H_uphi_g=np.array([[0.3, -0.2], [0.0, 0.1]], dtype=float),
        weights=weights,
    )

    mapped_weights = np.array([point.weight for point in mapping.point_data], dtype=float)
    assert np.allclose(mapped_weights, weights)
