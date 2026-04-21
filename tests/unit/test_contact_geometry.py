#!/usr/bin/env python3
"""Unit tests for the minimal contact-geometry foundation."""

from __future__ import annotations

import numpy as np

from deformsdfcontact.contact.geometry import (
    AffineMasterMap2D,
    AffinePhiField2D,
    evaluate_contact_geometry,
    query_point,
)


def test_query_point_satisfies_affine_projection_relation() -> None:
    master_map = AffineMasterMap2D(origin=[0.5, -1.0], tangent=[2.0, 1.0])
    x_slave = np.array([2.0, 0.5], dtype=float)

    X_c = query_point(x_slave, master_map)
    x_query = master_map.current_position(X_c)
    projection_residual = x_slave - x_query

    assert np.isclose(X_c[1], 0.0)
    assert np.isclose(projection_residual @ master_map.tangent, 0.0)


def test_gap_definition_matches_phi_evaluated_at_query_point() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = AffinePhiField2D(offset=-0.25, gradient_vector=[1.0, 0.0])
    shape_values = np.array([0.3, 0.7], dtype=float)
    x_slave = np.array([0.8, 1.2], dtype=float)

    result = evaluate_contact_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_values_at_query=shape_values,
    )

    assert np.isclose(result.g_n, result.phi_at_query)
    assert np.isclose(result.g_n, phi_field.value(result.X_c))


def test_Ga_matches_local_shape_values_by_definition() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = AffinePhiField2D(offset=0.0, gradient_vector=[1.0, 0.0])
    shape_values = np.array([0.2, 0.5, 0.3], dtype=float)
    x_slave = np.array([0.4, -0.7], dtype=float)

    result = evaluate_contact_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_values_at_query=shape_values,
    )

    assert np.allclose(result.G_a, shape_values)


def test_Gu_matches_finite_difference_for_affine_master_map() -> None:
    master_map = AffineMasterMap2D(origin=[0.2, -0.1], tangent=[1.5, 0.8])
    phi_field = AffinePhiField2D(offset=-0.35, gradient_vector=[1.7, -0.4])
    shape_values = np.array([0.4, 0.6], dtype=float)
    x_slave = np.array([1.1, 0.4], dtype=float)
    eps = 1.0e-7

    result = evaluate_contact_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_values_at_query=shape_values,
    )

    fd = np.zeros(4, dtype=float)
    base_parameters = master_map.parameter_vector
    for i in range(fd.shape[0]):
        direction = np.zeros_like(base_parameters)
        direction[i] = 1.0
        map_plus = master_map.with_parameter_vector(base_parameters + eps * direction)
        map_minus = master_map.with_parameter_vector(base_parameters - eps * direction)
        gap_plus = evaluate_contact_geometry(
            x_slave,
            map_plus,
            phi_field,
            shape_values_at_query=shape_values,
        ).g_n
        gap_minus = evaluate_contact_geometry(
            x_slave,
            map_minus,
            phi_field,
            shape_values_at_query=shape_values,
        ).g_n
        fd[i] = (gap_plus - gap_minus) / (2.0 * eps)

    assert np.allclose(result.G_u, fd, rtol=1.0e-6, atol=1.0e-8)


def test_zero_gap_when_query_point_lies_on_zero_level_set() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = AffinePhiField2D(offset=-0.5, gradient_vector=[1.0, 0.0])
    x_slave = np.array([0.5, 2.0], dtype=float)

    result = evaluate_contact_geometry(
        x_slave,
        master_map,
        phi_field,
        shape_values_at_query=np.array([1.0], dtype=float),
    )

    assert np.isclose(result.X_c[0], 0.5)
    assert np.isclose(result.g_n, 0.0)


def test_gap_sign_matches_phi_sign_on_opposite_sides_of_zero_level_set() -> None:
    master_map = AffineMasterMap2D(origin=[0.0, 0.0], tangent=[1.0, 0.0])
    phi_field = AffinePhiField2D(offset=-0.5, gradient_vector=[1.0, 0.0])
    shape_values = np.array([0.5, 0.5], dtype=float)

    negative = evaluate_contact_geometry(
        np.array([0.2, 1.0], dtype=float),
        master_map,
        phi_field,
        shape_values_at_query=shape_values,
    )
    positive = evaluate_contact_geometry(
        np.array([0.8, -1.0], dtype=float),
        master_map,
        phi_field,
        shape_values_at_query=shape_values,
    )

    assert negative.g_n < 0.0
    assert positive.g_n > 0.0
