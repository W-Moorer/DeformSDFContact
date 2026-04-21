#!/usr/bin/env python3
"""Unit tests for the extracted kinematics foundation."""

from __future__ import annotations

import numpy as np
import ufl

from deformsdfcontact.kinematics import (
    deformation_gradient_from_grad_u,
    finite_strain_kinematics,
    green_lagrange_strain,
    jacobian,
    kinematics,
    left_cauchy_green,
    right_cauchy_green,
    small_strain,
)


def test_constant_gradient_measures_match_expected_values() -> None:
    grad_u = ufl.as_matrix(((1.0, 2.0), (3.0, 4.0)))
    F = deformation_gradient_from_grad_u(grad_u)
    C = right_cauchy_green(F)
    b = left_cauchy_green(F)
    E = green_lagrange_strain(F)
    J = jacobian(F)

    assert F.ufl_shape == (2, 2)
    assert float(F[0, 0]) == 2.0
    assert float(F[1, 1]) == 5.0

    assert float(C[0, 0]) == 13.0
    assert float(C[0, 1]) == 19.0
    assert float(C[1, 1]) == 29.0

    assert float(b[0, 0]) == 8.0
    assert float(b[0, 1]) == 16.0
    assert float(b[1, 1]) == 34.0

    assert float(E[0, 0]) == 6.0
    assert float(E[0, 1]) == 9.5
    assert float(E[1, 1]) == 14.0
    assert float(J) == 4.0


def test_finite_strain_state_exposes_inverse_metric_and_jacobian() -> None:
    cell = ufl.triangle
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    x = ufl.SpatialCoordinate(domain)
    u = ufl.as_vector((x[0] + 2.0 * x[1], 3.0 * x[0] - 1.0 * x[1]))

    state = finite_strain_kinematics(u)
    F, C, C_inv = kinematics(u)

    expected_C = np.array([[13.0, 4.0], [4.0, 4.0]], dtype=float)
    expected_C_inv = np.linalg.inv(expected_C)

    assert state.dimension == 2
    assert float(state.F[0, 0]) == 2.0
    assert float(state.F[0, 1]) == 2.0
    assert float(state.F[1, 0]) == 3.0
    assert float(state.F[1, 1]) == 0.0

    assert float(state.C[0, 0]) == 13.0
    assert float(state.C[0, 1]) == 4.0
    assert float(state.C[1, 1]) == 4.0
    assert abs(float(state.C_inv[0, 0]) - expected_C_inv[0, 0]) < 1.0e-12
    assert abs(float(state.C_inv[0, 1]) - expected_C_inv[0, 1]) < 1.0e-12
    assert abs(float(state.C_inv[1, 1]) - expected_C_inv[1, 1]) < 1.0e-12
    assert float(state.J) == -6.0

    assert float(F[1, 0]) == 3.0
    assert float(C[0, 1]) == 4.0
    assert abs(float(C_inv[1, 0]) - expected_C_inv[1, 0]) < 1.0e-12


def test_small_strain_uses_symmetric_displacement_gradient() -> None:
    cell = ufl.triangle
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    x = ufl.SpatialCoordinate(domain)
    u = ufl.as_vector((x[0] + 2.0 * x[1], 3.0 * x[0] - 1.0 * x[1]))

    eps = small_strain(u)

    assert eps.ufl_shape == (2, 2)
    assert float(eps[0, 0]) == 1.0
    assert float(eps[0, 1]) == 2.5
    assert float(eps[1, 0]) == 2.5
    assert float(eps[1, 1]) == -1.0
