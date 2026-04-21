#!/usr/bin/env python3
"""Unit tests for the extracted materials foundation."""

from __future__ import annotations

import math

import numpy as np
import ufl

from deformsdfcontact.kinematics import finite_strain_kinematics, finite_strain_kinematics_from_F
from deformsdfcontact.materials import (
    CompressibleNeoHookean,
    IsotropicElasticParameters,
    LinearElasticSmallStrain,
)


def _ufl_matrix(values: np.ndarray) -> object:
    return ufl.as_matrix(tuple(tuple(float(v) for v in row) for row in values))


def _tensor_to_numpy(expr: object) -> np.ndarray:
    shape = getattr(expr, "ufl_shape", ())
    if shape == ():
        return np.asarray(float(expr), dtype=float)
    if len(shape) == 2:
        out = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[i, j] = float(expr[i, j])
        return out
    if len(shape) == 4:
        out = np.zeros(shape, dtype=float)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        out[i, j, k, l] = float(expr[i, j, k, l])
        return out
    raise TypeError(f"unsupported tensor shape {shape!r}")


def _state_from_matrix(F_values: np.ndarray):
    return finite_strain_kinematics_from_F(_ufl_matrix(F_values))


def _directional_fd_check(model, params, F0: np.ndarray, H: np.ndarray, step: float = 1.0e-7):
    state0 = _state_from_matrix(F0)
    tangent = _tensor_to_numpy(model.consistent_tangent(state0, params))
    predicted = np.einsum("ijkl,kl->ij", tangent, H)

    plus = _state_from_matrix(F0 + step * H)
    minus = _state_from_matrix(F0 - step * H)
    stress_plus = _tensor_to_numpy(model.stress_measure(plus, params))
    stress_minus = _tensor_to_numpy(model.stress_measure(minus, params))
    finite_difference = (stress_plus - stress_minus) / (2.0 * step)
    return predicted, finite_difference


def test_linear_elastic_small_strain_rigid_motion_has_zero_energy_and_stress() -> None:
    cell = ufl.triangle
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    x = ufl.SpatialCoordinate(domain)
    omega = 0.25
    u = ufl.as_vector((-omega * x[1], omega * x[0]))

    state = finite_strain_kinematics(u)
    params = IsotropicElasticParameters(E=12.0, nu=0.25)
    model = LinearElasticSmallStrain()

    stress = _tensor_to_numpy(model.stress_measure(state, params))
    energy = float(model.strain_energy_density(state, params))

    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert math.isclose(energy, 0.0, abs_tol=1.0e-12)


def test_compressible_neo_hookean_rigid_rotation_has_zero_energy_and_stress() -> None:
    theta = 0.37
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=float,
    )

    state = _state_from_matrix(rotation)
    params = IsotropicElasticParameters(E=12.0, nu=0.25)
    model = CompressibleNeoHookean()

    stress = _tensor_to_numpy(model.stress_measure(state, params))
    energy = float(model.strain_energy_density(state, params))

    assert np.allclose(stress, 0.0, atol=1.0e-12)
    assert math.isclose(float(state.J), 1.0, abs_tol=1.0e-12)
    assert math.isclose(energy, 0.0, abs_tol=1.0e-12)


def test_compressible_neo_hookean_matches_linear_elastic_in_small_strain_limit() -> None:
    F_values = np.array(
        [
            [1.0 + 1.0e-6, 2.0e-7],
            [2.0e-7, 1.0 - 3.0e-7],
        ],
        dtype=float,
    )

    state = _state_from_matrix(F_values)
    params = IsotropicElasticParameters(E=25.0, nu=0.30)
    linear = LinearElasticSmallStrain()
    neo_hookean = CompressibleNeoHookean()

    stress_linear = _tensor_to_numpy(linear.stress_measure(state, params))
    stress_neo = _tensor_to_numpy(neo_hookean.stress_measure(state, params))

    assert np.allclose(stress_neo, stress_linear, atol=5.0e-11, rtol=5.0e-6)


def test_linear_elastic_tangent_matches_directional_finite_difference() -> None:
    params = IsotropicElasticParameters(E=18.0, nu=0.22)
    model = LinearElasticSmallStrain()
    F0 = np.array([[1.05, 0.10], [-0.03, 0.97]], dtype=float)
    H = np.array([[0.30, -0.20], [0.10, 0.05]], dtype=float)

    predicted, finite_difference = _directional_fd_check(model, params, F0, H)

    assert np.allclose(predicted, finite_difference, atol=1.0e-8, rtol=1.0e-8)


def test_neo_hookean_tangent_matches_directional_finite_difference() -> None:
    params = IsotropicElasticParameters(E=18.0, nu=0.22)
    model = CompressibleNeoHookean()
    F0 = np.array([[1.10, 0.15], [-0.05, 0.95]], dtype=float)
    H = np.array([[0.25, -0.18], [0.07, 0.04]], dtype=float)

    predicted, finite_difference = _directional_fd_check(model, params, F0, H)

    assert np.allclose(predicted, finite_difference, atol=5.0e-7, rtol=5.0e-6)
