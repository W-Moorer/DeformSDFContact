#!/usr/bin/env python3
"""Unit tests for the assembly-neutral reinitialize form-mapping contract."""

from __future__ import annotations

import numpy as np
import pytest

from deformsdfcontact.sdf.form_mapping import build_reinitialize_element_mapping


def test_zero_beta_is_valid_and_local_fields_are_interpolated_per_quadrature_point() -> None:
    phi_local = np.array([0.0, 1.0], dtype=float)
    phi_target_local = np.array([0.0, 2.0], dtype=float)
    shape_values = np.array([[0.75, 0.25], [0.25, 0.75]], dtype=float)
    shape_gradients = np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float)
    weights = np.array([0.4, 0.6], dtype=float)

    mapping = build_reinitialize_element_mapping(
        phi_local,
        shape_values,
        shape_gradients,
        np.array([[1.0]], dtype=float),
        weights,
        phi_target=phi_target_local,
        beta=0.0,
    )

    assert mapping.nshape == 2
    assert mapping.nqp == 2
    assert mapping.dimension == 1
    assert np.isclose(mapping.quadrature_points[0].phi, 0.25)
    assert np.isclose(mapping.quadrature_points[1].phi, 0.75)
    assert np.allclose(mapping.quadrature_points[0].grad_phi, [1.0])
    assert np.allclose(mapping.quadrature_points[1].grad_phi, [1.0])
    assert np.isclose(mapping.quadrature_points[0].phi_target, 0.5)
    assert np.isclose(mapping.quadrature_points[1].phi_target, 1.5)
    assert np.isclose(mapping.quadrature_points[0].beta, 0.0)
    assert np.isclose(mapping.quadrature_points[1].beta, 0.0)


def test_scalar_beta_is_broadcast_to_all_quadrature_points() -> None:
    mapping = build_reinitialize_element_mapping(
        phi_local=np.array([1.0, 3.0], dtype=float),
        shape_values=np.array([[0.5, 0.5], [0.2, 0.8]], dtype=float),
        shape_gradients=np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float),
        A=np.array([[2.0]], dtype=float),
        weights=np.array([0.3, 0.7], dtype=float),
        beta=1.75,
    )

    beta_values = np.array([qp.beta for qp in mapping.quadrature_points], dtype=float)
    assert np.allclose(beta_values, [1.75, 1.75])


def test_pointwise_beta_array_is_preserved_and_negative_beta_is_rejected() -> None:
    mapping = build_reinitialize_element_mapping(
        phi_local=np.array([0.0, 1.0], dtype=float),
        shape_values=np.array([[0.6, 0.4], [0.1, 0.9]], dtype=float),
        shape_gradients=np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float),
        A=np.array([[[1.0]], [[1.5]]], dtype=float),
        weights=np.array([0.5, 0.5], dtype=float),
        beta=np.array([0.0, 2.5], dtype=float),
    )

    beta_values = np.array([qp.beta for qp in mapping.quadrature_points], dtype=float)
    assert np.allclose(beta_values, [0.0, 2.5])

    with pytest.raises(ValueError, match="beta must be nonnegative"):
        build_reinitialize_element_mapping(
            phi_local=np.array([0.0, 1.0], dtype=float),
            shape_values=np.array([[0.5, 0.5]], dtype=float),
            shape_gradients=np.array([[[-1.0], [1.0]]], dtype=float),
            A=np.array([[1.0]], dtype=float),
            weights=np.array([1.0], dtype=float),
            beta=-1.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "phi_local": np.array([0.0, 1.0], dtype=float),
                "shape_values": np.array([[0.5, 0.5, 0.0]], dtype=float),
                "shape_gradients": np.array([[[-1.0], [1.0]]], dtype=float),
                "A": np.array([[1.0]], dtype=float),
                "weights": np.array([1.0], dtype=float),
            },
            "shape_values",
        ),
        (
            {
                "phi_local": np.array([0.0, 1.0], dtype=float),
                "shape_values": np.array([[0.5, 0.5]], dtype=float),
                "shape_gradients": np.array([[[-1.0, 0.0], [1.0, 0.0]]], dtype=float),
                "A": np.array([[1.0]], dtype=float),
                "weights": np.array([1.0], dtype=float),
            },
            "A",
        ),
        (
            {
                "phi_local": np.array([0.0, 1.0], dtype=float),
                "shape_values": np.array([[0.5, 0.5], [0.25, 0.75]], dtype=float),
                "shape_gradients": np.array([[[-1.0], [1.0]]], dtype=float),
                "A": np.array([[1.0]], dtype=float),
                "weights": np.array([1.0, 1.0], dtype=float),
            },
            "shape_gradients",
        ),
        (
            {
                "phi_local": np.array([0.0, 1.0], dtype=float),
                "shape_values": np.array([[0.5, 0.5], [0.25, 0.75]], dtype=float),
                "shape_gradients": np.array([[[-1.0], [1.0]], [[-1.0], [1.0]]], dtype=float),
                "A": np.array([[1.0]], dtype=float),
                "weights": np.array([1.0], dtype=float),
            },
            "weights",
        ),
    ],
)
def test_shape_contract_is_enforced(kwargs: dict[str, np.ndarray], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_reinitialize_element_mapping(**kwargs)
