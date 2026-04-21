"""Pure pointwise second-order contact geometry for the minimal 2D setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import AffineMasterMap2D, query_point


def _as_vector(values: object, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _as_matrix(values: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape!r}, got {array.shape!r}")
    return array


def _as_shape_gradient_matrix(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(
            "shape_gradients_at_query must have shape (2, nphi_local), "
            f"got {array.shape!r}"
        )
    return array


class PointScalarFieldHessian2D(Protocol):
    """Minimal pointwise scalar-field protocol with Hessian access."""

    def value(self, X: object) -> float:
        """Return the scalar field value at one reference point."""

    def gradient(self, X: object) -> np.ndarray:
        """Return the reference gradient at one reference point."""

    def hessian(self, X: object) -> np.ndarray:
        """Return the reference Hessian at one reference point."""


@dataclass(frozen=True)
class QuadraticPhiField2D:
    """Quadratic reference scalar field `phi = c + l.X + 0.5 X^T H X`."""

    offset: float
    linear_vector: object
    hessian_matrix: object

    def __post_init__(self) -> None:
        linear = _as_vector(self.linear_vector, "linear_vector", 2)
        hessian = _as_matrix(self.hessian_matrix, "hessian_matrix", (2, 2))
        object.__setattr__(self, "linear_vector", linear)
        object.__setattr__(self, "hessian_matrix", 0.5 * (hessian + hessian.T))

    def value(self, X: object) -> float:
        point = _as_vector(X, "X", 2)
        return float(
            self.offset
            + self.linear_vector @ point
            + 0.5 * point @ self.hessian_matrix @ point
        )

    def gradient(self, X: object) -> np.ndarray:
        point = _as_vector(X, "X", 2)
        return self.linear_vector + self.hessian_matrix @ point

    def hessian(self, X: object) -> np.ndarray:
        _as_vector(X, "X", 2)
        return self.hessian_matrix.copy()


@dataclass(frozen=True)
class ContactSecondOrderGeometryResult:
    """Structured output of the minimal second-order contact geometry layer."""

    X_c: np.ndarray
    E: np.ndarray
    d2Xc_dd2: np.ndarray
    grad_phi_at_query: np.ndarray
    hessian_phi_at_query: np.ndarray
    H_uu_curvature: np.ndarray
    H_uu_query_acceleration: np.ndarray
    H_uu_g: np.ndarray
    H_uphi_g: np.ndarray | None


def query_sensitivity_second_order(
    x_slave: object,
    master_map: AffineMasterMap2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(E, d2Xc_dd2)` for the affine master projection map.

    Shapes:

    - `E`: `(2, 4)`
    - `d2Xc_dd2`: `(2, 4, 4)`
    """

    slave = _as_vector(x_slave, "x_slave", 2)
    origin = master_map.origin
    tangent = master_map.tangent

    r = slave - origin
    n = float(r @ tangent)
    m = float(tangent @ tangent)
    xi = float(n / m)

    E = np.zeros((2, 4), dtype=float)
    E[0, :2] = -tangent / m
    E[0, 2:] = r / m - 2.0 * xi * tangent / m

    H_xi = np.zeros((4, 4), dtype=float)
    H_bt = -np.eye(2, dtype=float) / m + 2.0 * np.outer(tangent, tangent) / (m * m)
    H_tt = (
        -2.0 * (np.outer(r, tangent) + np.outer(tangent, r)) / (m * m)
        - 2.0 * n * np.eye(2, dtype=float) / (m * m)
        + 8.0 * n * np.outer(tangent, tangent) / (m * m * m)
    )
    H_xi[:2, 2:] = H_bt
    H_xi[2:, :2] = H_bt.T
    H_xi[2:, 2:] = H_tt

    d2Xc_dd2 = np.zeros((2, 4, 4), dtype=float)
    d2Xc_dd2[0] = H_xi
    return E, d2Xc_dd2


def second_order_gap_geometry(
    E: object,
    d2Xc_dd2: object,
    grad_phi_at_query: object,
    hessian_phi_at_query: object,
    *,
    shape_gradients_at_query: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Return `(H_uu_curvature, H_uu_query_acceleration, H_uu_g, H_uphi_g)`."""

    E_array = _as_matrix(E, "E", (2, 4))
    d2Xc = np.asarray(d2Xc_dd2, dtype=float)
    if d2Xc.shape != (2, 4, 4):
        raise ValueError(f"d2Xc_dd2 must have shape (2, 4, 4), got {d2Xc.shape!r}")
    grad_phi = _as_vector(grad_phi_at_query, "grad_phi_at_query", 2)
    hessian_phi = _as_matrix(hessian_phi_at_query, "hessian_phi_at_query", (2, 2))

    H_uu_curvature = E_array.T @ hessian_phi @ E_array
    H_uu_query_acceleration = np.tensordot(grad_phi, d2Xc, axes=(0, 0))
    H_uu_g = H_uu_curvature + H_uu_query_acceleration

    H_uphi_g = None
    if shape_gradients_at_query is not None:
        B_phi = _as_shape_gradient_matrix(shape_gradients_at_query)
        H_uphi_g = E_array.T @ B_phi

    return H_uu_curvature, H_uu_query_acceleration, H_uu_g, H_uphi_g


def evaluate_contact_second_order_geometry(
    x_slave: object,
    master_map: AffineMasterMap2D,
    phi_field: PointScalarFieldHessian2D,
    *,
    shape_gradients_at_query: object | None = None,
) -> ContactSecondOrderGeometryResult:
    """Evaluate the minimal second-order contact geometry quantities."""

    X_c = query_point(x_slave, master_map)
    E, d2Xc_dd2 = query_sensitivity_second_order(x_slave, master_map)
    grad_phi_at_query = _as_vector(phi_field.gradient(X_c), "grad_phi_at_query", 2)
    hessian_phi_at_query = _as_matrix(
        phi_field.hessian(X_c),
        "hessian_phi_at_query",
        (2, 2),
    )
    H_uu_curvature, H_uu_query_acceleration, H_uu_g, H_uphi_g = (
        second_order_gap_geometry(
            E,
            d2Xc_dd2,
            grad_phi_at_query,
            hessian_phi_at_query,
            shape_gradients_at_query=shape_gradients_at_query,
        )
    )
    return ContactSecondOrderGeometryResult(
        X_c=X_c,
        E=E,
        d2Xc_dd2=d2Xc_dd2,
        grad_phi_at_query=grad_phi_at_query,
        hessian_phi_at_query=hessian_phi_at_query,
        H_uu_curvature=H_uu_curvature,
        H_uu_query_acceleration=H_uu_query_acceleration,
        H_uu_g=H_uu_g,
        H_uphi_g=H_uphi_g,
    )


__all__ = [
    "ContactSecondOrderGeometryResult",
    "PointScalarFieldHessian2D",
    "QuadraticPhiField2D",
    "evaluate_contact_second_order_geometry",
    "query_sensitivity_second_order",
    "second_order_gap_geometry",
]
