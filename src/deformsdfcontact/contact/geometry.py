"""Pure pointwise contact-geometry foundations for a minimal 2D setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _as_vector(values: object, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _reference_interface_abscissa(X: object, name: str = "X") -> float:
    array = np.asarray(X, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1 and array.shape[0] == 2:
        if not np.isclose(array[1], 0.0):
            raise ValueError(
                f"{name} must lie on the reference interface X[1] = 0, got {array!r}"
            )
        return float(array[0])
    raise ValueError(f"{name} must be a scalar abscissa or shape (2,), got {array.shape!r}")


class PointScalarField2D(Protocol):
    """Minimal pointwise scalar-field protocol used by the geometry evaluator."""

    def value(self, X: object) -> float:
        """Return the scalar field value at one reference point."""

    def gradient(self, X: object) -> np.ndarray:
        """Return the reference gradient at one reference point."""


@dataclass(frozen=True)
class AffineMasterMap2D:
    """Affine current master line `x = origin + xi * tangent` for `X = [xi, 0]`.

    The local geometry parameter vector is ordered as:

    `[origin_x, origin_y, tangent_x, tangent_y]`
    """

    origin: object
    tangent: object

    def __post_init__(self) -> None:
        origin = _as_vector(self.origin, "origin", 2)
        tangent = _as_vector(self.tangent, "tangent", 2)
        if float(np.linalg.norm(tangent)) <= 0.0:
            raise ValueError("tangent must have nonzero length")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "tangent", tangent)

    @property
    def parameter_vector(self) -> np.ndarray:
        return np.concatenate([self.origin, self.tangent]).copy()

    def with_parameter_vector(self, parameters: object) -> AffineMasterMap2D:
        vector = _as_vector(parameters, "parameters", 4)
        return AffineMasterMap2D(origin=vector[:2], tangent=vector[2:])

    def current_position(self, X: object) -> np.ndarray:
        """Return the current position of a reference point on the interface."""

        xi = _reference_interface_abscissa(X)
        return self.origin + xi * self.tangent

    def inverse_query(self, x_slave: object) -> np.ndarray:
        """Return the reference query point `[xi_c, 0]` for a slave point."""

        slave = _as_vector(x_slave, "x_slave", 2)
        tangent_norm_sq = float(self.tangent @ self.tangent)
        xi = float((slave - self.origin) @ self.tangent / tangent_norm_sq)
        return np.array([xi, 0.0], dtype=float)

    def query_sensitivity(self, x_slave: object) -> np.ndarray:
        """Return `dX_c / dd` with `d = [origin_x, origin_y, tangent_x, tangent_y]`."""

        slave = _as_vector(x_slave, "x_slave", 2)
        tangent_norm_sq = float(self.tangent @ self.tangent)
        r = slave - self.origin
        xi = float(r @ self.tangent / tangent_norm_sq)

        dxi_dorigin = -self.tangent / tangent_norm_sq
        dxi_dtangent = r / tangent_norm_sq - 2.0 * xi * self.tangent / tangent_norm_sq

        sensitivity = np.zeros((2, 4), dtype=float)
        sensitivity[0, :2] = dxi_dorigin
        sensitivity[0, 2:] = dxi_dtangent
        return sensitivity


@dataclass(frozen=True)
class AffinePhiField2D:
    """Affine reference scalar field `phi(X) = offset + gradient_vector . X`."""

    offset: float
    gradient_vector: object

    def __post_init__(self) -> None:
        gradient = _as_vector(self.gradient_vector, "gradient_vector", 2)
        object.__setattr__(self, "gradient_vector", gradient)

    def value(self, X: object) -> float:
        point = _as_vector(X, "X", 2)
        return float(self.offset + self.gradient_vector @ point)

    def gradient(self, X: object) -> np.ndarray:
        _as_vector(X, "X", 2)
        return self.gradient_vector.copy()

    def hessian(self, X: object) -> np.ndarray:
        """Return the zero Hessian of the affine field."""

        _as_vector(X, "X", 2)
        return np.zeros((2, 2), dtype=float)


@dataclass(frozen=True)
class ContactGeometryResult:
    """Structured output of the minimal contact-geometry evaluation."""

    x_slave: np.ndarray
    X_c: np.ndarray
    phi_at_query: float
    grad_phi_at_query: np.ndarray
    g_n: float
    G_u: np.ndarray
    G_a: np.ndarray


def query_point(x_slave: object, master_map: AffineMasterMap2D) -> np.ndarray:
    """Return the reference query point `X_c = [xi_c, 0]`."""

    return master_map.inverse_query(x_slave)


def normal_gap(phi_at_query: float) -> float:
    """Return the normal gap `g_n = phi(X_c)`."""

    return float(phi_at_query)


def gap_sensitivities(
    grad_phi_at_query: object,
    dXc_dgeometry: object,
    shape_values_at_query: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the local geometry and SDF sensitivities `(G_u, G_a)`.

    Shape conventions:

    - `grad_phi_at_query`: `(2,)`
    - `dXc_dgeometry`: `(2, nparam)`
    - `shape_values_at_query`: `(nphi_local,)`
    """

    grad_phi = _as_vector(grad_phi_at_query, "grad_phi_at_query", 2)
    dXc = np.asarray(dXc_dgeometry, dtype=float)
    if dXc.ndim != 2 or dXc.shape[0] != 2:
        raise ValueError(
            "dXc_dgeometry must have shape (2, nparam), "
            f"got {dXc.shape!r}"
        )
    shape_values = np.asarray(shape_values_at_query, dtype=float)
    if shape_values.ndim != 1:
        raise ValueError(
            "shape_values_at_query must have shape (nphi_local,), "
            f"got {shape_values.shape!r}"
        )
    G_u = grad_phi @ dXc
    G_a = shape_values.astype(float, copy=True)
    return G_u, G_a


def evaluate_contact_geometry(
    x_slave: object,
    master_map: AffineMasterMap2D,
    phi_field: PointScalarField2D,
    *,
    shape_values_at_query: object,
) -> ContactGeometryResult:
    """Evaluate the minimal contact geometry quantities for one slave point."""

    slave = _as_vector(x_slave, "x_slave", 2)
    X_c = query_point(slave, master_map)
    phi_at_query = float(phi_field.value(X_c))
    grad_phi_at_query = _as_vector(phi_field.gradient(X_c), "grad_phi_at_query", 2)
    dXc_dgeometry = master_map.query_sensitivity(slave)
    G_u, G_a = gap_sensitivities(
        grad_phi_at_query,
        dXc_dgeometry,
        shape_values_at_query,
    )
    return ContactGeometryResult(
        x_slave=slave.copy(),
        X_c=X_c,
        phi_at_query=phi_at_query,
        grad_phi_at_query=grad_phi_at_query,
        g_n=normal_gap(phi_at_query),
        G_u=G_u,
        G_a=G_a,
    )


__all__ = [
    "AffineMasterMap2D",
    "AffinePhiField2D",
    "ContactGeometryResult",
    "evaluate_contact_geometry",
    "gap_sensitivities",
    "normal_gap",
    "query_point",
]
