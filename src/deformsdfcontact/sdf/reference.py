"""Pure pointwise reference-geometry helpers for signed-distance work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_array_1d(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (dim,), got {array.shape!r}")
    return array


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must have nonzero length")
    return vector / norm


def _as_points(points: object, dimension: int) -> tuple[np.ndarray, bool]:
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        if array.shape[0] != dimension:
            raise ValueError(f"expected point shape ({dimension},), got {array.shape!r}")
        return array.reshape(1, dimension), True
    if array.ndim == 2:
        if array.shape[1] != dimension:
            raise ValueError(f"expected point batch shape (N, {dimension}), got {array.shape!r}")
        return array, False
    raise ValueError(f"points must have shape ({dimension},) or (N, {dimension}), got {array.shape!r}")


def _restore_scalar(values: np.ndarray, single: bool):
    return float(values[0]) if single else values


def _restore_bool(values: np.ndarray, single: bool):
    return bool(values[0]) if single else values


def _restore_points(values: np.ndarray, single: bool):
    return values[0] if single else values


@dataclass(frozen=True)
class ReferencePlane:
    """Analytic planar reference interface.

    Shape conventions:

    - `point_on_interface`: `(dim,)`
    - `unit_normal`: `(dim,)`, normalized internally
    - method input `X`: `(dim,)` or `(N, dim)`
    """

    point_on_interface: object
    unit_normal: object

    def __post_init__(self) -> None:
        point = _as_array_1d(self.point_on_interface, "point_on_interface")
        normal = _as_array_1d(self.unit_normal, "unit_normal")
        if point.shape != normal.shape:
            raise ValueError(
                "point_on_interface and unit_normal must have the same shape, "
                f"got {point.shape!r} and {normal.shape!r}"
            )
        object.__setattr__(self, "point_on_interface", point)
        object.__setattr__(self, "unit_normal", _normalize(normal, "unit_normal"))

    @property
    def dimension(self) -> int:
        return int(self.unit_normal.shape[0])

    def phi0(self, X: object):
        """Return the signed distance to the reference plane."""

        points, single = _as_points(X, self.dimension)
        signed_distance = (points - self.point_on_interface) @ self.unit_normal
        return _restore_scalar(signed_distance, single)

    def nearest_point(self, X: object):
        """Return the orthogonal projection of `X` onto the reference plane."""

        points, single = _as_points(X, self.dimension)
        signed_distance = (points - self.point_on_interface) @ self.unit_normal
        projected = points - signed_distance[:, None] * self.unit_normal[None, :]
        return _restore_points(projected, single)

    def normal_at_nearest_point(self, X: object):
        """Return the unit normal at the nearest point on the reference plane."""

        points, single = _as_points(X, self.dimension)
        normals = np.repeat(self.unit_normal[None, :], points.shape[0], axis=0)
        return _restore_points(normals, single)

    def sign(self, X: object):
        """Return the sign of the reference signed distance."""

        signed_distance = np.asarray(self.phi0(X), dtype=float)
        signs = np.sign(signed_distance).astype(int, copy=False)
        if signs.ndim == 0:
            return int(signs)
        return signs

    def in_narrow_band(self, X: object, delta: float):
        """Return whether `|phi0(X)| < delta`."""

        if delta < 0.0:
            raise ValueError("delta must be nonnegative")
        signed_distance = np.asarray(self.phi0(X), dtype=float)
        inside = np.abs(signed_distance) < float(delta)
        if inside.ndim == 0:
            return bool(inside)
        return inside
