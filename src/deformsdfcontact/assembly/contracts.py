"""Assembly-neutral contracts for monolithic residual and tangent blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_index_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (n,), got {array.shape!r}")
    return array


def _as_vector(values: object, name: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape!r}")
    return array


def _as_matrix(values: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape!r}, got {array.shape!r}")
    return array


@dataclass(frozen=True)
class MonolithicBlockLayout:
    """Global block layout for the monolithic `(u, phi)` system."""

    ndof_u: int
    ndof_phi: int

    def __post_init__(self) -> None:
        if int(self.ndof_u) < 0 or int(self.ndof_phi) < 0:
            raise ValueError("ndof_u and ndof_phi must be nonnegative")
        object.__setattr__(self, "ndof_u", int(self.ndof_u))
        object.__setattr__(self, "ndof_phi", int(self.ndof_phi))

    @property
    def total_dofs(self) -> int:
        return self.ndof_u + self.ndof_phi

    @property
    def phi_offset(self) -> int:
        return self.ndof_u

    @property
    def block_shapes(self) -> dict[str, tuple[int, int]]:
        return {
            "uu": (self.ndof_u, self.ndof_u),
            "uphi": (self.ndof_u, self.ndof_phi),
            "phiu": (self.ndof_phi, self.ndof_u),
            "phiphi": (self.ndof_phi, self.ndof_phi),
        }

    def lift_phi_dofs(self, phi_dofs: object) -> np.ndarray:
        return _as_index_vector(phi_dofs, "phi_dofs") + self.phi_offset


@dataclass(frozen=True)
class AssemblyPlan:
    """Minimal metadata for one backend-specific assembly realization."""

    layout: MonolithicBlockLayout
    backend_name: str
    note: str = ""


@dataclass(frozen=True)
class SolidLocalContribution:
    """Local solid contribution writing only the `(u, u)` block and `R_u`."""

    u_dofs: np.ndarray
    R_u: np.ndarray
    K_uu: np.ndarray

    def __post_init__(self) -> None:
        u_dofs = _as_index_vector(self.u_dofs, "u_dofs")
        R_u = _as_vector(self.R_u, "R_u", u_dofs.shape[0])
        K_uu = _as_matrix(self.K_uu, "K_uu", (u_dofs.shape[0], u_dofs.shape[0]))
        object.__setattr__(self, "u_dofs", u_dofs)
        object.__setattr__(self, "R_u", R_u)
        object.__setattr__(self, "K_uu", K_uu)


@dataclass(frozen=True)
class SDFLocalContribution:
    """Local SDF contribution writing `R_phi`, `K_phiu`, and `K_phiphi`."""

    u_dofs: np.ndarray
    phi_dofs: np.ndarray
    R_phi: np.ndarray
    K_phiu: np.ndarray
    K_phiphi: np.ndarray

    def __post_init__(self) -> None:
        u_dofs = _as_index_vector(self.u_dofs, "u_dofs")
        phi_dofs = _as_index_vector(self.phi_dofs, "phi_dofs")
        R_phi = _as_vector(self.R_phi, "R_phi", phi_dofs.shape[0])
        K_phiu = _as_matrix(self.K_phiu, "K_phiu", (phi_dofs.shape[0], u_dofs.shape[0]))
        K_phiphi = _as_matrix(
            self.K_phiphi,
            "K_phiphi",
            (phi_dofs.shape[0], phi_dofs.shape[0]),
        )
        object.__setattr__(self, "u_dofs", u_dofs)
        object.__setattr__(self, "phi_dofs", phi_dofs)
        object.__setattr__(self, "R_phi", R_phi)
        object.__setattr__(self, "K_phiu", K_phiu)
        object.__setattr__(self, "K_phiphi", K_phiphi)


@dataclass(frozen=True)
class ContactLocalContribution:
    """Local contact contribution writing `R_u`, `K_uu`, and `K_uphi`."""

    u_dofs: np.ndarray
    phi_dofs: np.ndarray
    R_u: np.ndarray
    K_uu: np.ndarray
    K_uphi: np.ndarray

    def __post_init__(self) -> None:
        u_dofs = _as_index_vector(self.u_dofs, "u_dofs")
        phi_dofs = _as_index_vector(self.phi_dofs, "phi_dofs")
        R_u = _as_vector(self.R_u, "R_u", u_dofs.shape[0])
        K_uu = _as_matrix(self.K_uu, "K_uu", (u_dofs.shape[0], u_dofs.shape[0]))
        K_uphi = _as_matrix(self.K_uphi, "K_uphi", (u_dofs.shape[0], phi_dofs.shape[0]))
        object.__setattr__(self, "u_dofs", u_dofs)
        object.__setattr__(self, "phi_dofs", phi_dofs)
        object.__setattr__(self, "R_u", R_u)
        object.__setattr__(self, "K_uu", K_uu)
        object.__setattr__(self, "K_uphi", K_uphi)


__all__ = [
    "AssemblyPlan",
    "ContactLocalContribution",
    "MonolithicBlockLayout",
    "SDFLocalContribution",
    "SolidLocalContribution",
]
