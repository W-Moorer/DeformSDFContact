"""UFL-first kinematics helpers for the extracted research package."""

from __future__ import annotations

from dataclasses import dataclass

import ufl


@dataclass(frozen=True)
class FiniteStrainKinematics:
    """Primary finite-strain measures derived from a displacement field."""

    dimension: int
    I: object
    grad_u: object
    F: object
    C: object
    C_inv: object
    J: object
    E: object


FiniteStrainState = FiniteStrainKinematics


def _square_tensor_dimension(tensor: object, name: str) -> int:
    shape = getattr(tensor, "ufl_shape", ())
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"{name} must be a square rank-2 tensor, got shape {shape!r}")
    return int(shape[0])


def spatial_dimension(u: object) -> int:
    """Return the geometric dimension carried by a displacement-like field."""

    return len(u)


def displacement_gradient(u: object) -> object:
    """Return the displacement gradient grad(u)."""

    return ufl.grad(u)


def deformation_gradient_from_grad_u(grad_u: object) -> object:
    """Build F = I + grad(u) from a square displacement-gradient tensor."""

    dimension = _square_tensor_dimension(grad_u, "grad_u")
    return ufl.Identity(dimension) + grad_u


def deformation_gradient(u: object) -> object:
    """Build F = I + grad(u) directly from a displacement field."""

    return deformation_gradient_from_grad_u(displacement_gradient(u))


def finite_strain_kinematics_from_F(F: object, grad_u: object | None = None) -> FiniteStrainKinematics:
    """Assemble the core finite-strain tensor measures from a deformation gradient."""

    dimension = _square_tensor_dimension(F, "F")
    I = ufl.Identity(dimension)
    if grad_u is None:
        grad_u = F - I
    C = right_cauchy_green(F)
    C_inv = inverse_right_cauchy_green(C)
    J = jacobian(F)
    E = 0.5 * (C - I)
    return FiniteStrainKinematics(
        dimension=dimension,
        I=I,
        grad_u=grad_u,
        F=F,
        C=C,
        C_inv=C_inv,
        J=J,
        E=E,
    )


def right_cauchy_green(F: object) -> object:
    """Return C = F^T F."""

    _square_tensor_dimension(F, "F")
    return F.T * F


def left_cauchy_green(F: object) -> object:
    """Return b = F F^T."""

    _square_tensor_dimension(F, "F")
    return F * F.T


def inverse_deformation_gradient(F: object) -> object:
    """Return F^{-1}."""

    _square_tensor_dimension(F, "F")
    return ufl.inv(F)


def inverse_right_cauchy_green(C: object) -> object:
    """Return C^{-1}."""

    _square_tensor_dimension(C, "C")
    return ufl.inv(C)


def jacobian(F: object) -> object:
    """Return J = det(F)."""

    _square_tensor_dimension(F, "F")
    return ufl.det(F)


def green_lagrange_strain(F: object) -> object:
    """Return E = 0.5 * (C - I)."""

    dimension = _square_tensor_dimension(F, "F")
    C = right_cauchy_green(F)
    return 0.5 * (C - ufl.Identity(dimension))


def small_strain(u: object) -> object:
    """Return the infinitesimal strain sym(grad(u))."""

    return ufl.sym(displacement_gradient(u))


def finite_strain_kinematics(u: object) -> FiniteStrainKinematics:
    """Assemble the core finite-strain tensor measures from a displacement field."""

    dimension = spatial_dimension(u)
    I = ufl.Identity(dimension)
    grad_u = displacement_gradient(u)
    F = deformation_gradient_from_grad_u(grad_u)
    return finite_strain_kinematics_from_F(F, grad_u=grad_u)


def kinematics(u: object) -> tuple[object, object, object]:
    """Compatibility helper matching the existing tuple-style benchmark API."""

    state = finite_strain_kinematics(u)
    return state.F, state.C, state.C_inv


__all__ = [
    "FiniteStrainState",
    "FiniteStrainKinematics",
    "deformation_gradient",
    "deformation_gradient_from_grad_u",
    "displacement_gradient",
    "finite_strain_kinematics",
    "finite_strain_kinematics_from_F",
    "green_lagrange_strain",
    "inverse_deformation_gradient",
    "inverse_right_cauchy_green",
    "jacobian",
    "kinematics",
    "left_cauchy_green",
    "right_cauchy_green",
    "small_strain",
    "spatial_dimension",
]
