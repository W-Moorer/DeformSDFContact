"""Tensor-level material foundations for the extracted research package."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import ufl

from .kinematics import FiniteStrainState


@dataclass(frozen=True)
class IsotropicElasticParameters:
    """Isotropic elastic parameters shared by the minimal foundation models."""

    E: float
    nu: float

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lame_lambda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


class MaterialModel(ABC):
    """Minimal tensor-level material interface.

    The common interface is expressed in terms of:

    - strain energy density W(F)
    - first Piola stress P = dW/dF
    - consistent material tangent A = dP/dF
    """

    @abstractmethod
    def strain_energy_density(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        """Return the scalar strain energy density."""

    @abstractmethod
    def stress_measure(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        """Return the first Piola stress tensor."""

    @abstractmethod
    def consistent_tangent(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        """Return the rank-4 tensor dP/dF."""


def _material_response(material: MaterialModel, state: FiniteStrainState, params: IsotropicElasticParameters):
    F = ufl.variable(state.F)
    W = material._strain_energy_from_variable_F(F, state, params)
    P = ufl.diff(W, F)
    A = ufl.diff(P, F)
    return W, P, A


class LinearElasticSmallStrain(MaterialModel):
    """Small-strain linear elasticity embedded in a tensor-level F-based interface."""

    def _strain_energy_from_variable_F(
        self,
        F: object,
        state: FiniteStrainState,
        params: IsotropicElasticParameters,
    ) -> object:
        I = ufl.Identity(state.dimension)
        eps = ufl.sym(F - I)
        mu = params.mu
        lame_lambda = params.lame_lambda
        return 0.5 * lame_lambda * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)

    def strain_energy_density(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        W, _, _ = _material_response(self, state, params)
        return W

    def stress_measure(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        _, P, _ = _material_response(self, state, params)
        return P

    def consistent_tangent(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        _, _, A = _material_response(self, state, params)
        return A


class CompressibleNeoHookean(MaterialModel):
    """Minimal compressible Neo-Hookean model with a first-Piola interface."""

    def _strain_energy_from_variable_F(
        self,
        F: object,
        state: FiniteStrainState,
        params: IsotropicElasticParameters,
    ) -> object:
        dimension = state.dimension
        mu = params.mu
        lame_lambda = params.lame_lambda
        C = F.T * F
        J = ufl.det(F)
        return 0.5 * mu * (ufl.tr(C) - dimension) - mu * ufl.ln(J) + 0.5 * lame_lambda * ufl.ln(J) ** 2

    def strain_energy_density(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        W, _, _ = _material_response(self, state, params)
        return W

    def stress_measure(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        _, P, _ = _material_response(self, state, params)
        return P

    def consistent_tangent(self, state: FiniteStrainState, params: IsotropicElasticParameters) -> object:
        _, _, A = _material_response(self, state, params)
        return A


def strain_energy_density(
    material: MaterialModel,
    state: FiniteStrainState,
    params: IsotropicElasticParameters,
) -> object:
    """Module-level access to the material strain energy density."""

    return material.strain_energy_density(state, params)


def stress_measure(
    material: MaterialModel,
    state: FiniteStrainState,
    params: IsotropicElasticParameters,
) -> object:
    """Module-level access to the first Piola stress."""

    return material.stress_measure(state, params)


def consistent_tangent(
    material: MaterialModel,
    state: FiniteStrainState,
    params: IsotropicElasticParameters,
) -> object:
    """Module-level access to the rank-4 material tangent dP/dF."""

    return material.consistent_tangent(state, params)


__all__ = [
    "CompressibleNeoHookean",
    "IsotropicElasticParameters",
    "LinearElasticSmallStrain",
    "MaterialModel",
    "consistent_tangent",
    "strain_energy_density",
    "stress_measure",
]
