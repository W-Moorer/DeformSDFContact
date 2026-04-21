"""Minimal local contact laws for backend-agnostic point kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def _as_scalar(value: object, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got {array.shape!r}")
    return float(array)


class ContactLaw(Protocol):
    """Protocol for minimal normal-contact local laws."""

    def evaluate(self, g_n: object) -> tuple[float, float]:
        """Return `(lambda_n, k_n)` for one scalar normal gap `g_n`."""


@dataclass(frozen=True)
class PenaltyContactLaw:
    """Normal penalty contact law active only in penetration.

    Semantics:

    - if `g_n < 0`, `lambda_n = penalty * (-g_n)` and `k_n = penalty`
    - if `g_n >= 0`, `lambda_n = 0` and `k_n = 0`

    Here `k_n` is the active-set stiffness with respect to the closure measure
    `(-g_n)`, so it is nonnegative whenever contact is active.
    """

    penalty: float

    def __post_init__(self) -> None:
        penalty = _as_scalar(self.penalty, "penalty")
        if penalty <= 0.0:
            raise ValueError(f"penalty must be strictly positive, got {penalty!r}")
        object.__setattr__(self, "penalty", penalty)

    def evaluate(self, g_n: object) -> tuple[float, float]:
        gap = _as_scalar(g_n, "g_n")
        if gap < 0.0:
            return self.penalty * (-gap), self.penalty
        return 0.0, 0.0


__all__ = ["ContactLaw", "PenaltyContactLaw"]
