"""Small contact-summary objects for prototype benchmark diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...assembly import ContactLocalContribution


def _optional_index(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _as_point(values: object | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {array.shape!r}")
    return array.copy()


@dataclass(frozen=True)
class ContactPointObservation:
    """One local contact observation used for prototype benchmark diagnostics."""

    slave_facet: int
    slave_cell: int
    master_facet: int | None
    master_cell: int | None
    candidate_count: int
    gap_n: float
    lambda_n: float
    weight: float
    reaction_scalar: float
    active: bool
    slave_point: np.ndarray | None = None
    query_point: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "slave_facet", int(self.slave_facet))
        object.__setattr__(self, "slave_cell", int(self.slave_cell))
        object.__setattr__(self, "master_facet", _optional_index(self.master_facet))
        object.__setattr__(self, "master_cell", _optional_index(self.master_cell))
        object.__setattr__(self, "candidate_count", int(self.candidate_count))
        object.__setattr__(self, "gap_n", float(self.gap_n))
        object.__setattr__(self, "lambda_n", float(self.lambda_n))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "reaction_scalar", float(self.reaction_scalar))
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "slave_point", _as_point(self.slave_point))
        object.__setattr__(self, "query_point", _as_point(self.query_point))


@dataclass(frozen=True)
class ContactAssemblySummary:
    """Prototype summary of one assembled contact backend realization."""

    backend_name: str
    point_observations: tuple[ContactPointObservation, ...]
    candidate_count: int
    owned_pair_count: int
    active_point_count: int
    gap_min: float | None
    gap_max: float | None
    gap_mean: float | None
    lambda_sum: float
    reaction_sum: float

    @classmethod
    def from_observations(
        cls,
        backend_name: str,
        observations: tuple[ContactPointObservation, ...],
    ) -> "ContactAssemblySummary":
        candidate_count = int(sum(item.candidate_count for item in observations))
        owned_pair_count = int(sum(1 for item in observations if item.master_facet is not None))
        active_point_count = int(sum(1 for item in observations if item.active))
        if observations:
            gaps = np.array([item.gap_n for item in observations], dtype=float)
            gap_min = float(np.min(gaps))
            gap_max = float(np.max(gaps))
            gap_mean = float(np.mean(gaps))
        else:
            gap_min = None
            gap_max = None
            gap_mean = None
        lambda_sum = float(sum(item.lambda_n * item.weight for item in observations))
        reaction_sum = float(sum(item.reaction_scalar for item in observations))
        return cls(
            backend_name=str(backend_name),
            point_observations=tuple(observations),
            candidate_count=candidate_count,
            owned_pair_count=owned_pair_count,
            active_point_count=active_point_count,
            gap_min=gap_min,
            gap_max=gap_max,
            gap_mean=gap_mean,
            lambda_sum=lambda_sum,
            reaction_sum=reaction_sum,
        )


@dataclass(frozen=True)
class ContactAssemblyResult:
    """Local contributions plus prototype diagnostic summary for one backend."""

    contributions: tuple[ContactLocalContribution, ...]
    summary: ContactAssemblySummary


__all__ = [
    "ContactAssemblyResult",
    "ContactAssemblySummary",
    "ContactPointObservation",
]
