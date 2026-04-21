"""DOLFINx 0.3.0-compatible transition assembly adapter."""

from .assembly import Dolfinx0p3MonolithicDryRunResult, assemble_monolithic_dry_run
from .callables import Dolfinx0p3ResidualJacobianCallables
from .contact_adapter import assemble_contact_local_contributions
from .contact_pairing_backend import assemble_contact_pairing_local_contributions
from .contact_query_backend import assemble_contact_query_local_contributions
from .problem import (
    TransitionMonolithicProblem,
    build_contact_strip_benchmark,
    build_unit_square_toy_problem,
)
from .sdf_adapter import assemble_sdf_local_contributions
from .solid_adapter import assemble_solid_local_contributions

__all__ = [
    "Dolfinx0p3ResidualJacobianCallables",
    "Dolfinx0p3MonolithicDryRunResult",
    "TransitionMonolithicProblem",
    "assemble_contact_local_contributions",
    "assemble_contact_pairing_local_contributions",
    "assemble_contact_query_local_contributions",
    "assemble_monolithic_dry_run",
    "assemble_sdf_local_contributions",
    "assemble_solid_local_contributions",
    "build_contact_strip_benchmark",
    "build_unit_square_toy_problem",
]
