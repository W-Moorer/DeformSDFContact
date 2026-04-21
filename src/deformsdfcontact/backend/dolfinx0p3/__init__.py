"""DOLFINx 0.3.0-compatible transition assembly adapter."""

from .assembly import Dolfinx0p3MonolithicDryRunResult, assemble_monolithic_dry_run
from .contact_adapter import assemble_contact_local_contributions
from .sdf_adapter import assemble_sdf_local_contributions
from .solid_adapter import assemble_solid_local_contributions

__all__ = [
    "Dolfinx0p3MonolithicDryRunResult",
    "assemble_contact_local_contributions",
    "assemble_monolithic_dry_run",
    "assemble_sdf_local_contributions",
    "assemble_solid_local_contributions",
]
