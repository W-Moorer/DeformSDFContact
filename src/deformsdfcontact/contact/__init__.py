"""Backend-agnostic contact geometry helpers."""

from .form_mapping import (
    ContactQuadraturePointData,
    ContactSurfaceMapping,
    build_contact_surface_mapping,
)
from .geometry import (
    AffineMasterMap2D,
    AffinePhiField2D,
    ContactGeometryResult,
    evaluate_contact_geometry,
    gap_sensitivities,
    normal_gap,
    query_point,
)
from .geometry_second_order import (
    ContactSecondOrderGeometryResult,
    QuadraticPhiField2D,
    evaluate_contact_second_order_geometry,
    query_sensitivity_second_order,
    second_order_gap_geometry,
)
from .kernels import (
    ContactPointKernelInput,
    ContactPointKernelResult,
    evaluate_contact_point_kernel,
)
from .laws import ContactLaw, PenaltyContactLaw
from .local_loop import ContactLocalLoopResult, execute_contact_local_loop
from .surface_local_loop import ContactSurfaceLocalResult, execute_contact_surface_local_loop

__all__ = [
    "AffineMasterMap2D",
    "AffinePhiField2D",
    "ContactLaw",
    "ContactGeometryResult",
    "ContactLocalLoopResult",
    "ContactPointKernelInput",
    "ContactPointKernelResult",
    "ContactQuadraturePointData",
    "ContactSecondOrderGeometryResult",
    "ContactSurfaceLocalResult",
    "ContactSurfaceMapping",
    "PenaltyContactLaw",
    "QuadraticPhiField2D",
    "build_contact_surface_mapping",
    "evaluate_contact_geometry",
    "evaluate_contact_point_kernel",
    "evaluate_contact_second_order_geometry",
    "execute_contact_local_loop",
    "execute_contact_surface_local_loop",
    "gap_sensitivities",
    "normal_gap",
    "query_point",
    "query_sensitivity_second_order",
    "second_order_gap_geometry",
]
