"""Backend-agnostic solid local kernels for the monolithic dry run."""

from .form_mapping import SolidElementMapping, SolidQuadraturePointData, build_solid_element_mapping
from .kernels import (
    SolidPointKernelInput,
    SolidPointKernelResult,
    evaluate_solid_point_kernel,
    plane_strain_constitutive_matrix,
    triangle_p1_B_matrix,
)
from .local_loop import SolidLocalLoopResult, execute_solid_local_loop

__all__ = [
    "SolidElementMapping",
    "SolidLocalLoopResult",
    "SolidPointKernelInput",
    "SolidPointKernelResult",
    "SolidQuadraturePointData",
    "build_solid_element_mapping",
    "evaluate_solid_point_kernel",
    "execute_solid_local_loop",
    "plane_strain_constitutive_matrix",
    "triangle_p1_B_matrix",
]
