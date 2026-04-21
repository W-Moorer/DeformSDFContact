"""Reference geometry and warm-start predictor helpers for SDF-related work."""

from .coupling import (
    SDFDisplacementCouplingPointInput,
    SDFDisplacementCouplingPointResult,
    evaluate_sdf_displacement_coupling_point,
    linearized_metric_sensitivity_from_shape_gradients,
)
from .coupling_form_mapping import (
    SDFCouplingElementMapping,
    SDFCouplingQuadraturePointData,
    build_sdf_coupling_element_mapping,
)
from .coupling_local_loop import SDFCouplingLocalLoopResult, execute_sdf_coupling_local_loop
from .form_mapping import (
    ReinitializeElementMapping,
    ReinitializeFormAdapter,
    ReinitializeQuadraturePointData,
    build_reinitialize_element_mapping,
)
from .local_loop import execute_reinitialize_local_loop
from .predictor import (
    ReferencePredictorResult,
    metric_stretch_factor,
    predict_from_reference_geometry,
    predict_pullback_distance,
)
from .reinitialize import (
    eikonal_defect,
    reinitialize_element_residual_tangent,
    reinitialize_point_residual,
    reinitialize_point_tangent,
)
from .reference import ReferencePlane

__all__ = [
    "ReferencePlane",
    "ReferencePredictorResult",
    "SDFCouplingElementMapping",
    "SDFCouplingLocalLoopResult",
    "SDFCouplingQuadraturePointData",
    "SDFDisplacementCouplingPointInput",
    "SDFDisplacementCouplingPointResult",
    "ReinitializeElementMapping",
    "ReinitializeFormAdapter",
    "ReinitializeQuadraturePointData",
    "build_sdf_coupling_element_mapping",
    "build_reinitialize_element_mapping",
    "evaluate_sdf_displacement_coupling_point",
    "execute_reinitialize_local_loop",
    "execute_sdf_coupling_local_loop",
    "eikonal_defect",
    "linearized_metric_sensitivity_from_shape_gradients",
    "metric_stretch_factor",
    "predict_from_reference_geometry",
    "predict_pullback_distance",
    "reinitialize_element_residual_tangent",
    "reinitialize_point_residual",
    "reinitialize_point_tangent",
]
