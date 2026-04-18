import numpy as np

from contact_geometry.evaluate_phi import eval_vector_function_data
from contact_geometry.query_point import evaluate_contact_point_data
from .single_point import (
    contact_residual_single_point,
    contact_tangent_uphi_single_point,
    contact_tangent_uu_single_point,
)


def _slave_current_point(quadrature_point, state):
    X_slave_ref = quadrature_point.X_ref
    u_slave = eval_vector_function_data(state["u"], X_slave_ref, quadrature_point.cell_id)["value"]
    offset = np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64)
    return X_slave_ref + u_slave + offset


def evaluate_surface_contact_points(quadrature_points, state):
    point_data = []
    X_init = None

    for quadrature_point in quadrature_points:
        x_current = _slave_current_point(quadrature_point, state)
        data = evaluate_contact_point_data(
            quadrature_point.X_ref,
            state["u"],
            state["phi"],
            quadrature_point.cell_id,
            X_init=X_init if X_init is not None else quadrature_point.X_ref,
            x_current=x_current,
        )
        data["quadrature_point"] = quadrature_point
        point_data.append(data)
        if data["converged"]:
            X_init = data["X_c"]

    return point_data


def collect_contact_diagnostics_surface(quadrature_points, point_data, residual=None):
    """Collect aggregated diagnostics for the current slave surface evaluation."""
    active_points = [data for data in point_data if data["g_n"] < 0.0]
    negative_gap_sum = float(
        sum(data["quadrature_point"].weight * (-data["g_n"]) for data in active_points)
    )
    penetrations = [(-data["g_n"]) for data in active_points]
    weighted_active_area = float(sum(data["quadrature_point"].weight for data in active_points))
    reference_slave_area = float(sum(qp.weight for qp in quadrature_points))
    num_slave_facets = len({qp.facet_id for qp in quadrature_points})
    diagnostics = {
        "active_contact_points": len(active_points),
        "negative_gap_sum": negative_gap_sum,
        "reference_slave_area": reference_slave_area,
        "num_slave_facets": num_slave_facets,
        "num_slave_quadrature_points": len(quadrature_points),
        "max_penetration": max(penetrations) if penetrations else 0.0,
        "mean_penetration": (
            negative_gap_sum / weighted_active_area if weighted_active_area > 0.0 else 0.0
        ),
        "reaction_norm": float(np.linalg.norm(residual)) if residual is not None else 0.0,
    }
    return diagnostics


def assemble_contact_contributions_surface(
    quadrature_points,
    state,
    penalty,
    need_residual=True,
    need_tangent_uu=False,
    need_tangent_uphi=False,
    need_diagnostics=True,
):
    """Assemble the requested slave-surface contact contributions."""
    point_data = evaluate_surface_contact_points(quadrature_points, state)
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()

    residual = np.zeros(ndof_u, dtype=np.float64) if need_residual else None
    tangent_uphi = (
        np.zeros((ndof_u, ndof_phi), dtype=np.float64) if need_tangent_uphi else None
    )
    tangent_uu = np.zeros((ndof_u, ndof_u), dtype=np.float64) if need_tangent_uu else None

    for data in point_data:
        weight = data["quadrature_point"].weight
        if need_residual:
            point_residual, _, _ = contact_residual_single_point(data["g_n"], data["G_u"], penalty)
            residual += weight * point_residual
        if need_tangent_uphi:
            point_tangent_uphi, _, _ = contact_tangent_uphi_single_point(
                data["g_n"], data["G_u"], data["G_a"], data["H_uphi_g"], penalty
            )
            tangent_uphi += weight * point_tangent_uphi
        if need_tangent_uu:
            point_tangent_uu, _, _ = contact_tangent_uu_single_point(
                data["g_n"], data["G_u"], data["H_uu_g"], penalty
            )
            tangent_uu += weight * point_tangent_uu

    diagnostics = None
    if need_diagnostics:
        diagnostics = collect_contact_diagnostics_surface(quadrature_points, point_data, residual)

    return {
        "R_u_c": residual,
        "K_uphi_c": tangent_uphi,
        "K_uu_c": tangent_uu,
        "diagnostics": diagnostics,
        "point_data": point_data,
    }


def assemble_contact_residual_surface(quadrature_points, state, penalty):
    out = assemble_contact_contributions_surface(
        quadrature_points, state, penalty, need_residual=True, need_diagnostics=True
    )
    diagnostics = out["diagnostics"]
    return (
        out["R_u_c"],
        diagnostics["active_contact_points"],
        diagnostics["negative_gap_sum"],
        out["point_data"],
    )


def assemble_contact_tangent_uphi_surface(quadrature_points, state, penalty):
    out = assemble_contact_contributions_surface(
        quadrature_points, state, penalty, need_tangent_uphi=True, need_diagnostics=False
    )
    return out["K_uphi_c"], out["point_data"]


def assemble_contact_tangent_uu_surface(quadrature_points, state, penalty):
    out = assemble_contact_contributions_surface(
        quadrature_points, state, penalty, need_tangent_uu=True, need_diagnostics=False
    )
    return out["K_uu_c"], out["point_data"]
