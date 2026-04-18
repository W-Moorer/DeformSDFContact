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


def assemble_contact_residual_surface(quadrature_points, state, penalty):
    point_data = evaluate_surface_contact_points(quadrature_points, state)
    ndof_u = state["u"].vector.getLocalSize()
    residual = np.zeros(ndof_u, dtype=np.float64)
    active_count = 0
    total_gap_measure = 0.0

    for data in point_data:
        weight = data["quadrature_point"].weight
        point_residual, _, _ = contact_residual_single_point(data["g_n"], data["G_u"], penalty)
        residual += weight * point_residual
        if data["g_n"] < 0.0:
            active_count += 1
            total_gap_measure += weight * (-data["g_n"])

    return residual, active_count, total_gap_measure, point_data


def assemble_contact_tangent_uphi_surface(quadrature_points, state, penalty):
    point_data = evaluate_surface_contact_points(quadrature_points, state)
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()
    tangent = np.zeros((ndof_u, ndof_phi), dtype=np.float64)

    for data in point_data:
        weight = data["quadrature_point"].weight
        point_tangent, _, _ = contact_tangent_uphi_single_point(
            data["g_n"], data["G_u"], data["G_a"], data["H_uphi_g"], penalty
        )
        tangent += weight * point_tangent

    return tangent, point_data


def assemble_contact_tangent_uu_surface(quadrature_points, state, penalty):
    point_data = evaluate_surface_contact_points(quadrature_points, state)
    ndof_u = state["u"].vector.getLocalSize()
    tangent = np.zeros((ndof_u, ndof_u), dtype=np.float64)

    for data in point_data:
        weight = data["quadrature_point"].weight
        point_tangent, _, _ = contact_tangent_uu_single_point(
            data["g_n"], data["G_u"], data["H_uu_g"], penalty
        )
        tangent += weight * point_tangent

    return tangent, point_data
