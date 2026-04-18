import numpy as np

from .evaluate_phi import eval_vector_function_data
from .sensitivities import compute_gap_sensitivities


def solve_query_point(x_s, u_master, candidate_cell, X_init, tol=1e-10, max_it=20):
    """
    Solve x_s = X + u_master(X) inside candidate_cell.
    Returns:
        X_c
        converged
        iters
    """
    X = np.asarray(X_init, dtype=np.float64).copy()
    identity = np.eye(u_master.function_space.mesh.geometry.dim)

    for it in range(max_it + 1):
        u_data = eval_vector_function_data(u_master, X, candidate_cell)
        residual = X + u_data["value"] - x_s
        res_norm = np.linalg.norm(residual)
        if res_norm < tol:
            return X, True, it

        jac = identity + u_data["grad"]
        delta = np.linalg.solve(jac, -residual)
        X += delta

        if np.linalg.norm(delta) < tol:
            return X, True, it + 1

    return X, False, max_it


def evaluate_contact_point_data(
    X_slave_ref,
    u_master,
    phi_function,
    candidate_cell,
    X_init=None,
    x_current=None,
):
    """
    Evaluate the contact geometry data for one slave quadrature point.
    """
    X_slave_ref = np.asarray(X_slave_ref, dtype=np.float64)
    if X_init is None:
        X_init = X_slave_ref

    if x_current is None:
        u_slave = eval_vector_function_data(u_master, X_slave_ref, candidate_cell)["value"]
        x_current = X_slave_ref + u_slave

    X_c, converged, iters = solve_query_point(
        np.asarray(x_current, dtype=np.float64),
        u_master,
        candidate_cell=candidate_cell,
        X_init=X_init,
    )
    sens = compute_gap_sensitivities(X_c, candidate_cell, u_master, phi_function)
    sens["X_c"] = X_c
    sens["converged"] = converged
    sens["iters"] = iters
    sens["x_current"] = np.asarray(x_current, dtype=np.float64)
    sens["X_slave_ref"] = X_slave_ref
    sens["candidate_cell"] = candidate_cell
    return sens
