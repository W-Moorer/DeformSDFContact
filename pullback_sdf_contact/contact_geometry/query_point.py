import numpy as np

from .evaluate_phi import eval_vector_function_data


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
