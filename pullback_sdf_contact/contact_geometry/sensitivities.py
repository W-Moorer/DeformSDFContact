import numpy as np

from .evaluate_phi import eval_phi_quantities, eval_vector_function_data


def compute_gap_sensitivities(X_c, cell_id, u_master, phi_function):
    """
    Return g_n, normal, E, G_u, G_a for the single-point benchmark.

    The current implementation treats x_s as fixed benchmark data, so
    delta X_c = -F(X_c)^{-1} delta u(X_c).
    """
    phi_val, grad_phi, hess_phi, Nphi_row, Bphi_mat = eval_phi_quantities(
        X_c, cell_id, phi_function
    )
    u_data = eval_vector_function_data(u_master, X_c, cell_id)

    gdim = u_master.function_space.mesh.geometry.dim
    F = np.eye(gdim) + u_data["grad"]
    p = grad_phi

    normal_unnormalized = np.linalg.solve(F.T, p)
    normal = normal_unnormalized / np.linalg.norm(normal_unnormalized)

    E = -np.linalg.solve(F, u_data["N_mat"])
    G_a = Nphi_row
    G_u = p @ E

    return phi_val, normal, E, G_u, G_a
