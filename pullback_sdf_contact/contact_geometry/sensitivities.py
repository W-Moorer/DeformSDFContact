import numpy as np

from .evaluate_phi import eval_phi_quantities, eval_vector_function_data


def directional_dGu_due_to_geometry(X_c, cell_id, du_dir_function, solid_data, phi_data):
    """
    Return the geometric part of delta G_u[du_dir] for the current benchmark.

    The single-point finite-difference scripts keep x_s fixed while perturbing
    the master field, so E = -F^{-1} L with L = N_u(X_c), and

      delta E = -F^{-1} (delta F) E - F^{-1} delta L

    where delta L is induced by the query-point update delta X_c = E * delta d.
    """
    E = solid_data["E"]
    F_inv = solid_data["F_inv"]
    B_tensor = solid_data["B_tensor"]
    p = phi_data["grad_phi"]
    du_data = eval_vector_function_data(du_dir_function, X_c, cell_id)
    du_vec = du_dir_function.vector.array_r

    delta_F = du_data["grad"]
    delta_X = E @ du_vec
    delta_L = np.tensordot(delta_X, B_tensor, axes=(0, 1))

    return -(p @ F_inv @ delta_F @ E) - (p @ F_inv @ delta_L)


def compute_gap_sensitivities(X_c, cell_id, u_master, phi_function):
    """
    Return the single-point gap geometry objects for the current benchmark.

    The finite-difference checks keep x_s fixed while perturbing the master
    field, so the geometric factors are built from E = -F^{-1} L with
    L = N_u(X_c).
    """
    phi_val, grad_phi, hess_phi, Nphi_row, Bphi_mat = eval_phi_quantities(
        X_c, cell_id, phi_function
    )
    u_data = eval_vector_function_data(u_master, X_c, cell_id)

    gdim = u_master.function_space.mesh.geometry.dim
    F = np.eye(gdim) + u_data["grad"]
    F_inv = np.linalg.inv(F)
    p = grad_phi

    normal_unnormalized = F_inv.T @ p
    normal = normal_unnormalized / np.linalg.norm(normal_unnormalized)

    L = u_data["N_mat"]
    B_tensor = u_data["B_tensor"]
    E = -F_inv @ L
    G_a = Nphi_row
    G_u = p @ E
    H_uphi_g = E.T @ Bphi_mat

    H_uu_curv = None if hess_phi is None else E.T @ hess_phi @ E
    if hess_phi is None:
        GF = None
        GL = None
        H_uu_g = None
    else:
        # C_{i,a} = (p^T F^{-1})_k * dN_a^{(k)}/dX_i
        C = np.tensordot(p @ F_inv, B_tensor, axes=(0, 0))
        GF = -(E.T @ C)
        GL = -(C.T @ E)
        H_uu_g = H_uu_curv + GF + GL

    return {
        "g_n": phi_val,
        "normal": normal,
        "E": E,
        "G_u": G_u,
        "G_a": G_a,
        "H_uphi_g": H_uphi_g,
        "H_uu_curv": H_uu_curv,
        "H_uu_g": H_uu_g,
        "GF": GF,
        "GL": GL,
        "F": F,
        "F_inv": F_inv,
        "L": L,
        "B_tensor": B_tensor,
        "grad_phi": p,
        "hess_phi": hess_phi,
    }
