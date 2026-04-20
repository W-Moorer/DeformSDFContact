import numpy as np

from .evaluate_phi import eval_phi_quantities, eval_vector_function_data

_SENSITIVITY_CACHE = {}
_SENSITIVITY_CACHE_STATS = {
    "contact_sensitivity_call_count": 0,
    "contact_sensitivity_cache_hit_count": 0,
    "contact_sensitivity_cache_miss_count": 0,
}


def reset_sensitivity_cache_stats():
    _SENSITIVITY_CACHE.clear()
    for key in _SENSITIVITY_CACHE_STATS:
        _SENSITIVITY_CACHE_STATS[key] = 0


def snapshot_sensitivity_cache_stats():
    return dict(_SENSITIVITY_CACHE_STATS)


def _sensitivity_cache_key(X_c, cell_id, u_master, phi_function, globalize):
    X_key = tuple(np.round(np.asarray(X_c, dtype=np.float64), 12))
    return (id(u_master), id(phi_function), int(cell_id), bool(globalize), X_key)


def directional_dGu_due_to_geometry(X_c, cell_id, du_dir_function, solid_data, phi_data, profile=None):
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
    du_data = eval_vector_function_data(du_dir_function, X_c, cell_id, profile=profile)
    du_vec = du_dir_function.vector.array_r

    delta_F = du_data["grad"]
    delta_X = E @ du_vec
    delta_L = np.tensordot(delta_X, B_tensor, axes=(0, 1))

    return -(p @ F_inv @ delta_F @ E) - (p @ F_inv @ delta_L)


def compute_gap_sensitivities(
    X_c,
    cell_id,
    u_master,
    phi_function,
    profile=None,
    globalize=True,
    u_data=None,
):
    """
    Return the single-point gap geometry objects for the current benchmark.

    The finite-difference checks keep x_s fixed while perturbing the master
    field, so the geometric factors are built from E = -F^{-1} L with
    L = N_u(X_c).
    """
    _SENSITIVITY_CACHE_STATS["contact_sensitivity_call_count"] += 1
    cache_key = _sensitivity_cache_key(X_c, cell_id, u_master, phi_function, globalize)
    cached = _SENSITIVITY_CACHE.get(cache_key)
    if cached is not None:
        _SENSITIVITY_CACHE_STATS["contact_sensitivity_cache_hit_count"] += 1
        return dict(cached)

    _SENSITIVITY_CACHE_STATS["contact_sensitivity_cache_miss_count"] += 1

    phi_val, grad_phi, hess_phi, Nphi_row, Bphi_mat, phi_data = eval_phi_quantities(
        X_c, cell_id, phi_function, profile=profile, globalize=globalize
    )
    if u_data is None or (globalize and ("N_mat" not in u_data or "B_tensor" not in u_data)):
        u_data = eval_vector_function_data(u_master, X_c, cell_id, profile=profile, globalize=globalize)

    gdim = u_master.function_space.mesh.geometry.dim
    F = np.eye(gdim) + u_data["grad"]
    F_inv = np.linalg.inv(F)
    p = grad_phi

    normal_unnormalized = F_inv.T @ p
    normal = normal_unnormalized / np.linalg.norm(normal_unnormalized)

    L_local = None
    B_tensor_local = None
    if globalize:
        L = u_data["N_mat"]
        B_tensor = u_data["B_tensor"]
        E = -F_inv @ L
        G_a = Nphi_row
        G_u = p @ E
        H_uphi_g = E.T @ Bphi_mat
        L_local = u_data["N_local"]
        B_tensor_local = u_data["B_tensor_local"]
        E_local = -F_inv @ L_local
        G_a_local = phi_data["basis_local"]
        G_u_local = p @ E_local
        H_uphi_g_local = E_local.T @ phi_data["grad_local"]
    else:
        L = None
        B_tensor = None
        E = None
        L_local = u_data["N_local"]
        B_tensor_local = u_data["B_tensor_local"]
        E_local = -F_inv @ L_local
        G_a_local = phi_data["basis_local"]
        G_u_local = p @ E_local
        H_uphi_g_local = E_local.T @ phi_data["grad_local"]
        G_a = G_a_local
        G_u = G_u_local
        H_uphi_g = H_uphi_g_local

    H_uu_curv_local = None if hess_phi is None else E_local.T @ hess_phi @ E_local
    H_uu_curv = H_uu_curv_local if not globalize else (None if hess_phi is None else E.T @ hess_phi @ E)
    if hess_phi is None:
        GF = None
        GL = None
        H_uu_g = None
        GF_local = None
        GL_local = None
        H_uu_g_local = None
    else:
        C_local = np.tensordot(p @ F_inv, B_tensor_local, axes=(0, 0))
        GF_local = -(E_local.T @ C_local)
        GL_local = -(C_local.T @ E_local)
        H_uu_g_local = H_uu_curv_local + GF_local + GL_local
        if globalize:
            C = np.tensordot(p @ F_inv, B_tensor, axes=(0, 0))
            GF = -(E.T @ C)
            GL = -(C.T @ E)
            H_uu_g = H_uu_curv + GF + GL
        else:
            GF = GF_local
            GL = GL_local
            H_uu_g = H_uu_g_local

    out = {
        "g_n": phi_val,
        "normal": normal,
        "E": E,
        "E_local": E_local,
        "G_u": G_u,
        "G_a": G_a,
        "H_uphi_g": H_uphi_g,
        "H_uu_curv": H_uu_curv,
        "H_uu_g": H_uu_g,
        "GF": GF,
        "GL": GL,
        "G_u_local": G_u_local,
        "G_a_local": G_a_local,
        "H_uphi_g_local": H_uphi_g_local,
        "H_uu_curv_local": H_uu_curv_local,
        "H_uu_g_local": H_uu_g_local,
        "GF_local": GF_local,
        "GL_local": GL_local,
        "F": F,
        "F_inv": F_inv,
        "L": L,
        "L_local": L_local,
        "B_tensor": B_tensor,
        "B_tensor_local": B_tensor_local,
        "grad_phi": p,
        "hess_phi": hess_phi,
        "u_dofs": u_data.get("cell_dofs"),
        "phi_dofs": phi_data["cell_dofs"],
    }
    _SENSITIVITY_CACHE[cache_key] = out
    return dict(out)
