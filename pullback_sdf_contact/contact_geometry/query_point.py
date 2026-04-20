import time

import numpy as np

from .evaluate_phi import eval_vector_function_data
from .sensitivities import compute_gap_sensitivities

_QUERY_CACHE = {}
_QUERY_CACHE_STATS = {
    "contact_query_call_count": 0,
    "contact_query_cache_hit_count": 0,
    "contact_query_cache_miss_count": 0,
}


def reset_query_cache_stats():
    _QUERY_CACHE.clear()
    for key in _QUERY_CACHE_STATS:
        _QUERY_CACHE_STATS[key] = 0


def snapshot_query_cache_stats():
    return dict(_QUERY_CACHE_STATS)


def _query_cache_key(x_s, candidate_cell, X_init, u_master):
    x_s_key = tuple(np.round(np.asarray(x_s, dtype=np.float64), 12))
    x_init_key = tuple(np.round(np.asarray(X_init, dtype=np.float64), 12))
    return (id(u_master), int(candidate_cell), x_s_key, x_init_key)


def solve_query_point(x_s, u_master, candidate_cell, X_init, tol=1e-10, max_it=20, profile=None):
    """
    Solve x_s = X + u_master(X) inside candidate_cell.
    Returns:
        X_c
        converged
        iters
    """
    _QUERY_CACHE_STATS["contact_query_call_count"] += 1
    cache_key = _query_cache_key(x_s, candidate_cell, X_init, u_master)
    cached = _QUERY_CACHE.get(cache_key)
    if cached is not None:
        _QUERY_CACHE_STATS["contact_query_cache_hit_count"] += 1
        return cached["X_c"].copy(), cached["converged"], cached["iters"], cached["u_data"]

    _QUERY_CACHE_STATS["contact_query_cache_miss_count"] += 1
    X = np.asarray(X_init, dtype=np.float64).copy()
    identity = np.eye(u_master.function_space.mesh.geometry.dim, dtype=np.float64)
    last_u_data = None

    for it in range(max_it + 1):
        u_data = eval_vector_function_data(
            u_master,
            X,
            candidate_cell,
            profile=profile,
            globalize=False,
        )
        last_u_data = u_data
        residual = X + u_data["value"] - x_s
        res_norm = np.linalg.norm(residual)
        if res_norm < tol:
            out = {
                "X_c": X.copy(),
                "converged": True,
                "iters": it,
                "u_data": u_data,
            }
            _QUERY_CACHE[cache_key] = out
            return out["X_c"].copy(), out["converged"], out["iters"], out["u_data"]

        jac = identity + u_data["grad"]
        delta = np.linalg.solve(jac, -residual)
        X += delta

        if np.linalg.norm(delta) < tol:
            u_data = eval_vector_function_data(
                u_master,
                X,
                candidate_cell,
                profile=profile,
                globalize=False,
            )
            out = {
                "X_c": X.copy(),
                "converged": True,
                "iters": it + 1,
                "u_data": u_data,
            }
            _QUERY_CACHE[cache_key] = out
            return out["X_c"].copy(), out["converged"], out["iters"], out["u_data"]

    out = {
        "X_c": X.copy(),
        "converged": False,
        "iters": max_it,
        "u_data": last_u_data,
    }
    _QUERY_CACHE[cache_key] = out
    return out["X_c"].copy(), out["converged"], out["iters"], out["u_data"]


def evaluate_contact_point_data(
    X_slave_ref,
    u_master,
    phi_function,
    candidate_cell,
    X_init=None,
    x_current=None,
    profile=None,
    globalize=True,
):
    """
    Evaluate the contact geometry data for one slave quadrature point.
    """
    X_slave_ref = np.asarray(X_slave_ref, dtype=np.float64)
    if X_init is None:
        X_init = X_slave_ref

    if x_current is None:
        u_slave = eval_vector_function_data(
            u_master,
            X_slave_ref,
            candidate_cell,
            profile=profile,
            globalize=False,
        )["value"]
        x_current = X_slave_ref + u_slave

    query_t0 = np.nan
    if profile is not None:
        query_t0 = time.perf_counter()
    X_c, converged, iters, u_data = solve_query_point(
        np.asarray(x_current, dtype=np.float64),
        u_master,
        candidate_cell=candidate_cell,
        X_init=X_init,
        profile=profile,
    )
    if profile is not None:
        profile["contact_query_time"] = profile.get("contact_query_time", 0.0) + (
            time.perf_counter() - query_t0
        )
    sens_t0 = np.nan
    if profile is not None:
        sens_t0 = time.perf_counter()
    sens = compute_gap_sensitivities(
        X_c,
        candidate_cell,
        u_master,
        phi_function,
        profile=profile,
        globalize=globalize,
        u_data=u_data,
    )
    if profile is not None:
        profile["contact_sensitivity_time"] = profile.get("contact_sensitivity_time", 0.0) + (
            time.perf_counter() - sens_t0
        )
    sens["X_c"] = X_c
    sens["converged"] = converged
    sens["iters"] = iters
    sens["x_current"] = np.asarray(x_current, dtype=np.float64)
    sens["X_slave_ref"] = X_slave_ref
    sens["candidate_cell"] = candidate_cell
    return sens
