import time

import numpy as np

from contact_geometry.evaluate_phi import (
    eval_vector_function_data,
    reset_geometry_eval_profile,
    snapshot_geometry_eval_profile,
)
from contact_geometry.query_point import (
    evaluate_contact_point_data,
    reset_query_cache_stats,
    snapshot_query_cache_stats,
)
from contact_geometry.sensitivities import (
    reset_sensitivity_cache_stats,
    snapshot_sensitivity_cache_stats,
)
from .laws import normal_law_penalty
from .single_point import (
    contact_residual_single_point,
    contact_tangent_uphi_single_point,
    contact_tangent_uu_single_point,
)


def _contact_profile_template():
    return {
        "contact_quadrature_loop_time": 0.0,
        "contact_geometry_eval_time": 0.0,
        "contact_gap_normal_eval_time": 0.0,
        "contact_active_filter_time": 0.0,
        "contact_local_residual_time": 0.0,
        "contact_local_tangent_uu_time": 0.0,
        "contact_local_tangent_uphi_time": 0.0,
        "contact_scatter_to_global_time": 0.0,
        "dofmap_lookup_time": 0.0,
        "surface_entity_iteration_time": 0.0,
        "basis_eval_or_tabulation_time": 0.0,
        "numpy_temp_allocation_time": 0.0,
        "reused_contact_cache": False,
        "reused_contact_buffers": False,
        "reused_contact_reference_eval": False,
        "contact_geometry_eval_call_count": 0,
        "contact_geometry_eval_avg_time": 0.0,
        "contact_query_time": 0.0,
        "contact_query_avg_time": 0.0,
        "contact_query_call_count": 0,
        "contact_query_cache_hit_count": 0,
        "contact_query_cache_miss_count": 0,
        "contact_sensitivity_time": 0.0,
        "contact_sensitivity_avg_time": 0.0,
        "contact_sensitivity_call_count": 0,
        "contact_sensitivity_cache_hit_count": 0,
        "contact_sensitivity_cache_miss_count": 0,
        "contact_local_tangent_uphi_call_count": 0,
        "contact_local_tangent_uu_call_count": 0,
        "cell_geometry_cache_hit_count": 0,
        "cell_geometry_cache_miss_count": 0,
        "function_cell_dof_cache_hit_count": 0,
        "function_cell_dof_cache_miss_count": 0,
        "vector_subfunction_cache_hit_count": 0,
        "vector_subfunction_cache_miss_count": 0,
    }


def _get_contact_cache(state, quadrature_points, ndof_u, ndof_phi):
    cache_store = state.setdefault("_contact_surface_cache", {})
    key = (len(quadrature_points), ndof_u, ndof_phi)
    cache = cache_store.get(key)
    created = False
    if cache is None:
        alloc_t0 = time.perf_counter()
        slave_ref_N_mats = []
        for quadrature_point in quadrature_points:
            slave_data = eval_vector_function_data(
                state["u"],
                quadrature_point.X_ref,
                quadrature_point.cell_id,
                globalize=False,
            )
            slave_ref_N_mats.append(
                (
                    slave_data["N_local"].copy(),
                    slave_data["cell_dofs"].copy(),
                )
            )
        cache = {
            "residual": np.zeros(ndof_u, dtype=np.float64),
            "tangent_uphi": np.zeros((ndof_u, ndof_phi), dtype=np.float64),
            "tangent_uu": np.zeros((ndof_u, ndof_u), dtype=np.float64),
            "outer_uphi": np.zeros((ndof_u, ndof_phi), dtype=np.float64),
            "outer_uu": np.zeros((ndof_u, ndof_u), dtype=np.float64),
            "slave_ref_N_mats": slave_ref_N_mats,
            "numpy_temp_allocation_time": time.perf_counter() - alloc_t0,
        }
        cache_store[key] = cache
        created = True
    return cache, created


def _slave_current_point(quadrature_point, state, *, slave_ref_N_mat=None):
    X_slave_ref = quadrature_point.X_ref
    if slave_ref_N_mat is None:
        u_slave = eval_vector_function_data(
            state["u"],
            X_slave_ref,
            quadrature_point.cell_id,
            globalize=False,
        )["value"]
    else:
        N_local, cell_dofs = slave_ref_N_mat
        u_slave = N_local.dot(state["u"].vector.array_r[cell_dofs])
    offset = np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64)
    return X_slave_ref + u_slave + offset


def evaluate_surface_contact_points(
    quadrature_points,
    state,
    *,
    profile_assembly_detail=False,
    slave_ref_N_mats=None,
):
    point_data = []
    X_init = None
    profile = _contact_profile_template()
    if profile_assembly_detail:
        reset_query_cache_stats()
        reset_sensitivity_cache_stats()
        reset_geometry_eval_profile()

    quadrature_t0 = time.perf_counter()
    for quadrature_point in quadrature_points:
        qp_t0 = time.perf_counter()
        slave_ref_N_mat = None if slave_ref_N_mats is None else slave_ref_N_mats[len(point_data)]
        geometry_t0 = time.perf_counter()
        x_current = _slave_current_point(
            quadrature_point,
            state,
            slave_ref_N_mat=slave_ref_N_mat,
        )
        data = evaluate_contact_point_data(
            quadrature_point.X_ref,
            state["u"],
            state["phi"],
            quadrature_point.cell_id,
            X_init=X_init if X_init is not None else quadrature_point.X_ref,
            x_current=x_current,
            profile=profile if profile_assembly_detail else None,
            globalize=False,
        )
        profile["contact_geometry_eval_time"] += time.perf_counter() - geometry_t0
        profile["contact_geometry_eval_call_count"] += 1
        gap_t0 = time.perf_counter()
        data["quadrature_point"] = quadrature_point
        point_data.append(data)
        if data["converged"]:
            X_init = data["X_c"]
        profile["contact_gap_normal_eval_time"] += time.perf_counter() - gap_t0
        profile["surface_entity_iteration_time"] += time.perf_counter() - qp_t0
    profile["contact_quadrature_loop_time"] = time.perf_counter() - quadrature_t0
    if profile["contact_geometry_eval_call_count"] > 0:
        profile["contact_geometry_eval_avg_time"] = (
            profile["contact_geometry_eval_time"] / profile["contact_geometry_eval_call_count"]
        )
    if profile_assembly_detail:
        for stats in (
            snapshot_query_cache_stats(),
            snapshot_sensitivity_cache_stats(),
            snapshot_geometry_eval_profile(),
        ):
            for key, value in stats.items():
                if key in profile:
                    profile[key] += value
                else:
                    profile[key] = value
        query_calls = profile.get("contact_query_call_count", 0)
        if query_calls > 0:
            profile["contact_query_avg_time"] = profile.get("contact_query_time", 0.0) / query_calls
        sensitivity_calls = profile.get("contact_sensitivity_call_count", 0)
        if sensitivity_calls > 0:
            profile["contact_sensitivity_avg_time"] = (
                profile.get("contact_sensitivity_time", 0.0) / sensitivity_calls
            )

    return point_data, profile


def collect_contact_diagnostics_surface(
    quadrature_points,
    point_data,
    residual=None,
    *,
    profile_assembly_detail=False,
):
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
        "contact_candidate_point_count": len(point_data),
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
    diagnostics["contact_active_fraction"] = (
        0.0
        if diagnostics["contact_candidate_point_count"] == 0
        else float(diagnostics["active_contact_points"]) / float(diagnostics["contact_candidate_point_count"])
    )
    if profile_assembly_detail:
        active_u_dofs = set()
        active_phi_dofs = set()
        for data in active_points:
            u_dofs = data.get("u_dofs")
            phi_dofs = data.get("phi_dofs")
            if u_dofs is not None:
                active_u_dofs.update(int(v) for v in np.asarray(u_dofs, dtype=np.int64).ravel())
            if phi_dofs is not None:
                active_phi_dofs.update(int(v) for v in np.asarray(phi_dofs, dtype=np.int64).ravel())
        diagnostics["K_phi_u_rows_touched_by_active_contact"] = int(len(active_phi_dofs))
        diagnostics["K_phi_u_cols_touched_by_active_contact"] = int(len(active_u_dofs))
    return diagnostics


def assemble_contact_contributions_surface(
    quadrature_points,
    state,
    penalty,
    need_residual=True,
    need_tangent_uu=False,
    need_tangent_uphi=False,
    need_diagnostics=True,
    build_path="current",
    profile_assembly_detail=False,
):
    """Assemble the requested slave-surface contact contributions."""
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()
    cache = None
    created_cache = False
    profile = _contact_profile_template()

    if build_path == "optimized":
        cache, created_cache = _get_contact_cache(state, quadrature_points, ndof_u, ndof_phi)
        profile["numpy_temp_allocation_time"] += cache.get("numpy_temp_allocation_time", 0.0)
        profile["reused_contact_cache"] = not created_cache
        profile["reused_contact_buffers"] = True
        profile["reused_contact_reference_eval"] = True
        residual = cache["residual"] if need_residual else None
        tangent_uphi = cache["tangent_uphi"] if need_tangent_uphi else None
        tangent_uu = cache["tangent_uu"] if need_tangent_uu else None
        if residual is not None:
            residual.fill(0.0)
        if tangent_uphi is not None:
            tangent_uphi.fill(0.0)
        if tangent_uu is not None:
            tangent_uu.fill(0.0)
        slave_ref_N_mats = cache["slave_ref_N_mats"]
        outer_uphi = cache["outer_uphi"] if need_tangent_uphi else None
        outer_uu = cache["outer_uu"] if need_tangent_uu else None
    else:
        alloc_t0 = time.perf_counter()
        residual = np.zeros(ndof_u, dtype=np.float64) if need_residual else None
        tangent_uphi = (
            np.zeros((ndof_u, ndof_phi), dtype=np.float64) if need_tangent_uphi else None
        )
        tangent_uu = np.zeros((ndof_u, ndof_u), dtype=np.float64) if need_tangent_uu else None
        outer_uphi = np.zeros((ndof_u, ndof_phi), dtype=np.float64) if need_tangent_uphi else None
        outer_uu = np.zeros((ndof_u, ndof_u), dtype=np.float64) if need_tangent_uu else None
        slave_ref_N_mats = None
        profile["numpy_temp_allocation_time"] += time.perf_counter() - alloc_t0

    point_data, point_profile = evaluate_surface_contact_points(
        quadrature_points,
        state,
        profile_assembly_detail=profile_assembly_detail,
        slave_ref_N_mats=slave_ref_N_mats,
    )
    for key, value in point_profile.items():
        if key in profile:
            profile[key] += value

    for data in point_data:
        weight = data["quadrature_point"].weight
        filter_t0 = time.perf_counter()
        lam, kn = normal_law_penalty(data["g_n"], penalty)
        inactive = lam == 0.0 and kn == 0.0
        profile["contact_active_filter_time"] += time.perf_counter() - filter_t0
        if inactive:
            continue
        u_dofs = data.get("u_dofs")
        phi_dofs = data.get("phi_dofs")
        G_u_local = data.get("G_u_local", data["G_u"])
        G_a_local = data.get("G_a_local", data["G_a"])
        H_uphi_g_local = data.get("H_uphi_g_local", data["H_uphi_g"])
        H_uu_g_local = data.get("H_uu_g_local", data["H_uu_g"])
        if need_residual:
            local_t0 = time.perf_counter()
            if u_dofs is None:
                residual += weight * lam * data["G_u"]
            else:
                residual[u_dofs] += weight * lam * G_u_local
            profile["contact_local_residual_time"] += time.perf_counter() - local_t0
        if need_tangent_uphi:
            local_t0 = time.perf_counter()
            profile["contact_local_tangent_uphi_call_count"] += 1
            if u_dofs is None or phi_dofs is None:
                tangent_uphi += weight * lam * data["H_uphi_g"]
                np.multiply(data["G_u"][:, None], data["G_a"][None, :], out=outer_uphi)
                tangent_uphi -= weight * kn * outer_uphi
            else:
                if outer_uphi is not None and outer_uphi.shape == (len(u_dofs), len(phi_dofs)):
                    np.multiply(G_u_local[:, None], G_a_local[None, :], out=outer_uphi)
                    local_outer = outer_uphi
                else:
                    local_outer = np.multiply(G_u_local[:, None], G_a_local[None, :])
                tangent_uphi[np.ix_(u_dofs, phi_dofs)] += weight * lam * H_uphi_g_local
                tangent_uphi[np.ix_(u_dofs, phi_dofs)] -= weight * kn * local_outer
            profile["contact_local_tangent_uphi_time"] += time.perf_counter() - local_t0
        if need_tangent_uu:
            local_t0 = time.perf_counter()
            profile["contact_local_tangent_uu_call_count"] += 1
            if u_dofs is None:
                tangent_uu += weight * lam * data["H_uu_g"]
                np.multiply(data["G_u"][:, None], data["G_u"][None, :], out=outer_uu)
                tangent_uu -= weight * kn * outer_uu
            else:
                if outer_uu is not None and outer_uu.shape == (len(u_dofs), len(u_dofs)):
                    np.multiply(G_u_local[:, None], G_u_local[None, :], out=outer_uu)
                    local_outer = outer_uu
                else:
                    local_outer = np.multiply(G_u_local[:, None], G_u_local[None, :])
                tangent_uu[np.ix_(u_dofs, u_dofs)] += weight * lam * H_uu_g_local
                tangent_uu[np.ix_(u_dofs, u_dofs)] -= weight * kn * local_outer
            profile["contact_local_tangent_uu_time"] += time.perf_counter() - local_t0
        scatter_t0 = time.perf_counter()
        profile["contact_scatter_to_global_time"] += time.perf_counter() - scatter_t0

    diagnostics = None
    if need_diagnostics:
        diagnostics = collect_contact_diagnostics_surface(
            quadrature_points,
            point_data,
            residual,
            profile_assembly_detail=profile_assembly_detail,
        )

    return {
        "R_u_c": residual,
        "K_uphi_c": tangent_uphi,
        "K_uu_c": tangent_uu,
        "diagnostics": diagnostics,
        "point_data": point_data,
        "profiling": profile,
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
