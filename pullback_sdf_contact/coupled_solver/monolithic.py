import csv
import os
import time
import numpy as np

from dolfinx import cpp, fem
from petsc4py import PETSc

from contact_mechanics.assembled_surface import assemble_contact_contributions_surface
from solid.solve import _owned_bc_dofs, assemble_linear_solid_system, dense_array_from_petsc_mat

RECOMMENDED_MONOLITHIC_BACKEND = "petsc_block"
RECOMMENDED_MONOLITHIC_MAX_NEWTON_ITER = 20
RECOMMENDED_MONOLITHIC_LINE_SEARCH = True
RECOMMENDED_MONOLITHIC_INITIAL_DAMPING = 1.0
RECOMMENDED_MONOLITHIC_MAX_BACKTRACKS = 8
RECOMMENDED_MONOLITHIC_BACKTRACK_FACTOR = 0.5
RECOMMENDED_MONOLITHIC_TOL_RES = 1e-8
RECOMMENDED_MONOLITHIC_TOL_INC = 1e-8
RECOMMENDED_MONOLITHIC_LINEAR_SOLVER_MODE = "lu"
RECOMMENDED_MONOLITHIC_KSP_TYPE = "gmres"
RECOMMENDED_MONOLITHIC_PC_TYPE = "fieldsplit"
RECOMMENDED_MONOLITHIC_BLOCK_PC_NAME = "fieldsplit_multiplicative_ilu"
RECOMMENDED_MONOLITHIC_KSP_RTOL = 1e-10
RECOMMENDED_MONOLITHIC_KSP_ATOL = 1e-12
RECOMMENDED_MONOLITHIC_KSP_MAX_IT = 400
RECOMMENDED_MONOLITHIC_PHI_CACHE_PRIME = True
RECOMMENDED_MONOLITHIC_PHI_SCATTER_REUSE = True
RECOMMENDED_MONOLITHIC_PHI_PROFILE_MODE = "light"
RECOMMENDED_MONOLITHIC_PHI_MATRIX_ASSEMBLY_BACKEND = "python"

BLOCK_PC_CONFIGS = {
    "fieldsplit_additive_ilu": {
        "split_type": "ADDITIVE",
        "sub_pc_type": "ilu",
    },
    "fieldsplit_multiplicative_ilu": {
        "split_type": "MULTIPLICATIVE",
        "sub_pc_type": "ilu",
    },
    "fieldsplit_symmetric_multiplicative_ilu": {
        "split_type": "SYMMETRIC_MULTIPLICATIVE",
        "sub_pc_type": "ilu",
    },
    "fieldsplit_schur_lower_selfp_ilu": {
        "split_type": "SCHUR",
        "schur_fact_type": "LOWER",
        "schur_pre_type": "SELFP",
        "sub_pc_type": "ilu",
    },
    "fieldsplit_schur_upper_selfp_ilu": {
        "split_type": "SCHUR",
        "schur_fact_type": "UPPER",
        "schur_pre_type": "SELFP",
        "sub_pc_type": "ilu",
    },
}


def recommended_monolithic_contact_options(**overrides):
    """Return the current project-recommended monolithic settings."""
    options = {
        "backend": RECOMMENDED_MONOLITHIC_BACKEND,
        "max_newton_iter": RECOMMENDED_MONOLITHIC_MAX_NEWTON_ITER,
        "line_search": RECOMMENDED_MONOLITHIC_LINE_SEARCH,
        "initial_damping": RECOMMENDED_MONOLITHIC_INITIAL_DAMPING,
        "max_backtracks": RECOMMENDED_MONOLITHIC_MAX_BACKTRACKS,
        "backtrack_factor": RECOMMENDED_MONOLITHIC_BACKTRACK_FACTOR,
        "tol_res": RECOMMENDED_MONOLITHIC_TOL_RES,
        "tol_inc": RECOMMENDED_MONOLITHIC_TOL_INC,
        "linear_solver_mode": RECOMMENDED_MONOLITHIC_LINEAR_SOLVER_MODE,
        "ksp_type": RECOMMENDED_MONOLITHIC_KSP_TYPE,
        "pc_type": RECOMMENDED_MONOLITHIC_PC_TYPE,
        "block_pc_name": RECOMMENDED_MONOLITHIC_BLOCK_PC_NAME,
        "build_path": "current",
        "reuse_ksp": False,
        "reuse_matrix_pattern": False,
        "reuse_fieldsplit_is": False,
        "profile_assembly_detail": False,
        "ksp_rtol": RECOMMENDED_MONOLITHIC_KSP_RTOL,
        "ksp_atol": RECOMMENDED_MONOLITHIC_KSP_ATOL,
        "ksp_max_it": RECOMMENDED_MONOLITHIC_KSP_MAX_IT,
        "phi_cache_prime": RECOMMENDED_MONOLITHIC_PHI_CACHE_PRIME,
        "phi_scatter_reuse": RECOMMENDED_MONOLITHIC_PHI_SCATTER_REUSE,
        "phi_profile_mode": RECOMMENDED_MONOLITHIC_PHI_PROFILE_MODE,
        "phi_matrix_assembly_backend": RECOMMENDED_MONOLITHIC_PHI_MATRIX_ASSEMBLY_BACKEND,
    }
    options.update(overrides)
    return options


def monolithic_block_pc_names():
    return sorted(BLOCK_PC_CONFIGS.keys())


def get_petsc_runtime_info():
    """Return runtime PETSc metadata and currently configured block-PC names."""
    version = PETSc.Sys.getVersion()
    return {
        "petsc_version": ".".join(str(part) for part in version),
        "block_pc_names": monolithic_block_pc_names(),
    }


def _safe_enum_lookup(enum_cls, name, *, feature_name):
    if hasattr(enum_cls, name):
        return getattr(enum_cls, name)
    raise RuntimeError(
        f"PETSc runtime does not expose {feature_name}='{name}' for {enum_cls.__name__}"
    )


def _trace_reuse_stage(stage):
    if os.environ.get("MONOLITHIC_REUSE_TRACE", "").strip():
        print(f"[monolithic-reuse-trace] {stage}", flush=True)


def _state_dof_layout(state):
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()
    return {
        "ndof_u": int(ndof_u),
        "ndof_phi": int(ndof_phi),
        "total_dofs": int(ndof_u + ndof_phi),
        "mesh_resolution": state.get("mesh_resolution", ""),
    }


def _compile_form(expr):
    if hasattr(fem, "form") and callable(fem.form):
        return fem.form(expr)
    return fem.Form(expr)


def _get_or_create_phi_form_cache(state):
    cache = state.get("_monolithic_phi_form_cache")
    if cache is None:
        cache = {
            "R_phi_form": _compile_form(state["R_phi_form"]),
            "K_phi_u_form": _compile_form(state["K_phi_u_form"]),
            "K_phi_phi_form": _compile_form(state["K_phi_phi_form"]),
        }
        state["_monolithic_phi_form_cache"] = cache
    return cache


def _phi_profile_template(profile_phi_detail=False):
    profile = {
        "phi_form_assembly_time": 0.0,
        "phi_matrix_extract_time": 0.0,
        "phi_matrix_convert_time": 0.0,
        "phi_matrix_extract_or_convert_time": 0.0,
        "phi_rhs_assembly_time": 0.0,
        "phi_rhs_extract_time": 0.0,
        "phi_rhs_extract_or_convert_time": 0.0,
        "phi_dof_lookup_time": 0.0,
        "phi_geometry_helper_time": 0.0,
        "phi_basis_tabulation_time": 0.0,
        "phi_form_call_count": 0,
        "phi_form_cache_hit_count": 0,
        "phi_form_cache_miss_count": 0,
        "phi_matrix_extract_call_count": 0,
        "phi_matrix_convert_call_count": 0,
        "phi_rhs_extract_call_count": 0,
        "phi_dof_lookup_call_count": 0,
        "phi_basis_tabulation_call_count": 0,
        "phi_residual_form_assembly_time": 0.0,
        "phi_residual_extract_time": 0.0,
        "phi_kphiu_form_assembly_time": 0.0,
        "phi_kphiu_extract_time": 0.0,
        "phi_kphiu_convert_time": 0.0,
        "phi_kphiphi_form_assembly_time": 0.0,
        "phi_kphiphi_extract_time": 0.0,
        "phi_kphiphi_convert_time": 0.0,
        "phi_residual_call_count": 0,
        "phi_kphiu_call_count": 0,
        "phi_kphiphi_call_count": 0,
        "reused_phi_rhs_vec": False,
        "reused_phi_kphiu_mat": False,
        "reused_phi_kphiphi_mat": False,
        "reused_phi_dense_buffers": False,
    }
    if profile_phi_detail:
        profile.update(
            {
                "phi_form_time_R_phi": 0.0,
                "phi_form_time_K_phi_u": 0.0,
                "phi_form_time_K_phi_phi": 0.0,
                "phi_extract_time_K_phi_u": 0.0,
                "phi_convert_time_K_phi_u": 0.0,
                "phi_extract_time_K_phi_phi": 0.0,
                "phi_convert_time_K_phi_phi": 0.0,
                "phi_form_call_count_R_phi": 0,
                "phi_form_call_count_K_phi_u": 0,
                "phi_form_call_count_K_phi_phi": 0,
                "phi_matrix_extract_call_count_K_phi_u": 0,
                "phi_matrix_convert_call_count_K_phi_u": 0,
                "phi_matrix_extract_call_count_K_phi_phi": 0,
                "phi_matrix_convert_call_count_K_phi_phi": 0,
                "phi_scatter_pattern_build_time": 0.0,
                "phi_scatter_pattern_build_count": 0,
            }
        )
    return profile


def _merge_profile(target, updates):
    for key, value in updates.items():
        if isinstance(value, bool):
            target[key] = bool(target.get(key, False) or value)
        elif isinstance(value, int):
            target[key] = int(target.get(key, 0) + value)
        else:
            target[key] = float(target.get(key, 0.0) + value)
    return target


def _get_or_create_phi_cache(state, profile=None):
    cache = state.get("_monolithic_phi_cache")
    if cache is None:
        form_cache = _get_or_create_phi_form_cache(state)
        R_phi_form = form_cache["R_phi_form"]
        K_phi_u_form = form_cache["K_phi_u_form"]
        K_phi_phi_form = form_cache["K_phi_phi_form"]
        K_phi_u_mat = fem.create_matrix(K_phi_u_form)
        K_phi_phi_mat = fem.create_matrix(K_phi_phi_form)
        R_phi_vec = fem.create_vector(R_phi_form)
        ndof_phi = state["phi"].vector.getLocalSize()
        ndof_u = state["u"].vector.getLocalSize()
        cache = {
            **form_cache,
            "R_phi_cpp_form": getattr(R_phi_form, "_cpp_object", R_phi_form),
            "K_phi_u_cpp_form": getattr(K_phi_u_form, "_cpp_object", K_phi_u_form),
            "K_phi_phi_cpp_form": getattr(K_phi_phi_form, "_cpp_object", K_phi_phi_form),
            "R_phi_vec": R_phi_vec,
            "K_phi_u_mat": K_phi_u_mat,
            "K_phi_phi_mat": K_phi_phi_mat,
            "J_phiu_dense": np.zeros((ndof_phi, ndof_u), dtype=np.float64),
            "J_phiphi_dense": np.zeros((ndof_phi, ndof_phi), dtype=np.float64),
            "J_phiu_scatter_rows": None,
            "J_phiu_scatter_cols": None,
            "J_phiphi_scatter_rows": None,
            "J_phiphi_scatter_cols": None,
        }
        state["_monolithic_phi_cache"] = cache
        if profile is not None:
            profile["phi_form_cache_miss_count"] += 3
    else:
        if profile is not None:
            profile["phi_form_cache_hit_count"] += 3
    return cache


def _assemble_vector_array(form_expr):
    compiled = _compile_form(form_expr)
    vec = fem.assemble_vector(compiled)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec.array.copy()


def _assemble_matrix_dense(form_expr):
    compiled = _compile_form(form_expr)
    mat = fem.assemble_matrix(compiled)
    mat.assemble()
    return dense_array_from_petsc_mat(mat)


def _assemble_vector_array_compiled(compiled_form):
    vec = fem.assemble_vector(compiled_form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec.array.copy()


def _assemble_matrix_dense_compiled(compiled_form):
    mat = fem.assemble_matrix(compiled_form)
    mat.assemble()
    return dense_array_from_petsc_mat(mat)


def _assemble_vector_array_compiled_reuse(vec, compiled_form):
    with vec.localForm() as vec_local:
        vec_local.set(0.0)
    fem.assemble_vector(vec, compiled_form)
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vec.array.copy()


def _dense_from_csr(
    mat,
    *,
    dense_buffer,
    scatter_rows=None,
    scatter_cols=None,
    track_scatter_build=False,
):
    extract_t0 = time.perf_counter()
    indptr, indices, values = mat.getValuesCSR()
    extract_time = time.perf_counter() - extract_t0
    convert_t0 = time.perf_counter()
    scatter_build_time = 0.0
    scatter_build_count = 0
    dense_buffer.fill(0.0)
    if scatter_rows is None or scatter_cols is None:
        scatter_build_t0 = time.perf_counter()
        row_sizes = np.diff(indptr)
        scatter_rows = np.repeat(
            np.arange(len(indptr) - 1, dtype=np.int32),
            row_sizes,
        )
        scatter_cols = np.asarray(indices, dtype=np.int32)
        if track_scatter_build:
            scatter_build_time = time.perf_counter() - scatter_build_t0
            scatter_build_count = 1
    dense_buffer[scatter_rows, scatter_cols] = values
    convert_time = time.perf_counter() - convert_t0
    return (
        dense_buffer,
        float(extract_time),
        float(convert_time),
        scatter_rows,
        scatter_cols,
        float(scatter_build_time),
        int(scatter_build_count),
    )


def _assemble_matrix_dense_compiled_reuse(
    mat,
    compiled_form,
    *,
    dense_buffer,
    scatter_rows=None,
    scatter_cols=None,
    track_scatter_build=False,
    assembly_backend="python",
):
    mat.zeroEntries()
    if assembly_backend == "python":
        fem.assemble_matrix(mat, compiled_form)
    elif assembly_backend == "cpp_petsc":
        cpp.fem.assemble_matrix_petsc(mat, getattr(compiled_form, "_cpp_object", compiled_form), [])
    else:
        raise ValueError(f"Unsupported phi_matrix_assembly_backend: {assembly_backend}")
    mat.assemble()
    return _dense_from_csr(
        mat,
        dense_buffer=dense_buffer,
        scatter_rows=scatter_rows,
        scatter_cols=scatter_cols,
        track_scatter_build=track_scatter_build,
    )


def _prime_phi_cache(state):
    if state.get("_monolithic_phi_cache_primed", False):
        return 0.0
    t0 = time.perf_counter()
    _get_or_create_phi_cache(state, profile=None)
    state["_monolithic_phi_cache_primed"] = True
    return float(time.perf_counter() - t0)


def _capture_phi_profile_detail(phi_profile, key, value):
    if key in phi_profile:
        phi_profile[key] += value


def _capture_phi_profile_count(phi_profile, key, value=1):
    if key in phi_profile:
        phi_profile[key] += int(value)


def _snapshot_state_fields(state):
    return {
        "u": state["u"].vector.array_r.copy(),
        "phi": state["phi"].vector.array_r.copy(),
        "step_offset": np.asarray(
            state.get("slave_current_offset", np.zeros(3)), dtype=np.float64
        ).copy(),
        "current_load_value": float(state.get("current_load_value", 0.0)),
    }


def _restore_state_fields(state, snapshot):
    state["u"].vector.array_w[:] = snapshot["u"]
    state["u"].vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    state["phi"].vector.array_w[:] = snapshot["phi"]
    state["phi"].vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    state["slave_current_offset"] = snapshot["step_offset"].copy()
    state["current_load_value"] = snapshot["current_load_value"]


def _apply_step_data(state, step_data):
    state["current_step_id"] = int(step_data.get("step", state.get("current_step_id", 0)))
    load_value = float(
        step_data.get("attempt_load", step_data.get("load_value", state.get("current_load_value", 0.0)))
    )
    state["current_load_value"] = load_value

    if "apply_step_data" in state and callable(state["apply_step_data"]):
        state["apply_step_data"](state, step_data)
        return

    if "slave_current_offset" in step_data:
        state["slave_current_offset"] = np.asarray(step_data["slave_current_offset"], dtype=np.float64)
        return

    offset = np.asarray(state.get("slave_current_offset", np.zeros(3)), dtype=np.float64).copy()
    offset[2] = -load_value
    state["slave_current_offset"] = offset


def _write_history_csv(history, history_path):
    if history_path is None:
        return
    fieldnames = [
        "attempt_state_index",
        "accepted_state_index",
        "accepted",
        "step",
        "load_value",
        "target_load",
        "attempt_load",
        "accepted_load_before_step",
        "load_increment",
        "converged",
        "backend",
        "build_path",
        "linear_solver_mode",
        "ksp_type",
        "pc_type",
        "block_pc_name",
        "linear_converged",
        "linear_iterations",
        "ksp_reason",
        "outer_residual_norm_before_linear",
        "outer_residual_norm_after_linear",
        "relative_linear_reduction",
        "assembly_time",
        "struct_block_assembly_time",
        "contact_block_assembly_time",
        "phi_block_assembly_time",
        "contact_quadrature_loop_time",
        "contact_geometry_eval_time",
        "contact_query_time",
        "contact_gap_normal_eval_time",
        "contact_active_filter_time",
        "contact_local_residual_time",
        "contact_local_tangent_uu_time",
        "contact_local_tangent_uphi_time",
        "contact_geometry_eval_call_count",
        "contact_geometry_eval_avg_time",
        "contact_query_avg_time",
        "contact_query_call_count",
        "contact_query_cache_hit_count",
        "contact_query_cache_miss_count",
        "contact_sensitivity_call_count",
        "contact_sensitivity_cache_hit_count",
        "contact_sensitivity_cache_miss_count",
        "contact_sensitivity_time",
        "contact_sensitivity_avg_time",
        "contact_local_tangent_uphi_call_count",
        "contact_local_tangent_uu_call_count",
        "cell_geometry_cache_hit_count",
        "cell_geometry_cache_miss_count",
        "function_cell_dof_cache_hit_count",
        "function_cell_dof_cache_miss_count",
        "vector_subfunction_cache_hit_count",
        "vector_subfunction_cache_miss_count",
        "contact_scatter_to_global_time",
        "phi_form_assembly_time",
        "phi_matrix_extract_or_convert_time",
        "phi_rhs_extract_or_convert_time",
        "dofmap_lookup_time",
        "surface_entity_iteration_time",
        "basis_eval_or_tabulation_time",
        "numpy_temp_allocation_time",
        "block_build_time",
        "bc_elimination_time",
        "global_matrix_allocation_time",
        "global_matrix_fill_time",
        "global_rhs_build_time",
        "petsc_object_setup_time",
        "ksp_setup_time",
        "ksp_solve_time",
        "linear_solve_time",
        "state_update_time",
        "newton_step_walltime",
        "load_step_walltime",
        "newton_iterations",
        "residual_norm",
        "increment_norm",
        "active_contact_points",
        "negative_gap_sum",
        "max_penetration",
        "reaction_norm",
        "mesh_resolution",
        "ndof_u",
        "ndof_phi",
        "nnz_global",
        "nnz_Juu",
        "nnz_Juphi",
        "nnz_Jphiu",
        "nnz_Jphiphi",
        "reused_global_matrix",
        "reused_global_rhs_vec",
        "reused_ksp",
        "reused_fieldsplit_is",
        "reused_subksp",
        "reuse_attempted",
        "reuse_failed_stage",
        "reuse_failure_message",
        "linear_iterations_list",
        "ksp_reason_list",
        "outer_residual_norm_before_linear_list",
        "outer_residual_norm_after_linear_list",
        "relative_linear_reduction_list",
        "assembly_time_list",
        "struct_block_assembly_time_list",
        "contact_block_assembly_time_list",
        "phi_block_assembly_time_list",
        "contact_quadrature_loop_time_list",
        "contact_geometry_eval_time_list",
        "contact_query_time_list",
        "contact_gap_normal_eval_time_list",
        "contact_active_filter_time_list",
        "contact_local_residual_time_list",
        "contact_local_tangent_uu_time_list",
        "contact_local_tangent_uphi_time_list",
        "contact_geometry_eval_call_count_list",
        "contact_geometry_eval_avg_time_list",
        "contact_query_avg_time_list",
        "contact_query_call_count_list",
        "contact_query_cache_hit_count_list",
        "contact_query_cache_miss_count_list",
        "contact_sensitivity_call_count_list",
        "contact_sensitivity_cache_hit_count_list",
        "contact_sensitivity_cache_miss_count_list",
        "contact_sensitivity_time_list",
        "contact_sensitivity_avg_time_list",
        "contact_local_tangent_uphi_call_count_list",
        "contact_local_tangent_uu_call_count_list",
        "cell_geometry_cache_hit_count_list",
        "cell_geometry_cache_miss_count_list",
        "function_cell_dof_cache_hit_count_list",
        "function_cell_dof_cache_miss_count_list",
        "vector_subfunction_cache_hit_count_list",
        "vector_subfunction_cache_miss_count_list",
        "contact_scatter_to_global_time_list",
        "phi_form_assembly_time_list",
        "phi_matrix_extract_or_convert_time_list",
        "phi_rhs_extract_or_convert_time_list",
        "dofmap_lookup_time_list",
        "surface_entity_iteration_time_list",
        "basis_eval_or_tabulation_time_list",
        "numpy_temp_allocation_time_list",
        "block_build_time_list",
        "bc_elimination_time_list",
        "global_matrix_allocation_time_list",
        "global_matrix_fill_time_list",
        "global_rhs_build_time_list",
        "petsc_object_setup_time_list",
        "ksp_setup_time_list",
        "ksp_solve_time_list",
        "linear_solve_time_list",
        "state_update_time_list",
        "newton_step_walltime_list",
        "cutback_level",
        "step_length",
        "cutback_triggered",
        "cutback_reason",
        "terminated_early",
        "termination_reason",
        "requested_final_target_load",
        "final_accepted_load",
        "reached_final_target",
        "output_index",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            row = {key: item.get(key, "") for key in fieldnames}
            if isinstance(row.get("linear_iterations_list"), list):
                row["linear_iterations_list"] = "|".join(
                    str(value) for value in row["linear_iterations_list"]
                )
            if isinstance(row.get("ksp_reason_list"), list):
                row["ksp_reason_list"] = "|".join(
                    str(value) for value in row["ksp_reason_list"]
                )
            if isinstance(row.get("outer_residual_norm_before_linear_list"), list):
                row["outer_residual_norm_before_linear_list"] = "|".join(
                    f"{value:.16e}" for value in row["outer_residual_norm_before_linear_list"]
                )
            if isinstance(row.get("outer_residual_norm_after_linear_list"), list):
                row["outer_residual_norm_after_linear_list"] = "|".join(
                    f"{value:.16e}" for value in row["outer_residual_norm_after_linear_list"]
                )
            if isinstance(row.get("relative_linear_reduction_list"), list):
                row["relative_linear_reduction_list"] = "|".join(
                    f"{value:.16e}" for value in row["relative_linear_reduction_list"]
                )
            for key in (
                "assembly_time_list",
                "struct_block_assembly_time_list",
                "contact_block_assembly_time_list",
                "phi_block_assembly_time_list",
                "contact_quadrature_loop_time_list",
                "contact_geometry_eval_time_list",
                "contact_query_time_list",
                "contact_gap_normal_eval_time_list",
                "contact_active_filter_time_list",
                "contact_local_residual_time_list",
                "contact_local_tangent_uu_time_list",
                "contact_local_tangent_uphi_time_list",
                "contact_geometry_eval_avg_time_list",
                "contact_query_avg_time_list",
                "contact_scatter_to_global_time_list",
                "phi_form_assembly_time_list",
                "phi_matrix_extract_or_convert_time_list",
                "phi_rhs_extract_or_convert_time_list",
                "dofmap_lookup_time_list",
                "surface_entity_iteration_time_list",
                "basis_eval_or_tabulation_time_list",
                "numpy_temp_allocation_time_list",
                "block_build_time_list",
                "bc_elimination_time_list",
                "global_matrix_allocation_time_list",
                "global_matrix_fill_time_list",
                "global_rhs_build_time_list",
                "petsc_object_setup_time_list",
                "ksp_setup_time_list",
                "ksp_solve_time_list",
                "linear_solve_time_list",
                "state_update_time_list",
                "newton_step_walltime_list",
            ):
                if isinstance(row.get(key), list):
                    row[key] = "|".join(f"{value:.16e}" for value in row[key])
            for key in (
                "contact_geometry_eval_call_count_list",
                "contact_query_call_count_list",
                "contact_query_cache_hit_count_list",
                "contact_query_cache_miss_count_list",
                "contact_sensitivity_call_count_list",
                "contact_sensitivity_cache_hit_count_list",
                "contact_sensitivity_cache_miss_count_list",
                "contact_sensitivity_time_list",
                "contact_sensitivity_avg_time_list",
                "contact_local_tangent_uphi_call_count_list",
                "contact_local_tangent_uu_call_count_list",
                "cell_geometry_cache_hit_count_list",
                "cell_geometry_cache_miss_count_list",
                "function_cell_dof_cache_hit_count_list",
                "function_cell_dof_cache_miss_count_list",
                "vector_subfunction_cache_hit_count_list",
                "vector_subfunction_cache_miss_count_list",
            ):
                if isinstance(row.get(key), list):
                    row[key] = "|".join(str(int(value)) for value in row[key])
            writer.writerow(row)


def _summarize_loadpath_result(
    attempt_history,
    accepted_history,
    requested_final_target_load,
    terminated_early,
    termination_reason,
):
    final_accepted_load = (
        float(accepted_history[-1]["load_value"]) if accepted_history else 0.0
    )
    reached_final_target = (
        not terminated_early
        and abs(final_accepted_load - requested_final_target_load) < 1e-12
    )
    return {
        "attempt_history": attempt_history,
        "accepted_history": accepted_history,
        "requested_final_target_load": float(requested_final_target_load),
        "final_accepted_load": final_accepted_load,
        "reached_final_target": bool(reached_final_target),
        "terminated_early": bool(terminated_early),
        "termination_reason": termination_reason,
        "attempt_count": len(attempt_history),
        "accepted_step_count": len(accepted_history),
        "accepted_nonzero_step_count": sum(
            1 for item in accepted_history if abs(float(item["load_value"])) > 1e-14
        ),
    }


def _current_state_vector(state):
    return np.concatenate([state["u"].vector.array_r.copy(), state["phi"].vector.array_r.copy()])


def _nnz_dense(matrix):
    return int(np.count_nonzero(np.asarray(matrix)))


def _apply_state_increment(state, delta_u, delta_phi, scale=1.0):
    state["u"].vector.array_w[:] = state["u"].vector.array_r + scale * delta_u
    state["u"].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    state["phi"].vector.array_w[:] = state["phi"].vector.array_r + scale * delta_phi
    state["phi"].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def _apply_block_dirichlet(J, residual, current_values, constrained_dofs):
    if constrained_dofs.size == 0:
        return
    J[constrained_dofs, :] = 0.0
    J[:, constrained_dofs] = 0.0
    J[constrained_dofs, constrained_dofs] = 1.0
    residual[constrained_dofs] = current_values[constrained_dofs]


def _create_petsc_vec_from_array(array, comm):
    vec = PETSc.Vec().createSeq(len(array), comm=comm)
    vec.setValues(np.arange(len(array), dtype=np.int32), np.asarray(array, dtype=np.float64))
    vec.assemble()
    return vec


def _create_petsc_aij_from_dense(matrix, comm):
    matrix = np.asarray(matrix, dtype=np.float64)
    nrows, ncols = matrix.shape
    mat = PETSc.Mat().createAIJ([nrows, ncols], comm=comm)
    mat.setUp()
    rows = np.arange(nrows, dtype=np.int32)
    mat.setValues(rows, rows, matrix)
    mat.assemble()
    return mat


def _fill_petsc_vec_from_array(vec, array):
    arr = vec.getArray()
    arr[:] = np.asarray(array, dtype=np.float64)
    vec.assemblyBegin()
    vec.assemblyEnd()


def _fill_petsc_aij_from_dense(mat, matrix, rows):
    mat.zeroEntries()
    mat.setValues(rows, rows, np.asarray(matrix, dtype=np.float64))
    mat.assemble()


def _build_cache_key(state, *, linear_solver_mode, ksp_type, pc_type, block_pc_name):
    dof_layout = _state_dof_layout(state)
    return (
        dof_layout["mesh_resolution"],
        dof_layout["ndof_u"],
        dof_layout["ndof_phi"],
        linear_solver_mode,
        ksp_type,
        pc_type,
        block_pc_name,
    )


def _get_or_create_build_cache(
    state,
    *,
    linear_solver_mode,
    ksp_type,
    pc_type,
    block_pc_name,
):
    cache_store = state.setdefault("_monolithic_build_cache", {})
    key = _build_cache_key(
        state,
        linear_solver_mode=linear_solver_mode,
        ksp_type=ksp_type,
        pc_type=pc_type,
        block_pc_name=block_pc_name,
    )
    cache = cache_store.get(key)
    created = False
    if cache is None:
        dof_layout = _state_dof_layout(state)
        total_dofs = dof_layout["total_dofs"]
        rows = np.arange(total_dofs, dtype=np.int32)
        comm = state["domain"].mpi_comm()
        mat = PETSc.Mat().createAIJ([total_dofs, total_dofs], nnz=total_dofs, comm=comm)
        mat.setUp()
        rhs = PETSc.Vec().createSeq(total_dofs, comm=comm)
        u_is = PETSc.IS().createGeneral(np.arange(dof_layout["ndof_u"], dtype=np.int32), comm=comm)
        phi_is = PETSc.IS().createGeneral(
            np.arange(dof_layout["ndof_u"], total_dofs, dtype=np.int32), comm=comm
        )
        cache = {
            "rows": rows,
            "global_jacobian_dense": np.zeros((total_dofs, total_dofs), dtype=np.float64),
            "global_residual_array": np.zeros(total_dofs, dtype=np.float64),
            "global_jacobian_mat": mat,
            "global_residual_vec": rhs,
            "u_is": u_is,
            "phi_is": phi_is,
            "ksp": None,
            "subksps_initialized": False,
        }
        cache_store[key] = cache
        created = True
    return cache, created


def _configure_ksp_for_fieldsplit(
    solver,
    A,
    ndof_u,
    ndof_phi,
    *,
    block_pc_name,
    u_is=None,
    phi_is=None,
):
    if block_pc_name not in BLOCK_PC_CONFIGS:
        raise ValueError(
            f"Unsupported block_pc_name: {block_pc_name}. "
            f"Expected one of {sorted(BLOCK_PC_CONFIGS)}."
        )
    block_cfg = BLOCK_PC_CONFIGS[block_pc_name]
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    split_type = block_cfg["split_type"]
    if split_type == "SCHUR":
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(
            _safe_enum_lookup(
                PETSc.PC.SchurFactType,
                block_cfg["schur_fact_type"],
                feature_name="schur_fact_type",
            )
        )
        pc.setFieldSplitSchurPreType(
            _safe_enum_lookup(
                PETSc.PC.SchurPreType,
                block_cfg["schur_pre_type"],
                feature_name="schur_pre_type",
            )
        )
    else:
        pc.setFieldSplitType(
            _safe_enum_lookup(
                PETSc.PC.CompositeType,
                split_type,
                feature_name="fieldsplit composite type",
            )
        )
    comm = A.getComm()
    if u_is is None:
        u_is = PETSc.IS().createGeneral(np.arange(ndof_u, dtype=np.int32), comm=comm)
    if phi_is is None:
        phi_is = PETSc.IS().createGeneral(
            np.arange(ndof_u, ndof_u + ndof_phi, dtype=np.int32), comm=comm
        )
    pc.setFieldSplitIS(("u", u_is), ("phi", phi_is))
    solver.setUp()
    subksps = pc.getFieldSplitSubKSP()
    if len(subksps) != 2:
        raise RuntimeError(f"Expected 2 fieldsplit sub-KSPs, got {len(subksps)}")
    for sub in subksps:
        sub.setType("preonly")
        sub.getPC().setType(block_cfg["sub_pc_type"])

    return subksps


def _solve_petsc_linear_system(
    A,
    rhs,
    *,
    ndof_u,
    ndof_phi,
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    reuse_context=None,
    reuse_ksp=False,
    reuse_fieldsplit_is=False,
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
):
    ksp_setup_t0 = time.perf_counter()
    reused_ksp = False
    reused_fieldsplit_is = False
    reused_subksp = False
    reuse_attempted = bool(reuse_context is not None and (reuse_ksp or reuse_fieldsplit_is))
    reuse_failed_stage = ""
    reuse_failure_message = ""
    x = rhs.duplicate()
    x.set(0.0)

    try:
        if reuse_context is not None and reuse_ksp and reuse_context.get("ksp") is not None:
            reuse_failed_stage = "setOperators"
            solver = reuse_context["ksp"]
            _trace_reuse_stage("before_setOperators_reuse")
            solver.setOperators(A)
            _trace_reuse_stage("after_setOperators_reuse")
            reused_ksp = True
            reused_fieldsplit_is = bool(reuse_context.get("fieldsplit_is_initialized", False))
            reused_subksp = bool(reuse_context.get("subksps_initialized", False))
        else:
            solver = PETSc.KSP().create(A.getComm())
            reuse_failed_stage = "setOperators"
            _trace_reuse_stage("before_setOperators_new")
            solver.setOperators(A)
            _trace_reuse_stage("after_setOperators_new")

        if linear_solver_mode == "lu":
            solver.setType("preonly")
            solver.getPC().setType("lu")
            resolved_ksp_type = "preonly"
            resolved_pc_type = "lu"
            resolved_block_pc_name = "global_lu"
        elif linear_solver_mode == "krylov":
            solver.setType(ksp_type)
            solver.setTolerances(rtol=float(ksp_rtol), atol=float(ksp_atol), max_it=int(ksp_max_it))
            pc = solver.getPC()
            resolved_ksp_type = ksp_type
            resolved_pc_type = pc_type
            if pc_type == "fieldsplit":
                resolved_block_pc_name = block_pc_name
                reuse_failed_stage = "configure_fieldsplit"
                _trace_reuse_stage("before_configure_fieldsplit")
                _configure_ksp_for_fieldsplit(
                    solver,
                    A,
                    ndof_u,
                    ndof_phi,
                    block_pc_name=block_pc_name,
                    u_is=(
                        None
                        if reuse_context is None or not reuse_fieldsplit_is
                        else reuse_context.get("u_is")
                    ),
                    phi_is=(
                        None
                        if reuse_context is None or not reuse_fieldsplit_is
                        else reuse_context.get("phi_is")
                    ),
                )
                _trace_reuse_stage("after_configure_fieldsplit")
                if reuse_context is not None and reuse_fieldsplit_is:
                    reuse_context["fieldsplit_is_initialized"] = True
                    reused_fieldsplit_is = True
                if reuse_context is not None and reuse_ksp:
                    reuse_context["subksps_initialized"] = True
                    reused_subksp = True
            else:
                pc.setType(pc_type)
                resolved_block_pc_name = f"global_{pc_type}"
        else:
            raise ValueError(f"Unsupported linear_solver_mode: {linear_solver_mode}")
    except Exception as exc:
        reuse_failure_message = str(exc)
        raise
    ksp_setup_time = time.perf_counter() - ksp_setup_t0

    if reuse_context is not None and reuse_ksp:
        reuse_context["ksp"] = solver

    ksp_solve_t0 = time.perf_counter()
    reuse_failed_stage = "solve"
    _trace_reuse_stage("before_solve")
    solver.solve(rhs, x)
    _trace_reuse_stage("after_solve")
    ksp_solve_time = time.perf_counter() - ksp_solve_t0
    info = {
        "linear_solver_mode": linear_solver_mode,
        "ksp_type": resolved_ksp_type,
        "pc_type": resolved_pc_type,
        "block_pc_name": resolved_block_pc_name,
        "linear_iterations": int(solver.getIterationNumber()),
        "ksp_reason": int(solver.getConvergedReason()),
        "linear_converged": int(solver.getConvergedReason()) > 0,
        "ksp_setup_time": float(ksp_setup_time),
        "ksp_solve_time": float(ksp_solve_time),
        "reused_ksp": bool(reused_ksp),
        "reused_fieldsplit_is": bool(reused_fieldsplit_is),
        "reused_subksp": bool(reused_subksp),
        "reuse_attempted": bool(reuse_attempted),
        "reuse_failed_stage": reuse_failed_stage if reuse_failure_message else "",
        "reuse_failure_message": reuse_failure_message,
    }
    return x.getArray().copy(), solver, info


def _dense_block_system_from_blocks(state, assembled, *, dense_matrix=None, residual_array=None):
    ndof_u = state["u"].vector.getLocalSize()
    ndof_phi = state["phi"].vector.getLocalSize()
    if residual_array is None:
        residual = np.concatenate([assembled["R_u_total"], assembled["R_phi_total"]])
    else:
        residual = residual_array
        residual[:ndof_u] = assembled["R_u_total"]
        residual[ndof_u:] = assembled["R_phi_total"]
    if dense_matrix is None:
        J = np.zeros((ndof_u + ndof_phi, ndof_u + ndof_phi), dtype=np.float64)
    else:
        J = dense_matrix
        J.fill(0.0)
    J[:ndof_u, :ndof_u] = assembled["J_uu"]
    J[:ndof_u, ndof_u:] = assembled["J_uphi"]
    J[ndof_u:, :ndof_u] = assembled["J_phiu"]
    J[ndof_u:, ndof_u:] = assembled["J_phiphi"]

    current_values = _current_state_vector(state)
    bc_t0 = time.perf_counter()
    _apply_block_dirichlet(J, residual, current_values, assembled["u_constrained_dofs"])
    _apply_block_dirichlet(
        J, residual, current_values, ndof_u + assembled["phi_constrained_dofs"]
    )
    bc_elimination_time = time.perf_counter() - bc_t0
    return J, residual, float(bc_elimination_time)


def _assemble_global_backend_objects(
    state,
    assembled,
    backend,
    *,
    build_path="current",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    reuse_matrix_pattern=False,
    reuse_fieldsplit_is=False,
):
    global_matrix_allocation_time = 0.0
    global_matrix_fill_time = 0.0
    global_rhs_build_time = 0.0
    petsc_object_setup_time = 0.0
    reused_global_matrix = False
    reused_global_rhs_vec = False
    reused_fieldsplit_is = False

    if build_path == "optimized" and backend == "petsc_block" and (
        reuse_matrix_pattern or reuse_fieldsplit_is
    ):
        setup_t0 = time.perf_counter()
        cache, created = _get_or_create_build_cache(
            state,
            linear_solver_mode=linear_solver_mode,
            ksp_type=ksp_type,
            pc_type=pc_type,
            block_pc_name=block_pc_name,
        )
        petsc_object_setup_time = time.perf_counter() - setup_t0
        reused_global_matrix = bool(reuse_matrix_pattern and not created)
        reused_global_rhs_vec = bool(reuse_matrix_pattern and not created)
        reused_fieldsplit_is = bool(cache.get("fieldsplit_is_initialized", False))

        rhs_t0 = time.perf_counter()
        global_jacobian_dense, global_residual_array, bc_elimination_time = _dense_block_system_from_blocks(
            state,
            assembled,
            dense_matrix=cache["global_jacobian_dense"] if reuse_matrix_pattern else None,
            residual_array=cache["global_residual_array"] if reuse_matrix_pattern else None,
        )
        global_rhs_build_time = time.perf_counter() - rhs_t0
        out = {
            "global_jacobian_dense": global_jacobian_dense,
            "global_residual_array": global_residual_array,
            "global_jacobian_mat": None,
            "global_residual_vec": None,
            "bc_elimination_time": float(bc_elimination_time),
            "global_matrix_allocation_time": float(global_matrix_allocation_time),
            "global_matrix_fill_time": float(global_matrix_fill_time),
            "global_rhs_build_time": float(global_rhs_build_time),
            "petsc_object_setup_time": float(petsc_object_setup_time),
            "reused_global_matrix": bool(reused_global_matrix),
            "reused_global_rhs_vec": bool(reused_global_rhs_vec),
            "reused_fieldsplit_is": bool(reused_fieldsplit_is),
            "reuse_context": cache,
        }
        if reuse_matrix_pattern:
            mat_fill_t0 = time.perf_counter()
            _fill_petsc_aij_from_dense(cache["global_jacobian_mat"], global_jacobian_dense, cache["rows"])
            global_matrix_fill_time = time.perf_counter() - mat_fill_t0
            rhs_fill_t0 = time.perf_counter()
            _fill_petsc_vec_from_array(cache["global_residual_vec"], global_residual_array)
            global_rhs_build_time += time.perf_counter() - rhs_fill_t0
            out["global_jacobian_mat"] = cache["global_jacobian_mat"]
            out["global_residual_vec"] = cache["global_residual_vec"]
        else:
            alloc_t0 = time.perf_counter()
            out["global_jacobian_mat"] = _create_petsc_aij_from_dense(global_jacobian_dense, state["domain"].mpi_comm())
            out["global_residual_vec"] = _create_petsc_vec_from_array(global_residual_array, state["domain"].mpi_comm())
            alloc_elapsed = time.perf_counter() - alloc_t0
            global_matrix_allocation_time = float(alloc_elapsed)
            global_matrix_fill_time = float(alloc_elapsed)
        out["global_matrix_fill_time"] = float(global_matrix_fill_time)
        out["global_rhs_build_time"] = float(global_rhs_build_time)
        return out

    dense_t0 = time.perf_counter()
    global_jacobian_dense, global_residual_array, bc_elimination_time = _dense_block_system_from_blocks(
        state, assembled
    )
    global_rhs_build_time = time.perf_counter() - dense_t0
    out = {
        "global_jacobian_dense": global_jacobian_dense,
        "global_residual_array": global_residual_array,
        "global_jacobian_mat": None,
        "global_residual_vec": None,
        "bc_elimination_time": float(bc_elimination_time),
        "global_matrix_allocation_time": float(global_matrix_allocation_time),
        "global_matrix_fill_time": float(global_matrix_fill_time),
        "global_rhs_build_time": float(global_rhs_build_time),
        "petsc_object_setup_time": float(petsc_object_setup_time),
        "reused_global_matrix": False,
        "reused_global_rhs_vec": False,
        "reused_fieldsplit_is": False,
        "reuse_context": None,
    }
    if backend == "petsc_block":
        comm = state["domain"].mpi_comm()
        alloc_t0 = time.perf_counter()
        out["global_jacobian_mat"] = _create_petsc_aij_from_dense(global_jacobian_dense, comm)
        out["global_residual_vec"] = _create_petsc_vec_from_array(global_residual_array, comm)
        alloc_elapsed = time.perf_counter() - alloc_t0
        out["global_matrix_allocation_time"] = float(alloc_elapsed)
        out["global_matrix_fill_time"] = float(alloc_elapsed)
        out["global_rhs_build_time"] = float(global_rhs_build_time)
    elif backend != "dense":
        raise ValueError(f"Unsupported monolithic backend: {backend}")
    return out


def assemble_monolithic_contact_system(
    state,
    cfg,
    *,
    backend="dense",
    need_jacobian=True,
    build_path="current",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    reuse_matrix_pattern=False,
    reuse_fieldsplit_is=False,
    profile_assembly_detail=False,
    phi_scatter_reuse=True,
    profile_phi_detail=True,
    phi_matrix_assembly_backend="python",
):
    """Assemble the monolithic block residual and Jacobian."""
    dof_layout = _state_dof_layout(state)
    struct_t0 = time.perf_counter()
    A_struct, b_struct, solid_meta = assemble_linear_solid_system(
        state["u"],
        state["R_u_form"],
        state["solid_bcs"],
        contact_rhs=None,
        contact_tangent_uu=None,
        contact_structure_mode="rhs_only",
        reference_u=state["u"].vector.array_r.copy(),
    )
    J_uu_struct = dense_array_from_petsc_mat(A_struct)
    u_vec = state["u"].vector.array_r.copy()
    R_u_struct = J_uu_struct.dot(u_vec) - b_struct.array.copy()
    struct_block_assembly_time = time.perf_counter() - struct_t0

    contact_t0 = time.perf_counter()
    contact = assemble_contact_contributions_surface(
        state["quadrature_points"],
        state,
        state["penalty"],
        need_residual=True,
        need_tangent_uu=need_jacobian,
        need_tangent_uphi=need_jacobian,
        need_diagnostics=True,
        build_path=build_path,
        profile_assembly_detail=profile_assembly_detail,
    )
    contact_block_assembly_time = time.perf_counter() - contact_t0

    R_u_total = R_u_struct - contact["R_u_c"]
    phi_profile = _phi_profile_template(profile_phi_detail=profile_phi_detail)
    phi_cache = None
    phi_form_cache_hits = 0
    phi_form_cache_misses = 0
    if build_path == "optimized":
        phi_cache = _get_or_create_phi_cache(state, profile=None)
        if state.get("_monolithic_phi_cache") is phi_cache:
            phi_form_cache_hits = 3 if state.get("_monolithic_phi_cache_primed", False) else 0
            phi_form_cache_misses = 0 if phi_form_cache_hits else 3
    phi_t0 = time.perf_counter()
    compiled_R_phi = state["R_phi_form"] if phi_cache is None else phi_cache["R_phi_form"]
    rhs_assemble_t0 = time.perf_counter()
    if phi_cache is None:
        compiled_R_phi = _compile_form(compiled_R_phi)
        phi_vec = fem.assemble_vector(compiled_R_phi)
        rhs_assembly_time = time.perf_counter() - rhs_assemble_t0
        rhs_extract_t0 = time.perf_counter()
        phi_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        R_phi_total = phi_vec.array.copy()
        rhs_extract_time = time.perf_counter() - rhs_extract_t0
        phi_form_cache_misses += 1
    else:
        R_phi_total = _assemble_vector_array_compiled_reuse(phi_cache["R_phi_vec"], compiled_R_phi)
        rhs_assembly_time = time.perf_counter() - rhs_assemble_t0
        rhs_extract_t0 = time.perf_counter()
        R_phi_total = np.asarray(R_phi_total, dtype=np.float64)
        rhs_extract_time = time.perf_counter() - rhs_extract_t0
    phi_block_assembly_time = time.perf_counter() - phi_t0

    out = {
        "R_u_total": R_u_total,
        "R_phi_total": R_phi_total,
        "u_constrained_dofs": _owned_bc_dofs(state["solid_bcs"]),
        "phi_constrained_dofs": _owned_bc_dofs(state["phi_bcs"]),
    }

    if need_jacobian:
        block_build_t0 = time.perf_counter()
        out["J_uu"] = J_uu_struct - contact["K_uu_c"]
        out["J_uphi"] = -contact["K_uphi_c"]
        compiled_K_phi_u = state["K_phi_u_form"] if phi_cache is None else phi_cache["K_phi_u_form"]
        compiled_K_phi_phi = state["K_phi_phi_form"] if phi_cache is None else phi_cache["K_phi_phi_form"]
        phiu_scatter_build_time = 0.0
        phiu_scatter_build_count = 0
        phiphi_scatter_build_time = 0.0
        phiphi_scatter_build_count = 0
        if phi_cache is None:
            compiled_K_phi_u = _compile_form(compiled_K_phi_u)
            compiled_K_phi_phi = _compile_form(compiled_K_phi_phi)

            phiu_form_t0 = time.perf_counter()
            phiu_mat = fem.assemble_matrix(compiled_K_phi_u)
            phiu_mat.assemble()
            phiu_form_time = time.perf_counter() - phiu_form_t0
            phiu_extract_t0 = time.perf_counter()
            out["J_phiu"] = dense_array_from_petsc_mat(phiu_mat)
            phiu_extract_time = time.perf_counter() - phiu_extract_t0
            phiu_convert_time = 0.0

            phiphi_form_t0 = time.perf_counter()
            phiphi_mat = fem.assemble_matrix(compiled_K_phi_phi)
            phiphi_mat.assemble()
            phiphi_form_time = time.perf_counter() - phiphi_form_t0
            phiphi_extract_t0 = time.perf_counter()
            out["J_phiphi"] = dense_array_from_petsc_mat(phiphi_mat)
            phiphi_extract_time = time.perf_counter() - phiphi_extract_t0
            phiphi_convert_time = 0.0
            phi_form_cache_misses += 2
        else:
            phiu_form_t0 = time.perf_counter()
            (
                out["J_phiu"],
                phiu_extract_time,
                phiu_convert_time,
                phi_cache["J_phiu_scatter_rows"],
                phi_cache["J_phiu_scatter_cols"],
                phiu_scatter_build_time,
                phiu_scatter_build_count,
            ) = _assemble_matrix_dense_compiled_reuse(
                phi_cache["K_phi_u_mat"],
                compiled_K_phi_u,
                dense_buffer=phi_cache["J_phiu_dense"],
                scatter_rows=phi_cache.get("J_phiu_scatter_rows") if phi_scatter_reuse else None,
                scatter_cols=phi_cache.get("J_phiu_scatter_cols") if phi_scatter_reuse else None,
                track_scatter_build=profile_phi_detail,
                assembly_backend=phi_matrix_assembly_backend,
            )
            phiu_form_time = time.perf_counter() - phiu_form_t0 - phiu_extract_time - phiu_convert_time

            phiphi_form_t0 = time.perf_counter()
            (
                out["J_phiphi"],
                phiphi_extract_time,
                phiphi_convert_time,
                phi_cache["J_phiphi_scatter_rows"],
                phi_cache["J_phiphi_scatter_cols"],
                phiphi_scatter_build_time,
                phiphi_scatter_build_count,
            ) = _assemble_matrix_dense_compiled_reuse(
                phi_cache["K_phi_phi_mat"],
                compiled_K_phi_phi,
                dense_buffer=phi_cache["J_phiphi_dense"],
                scatter_rows=phi_cache.get("J_phiphi_scatter_rows") if phi_scatter_reuse else None,
                scatter_cols=phi_cache.get("J_phiphi_scatter_cols") if phi_scatter_reuse else None,
                track_scatter_build=profile_phi_detail,
                assembly_backend=phi_matrix_assembly_backend,
            )
            phiphi_form_time = time.perf_counter() - phiphi_form_t0 - phiphi_extract_time - phiphi_convert_time
            if not phi_scatter_reuse:
                phi_cache["J_phiu_scatter_rows"] = None
                phi_cache["J_phiu_scatter_cols"] = None
                phi_cache["J_phiphi_scatter_rows"] = None
                phi_cache["J_phiphi_scatter_cols"] = None

        out["phi_block_assembly_time"] = float(time.perf_counter() - phi_t0)
        out["nnz_Juu"] = _nnz_dense(out["J_uu"])
        out["nnz_Juphi"] = _nnz_dense(out["J_uphi"])
        out["nnz_Jphiu"] = _nnz_dense(out["J_phiu"])
        out["nnz_Jphiphi"] = _nnz_dense(out["J_phiphi"])
        out.update(
            _assemble_global_backend_objects(
                state,
                out,
                backend,
                build_path=build_path,
                linear_solver_mode=linear_solver_mode,
                ksp_type=ksp_type,
                pc_type=pc_type,
                block_pc_name=block_pc_name,
                reuse_matrix_pattern=reuse_matrix_pattern,
                reuse_fieldsplit_is=reuse_fieldsplit_is,
            )
        )
        out["nnz_global"] = _nnz_dense(out["global_jacobian_dense"])
        block_build_time = time.perf_counter() - block_build_t0
    else:
        out["J_uu"] = None
        out["J_uphi"] = None
        out["J_phiu"] = None
        out["J_phiphi"] = None
        out["global_jacobian_dense"] = None
        out["global_jacobian_mat"] = None
        out["global_residual_array"] = np.concatenate([R_u_total, R_phi_total])
        out["global_residual_vec"] = (
            _create_petsc_vec_from_array(out["global_residual_array"], state["domain"].mpi_comm())
            if backend == "petsc_block"
            else None
        )
        out["nnz_Juu"] = 0
        out["nnz_Juphi"] = 0
        out["nnz_Jphiu"] = 0
        out["nnz_Jphiphi"] = 0
        out["nnz_global"] = 0
        out["bc_elimination_time"] = 0.0
        out["global_matrix_allocation_time"] = 0.0
        out["global_matrix_fill_time"] = 0.0
        out["global_rhs_build_time"] = 0.0
        out["petsc_object_setup_time"] = 0.0
        out["reused_global_matrix"] = False
        out["reused_global_rhs_vec"] = False
        out["reused_fieldsplit_is"] = False
        out["reuse_context"] = None
        block_build_time = 0.0

    out.update(
        {
            "backend": backend,
            "build_path": build_path,
            "R_u_total": R_u_total,
            "R_phi_total": R_phi_total,
            "R_u_struct": R_u_struct,
            "R_u_contact": contact["R_u_c"],
            "J_uu_struct": J_uu_struct,
            "solid_meta": solid_meta,
            "diagnostics": contact["diagnostics"],
            "point_data": contact["point_data"],
            "contact_profiling": contact.get("profiling", {}),
            "u_constrained_dofs": _owned_bc_dofs(state["solid_bcs"]),
            "phi_constrained_dofs": _owned_bc_dofs(state["phi_bcs"]),
            "struct_block_assembly_time": float(struct_block_assembly_time),
            "contact_block_assembly_time": float(contact_block_assembly_time),
            "phi_block_assembly_time": float(out.get("phi_block_assembly_time", phi_block_assembly_time)),
            **dof_layout,
        }
    )
    for key, value in contact.get("profiling", {}).items():
        out[key] = value

    phi_profile["phi_form_cache_hit_count"] += int(phi_form_cache_hits)
    phi_profile["phi_form_cache_miss_count"] += int(phi_form_cache_misses)
    phi_profile["reused_phi_rhs_vec"] = bool(phi_cache is not None)
    phi_profile["phi_rhs_assembly_time"] += float(rhs_assembly_time)
    phi_profile["phi_rhs_extract_time"] += float(rhs_extract_time)
    phi_profile["phi_form_call_count"] += 1
    phi_profile["phi_rhs_extract_call_count"] += 1
    phi_profile["phi_residual_call_count"] += 1
    phi_profile["phi_residual_form_assembly_time"] += float(rhs_assembly_time)
    phi_profile["phi_residual_extract_time"] += float(rhs_extract_time)
    phi_profile["phi_form_assembly_time"] += float(rhs_assembly_time)
    phi_profile["phi_rhs_extract_or_convert_time"] += float(rhs_extract_time)
    if profile_phi_detail:
        _capture_phi_profile_detail(phi_profile, "phi_form_time_R_phi", float(rhs_assembly_time))
        _capture_phi_profile_count(phi_profile, "phi_form_call_count_R_phi", 1)
    if need_jacobian:
        phi_profile["reused_phi_kphiu_mat"] = bool(phi_cache is not None)
        phi_profile["reused_phi_kphiphi_mat"] = bool(phi_cache is not None)
        phi_profile["reused_phi_dense_buffers"] = bool(phi_cache is not None)
        phi_profile["phi_form_call_count"] += 2
        phi_profile["phi_matrix_extract_call_count"] += 2
        phi_profile["phi_matrix_convert_call_count"] += 2
        phi_profile["phi_kphiu_call_count"] += 1
        phi_profile["phi_kphiphi_call_count"] += 1
        phi_profile["phi_kphiu_form_assembly_time"] += float(phiu_form_time)
        phi_profile["phi_kphiu_extract_time"] += float(phiu_extract_time)
        phi_profile["phi_kphiu_convert_time"] += float(phiu_convert_time)
        phi_profile["phi_kphiphi_form_assembly_time"] += float(phiphi_form_time)
        phi_profile["phi_kphiphi_extract_time"] += float(phiphi_extract_time)
        phi_profile["phi_kphiphi_convert_time"] += float(phiphi_convert_time)
        if profile_phi_detail:
            _capture_phi_profile_detail(phi_profile, "phi_form_time_K_phi_u", float(phiu_form_time))
            _capture_phi_profile_detail(phi_profile, "phi_form_time_K_phi_phi", float(phiphi_form_time))
            _capture_phi_profile_detail(phi_profile, "phi_extract_time_K_phi_u", float(phiu_extract_time))
            _capture_phi_profile_detail(phi_profile, "phi_convert_time_K_phi_u", float(phiu_convert_time))
            _capture_phi_profile_detail(phi_profile, "phi_extract_time_K_phi_phi", float(phiphi_extract_time))
            _capture_phi_profile_detail(phi_profile, "phi_convert_time_K_phi_phi", float(phiphi_convert_time))
            _capture_phi_profile_count(phi_profile, "phi_form_call_count_K_phi_u", 1)
            _capture_phi_profile_count(phi_profile, "phi_form_call_count_K_phi_phi", 1)
            _capture_phi_profile_count(phi_profile, "phi_matrix_extract_call_count_K_phi_u", 1)
            _capture_phi_profile_count(phi_profile, "phi_matrix_convert_call_count_K_phi_u", 1)
            _capture_phi_profile_count(phi_profile, "phi_matrix_extract_call_count_K_phi_phi", 1)
            _capture_phi_profile_count(phi_profile, "phi_matrix_convert_call_count_K_phi_phi", 1)
            _capture_phi_profile_detail(
                phi_profile,
                "phi_scatter_pattern_build_time",
                float(phiu_scatter_build_time + phiphi_scatter_build_time),
            )
            _capture_phi_profile_count(
                phi_profile,
                "phi_scatter_pattern_build_count",
                int(phiu_scatter_build_count + phiphi_scatter_build_count),
            )
        phi_profile["phi_form_assembly_time"] += float(phiu_form_time + phiphi_form_time)
        phi_profile["phi_matrix_extract_time"] += float(phiu_extract_time + phiphi_extract_time)
        phi_profile["phi_matrix_convert_time"] += float(phiu_convert_time + phiphi_convert_time)
        phi_profile["phi_matrix_extract_or_convert_time"] += float(
            phiu_extract_time + phiphi_extract_time + phiu_convert_time + phiphi_convert_time
        )

    _merge_profile(out, phi_profile)
    assembly_time = (
        struct_block_assembly_time
        + contact_block_assembly_time
        + float(out.get("phi_block_assembly_time", phi_block_assembly_time))
    )
    out["assembly_time"] = float(assembly_time)
    out["block_build_time"] = float(block_build_time)
    return out


def _evaluate_total_residual_norm(state, cfg, *, backend="dense"):
    assembled = assemble_monolithic_contact_system(state, cfg, backend=backend, need_jacobian=False)
    residual = assembled["global_residual_array"]
    return float(np.linalg.norm(residual)), assembled


def _backtracking_line_search(
    state,
    cfg,
    delta_u,
    delta_phi,
    residual_norm_before,
    *,
    backend="dense",
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
):
    snapshot = _snapshot_state_fields(state)
    alpha = float(initial_damping)
    best = None

    for _ in range(max_backtracks + 1):
        _restore_state_fields(state, snapshot)
        _apply_state_increment(state, delta_u, delta_phi, scale=alpha)
        residual_after, assembled_after = _evaluate_total_residual_norm(state, cfg, backend=backend)
        candidate = (alpha, residual_after, assembled_after)
        if best is None or residual_after < best[1]:
            best = candidate
        if residual_after < residual_norm_before:
            return {
                "accepted": True,
                "step_length": alpha,
                "residual_after": residual_after,
                "assembled_after": assembled_after,
                "used_fallback": False,
            }
        alpha *= float(backtrack_factor)

    _restore_state_fields(state, snapshot)
    if best is None:
        return {
            "accepted": False,
            "step_length": 0.0,
            "residual_after": residual_norm_before,
            "assembled_after": None,
            "used_fallback": False,
        }

    alpha_best, residual_best, assembled_best = best
    _apply_state_increment(state, delta_u, delta_phi, scale=alpha_best)
    return {
        "accepted": True,
        "step_length": alpha_best,
        "residual_after": residual_best,
        "assembled_after": assembled_best,
        "used_fallback": True,
    }


def solve_monolithic_contact(
    state,
    cfg,
    *,
    backend="dense",
    build_path="current",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    reuse_ksp=False,
    reuse_matrix_pattern=False,
    reuse_fieldsplit_is=False,
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
    max_newton_iter=15,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
    profile_assembly_detail=False,
    phi_scatter_reuse=True,
    profile_phi_detail=True,
    phi_matrix_assembly_backend="python",
    verbose=True,
):
    """Solve one fixed-load monolithic contact state."""
    rank0 = state["domain"].mpi_comm().rank == 0
    dof_layout = _state_dof_layout(state)
    history = []
    failure_reason = ""

    for newton_it in range(1, max_newton_iter + 1):
        newton_step_t0 = time.perf_counter()
        assembled = assemble_monolithic_contact_system(
            state,
            cfg,
            backend=backend,
            need_jacobian=True,
            build_path=build_path,
            linear_solver_mode=linear_solver_mode,
            ksp_type=ksp_type,
            pc_type=pc_type,
            block_pc_name=block_pc_name,
            reuse_matrix_pattern=reuse_matrix_pattern,
            reuse_fieldsplit_is=reuse_fieldsplit_is,
            profile_assembly_detail=profile_assembly_detail,
            phi_scatter_reuse=phi_scatter_reuse,
            profile_phi_detail=profile_phi_detail,
            phi_matrix_assembly_backend=phi_matrix_assembly_backend,
        )
        residual = assembled["global_residual_array"]
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tol_res:
            total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
            info = {
                "newton_iterations": newton_it - 1,
                "converged": True,
                "failure_reason": "",
                "backend": backend,
                "linear_solver_mode": linear_solver_mode,
                "ksp_type": "preonly" if linear_solver_mode == "lu" else ksp_type,
                "pc_type": "lu" if linear_solver_mode == "lu" else pc_type,
                "block_pc_name": "global_lu" if linear_solver_mode == "lu" else block_pc_name,
                "linear_converged": True,
                "linear_iterations": 0,
                "total_linear_iterations": total_linear_iterations,
                "ksp_reason": 0,
                "residual_norm": residual_norm,
                "increment_norm": 0.0,
                "active_contact_points": assembled["diagnostics"]["active_contact_points"],
                "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
                "reaction_norm": assembled["diagnostics"]["reaction_norm"],
                "max_penetration": assembled["diagnostics"]["max_penetration"],
                "step_length": 0.0,
                "outer_residual_norm_before_linear": 0.0,
                "outer_residual_norm_after_linear": residual_norm,
                "relative_linear_reduction": 0.0,
                "struct_block_assembly_time": float(assembled.get("struct_block_assembly_time", 0.0)),
                "contact_block_assembly_time": float(assembled.get("contact_block_assembly_time", 0.0)),
                "phi_block_assembly_time": float(assembled.get("phi_block_assembly_time", 0.0)),
                "contact_quadrature_loop_time": float(assembled.get("contact_quadrature_loop_time", 0.0)),
                "contact_geometry_eval_time": float(assembled.get("contact_geometry_eval_time", 0.0)),
                "contact_query_time": float(assembled.get("contact_query_time", 0.0)),
                "contact_gap_normal_eval_time": float(assembled.get("contact_gap_normal_eval_time", 0.0)),
                "contact_active_filter_time": float(assembled.get("contact_active_filter_time", 0.0)),
                "contact_local_residual_time": float(assembled.get("contact_local_residual_time", 0.0)),
                "contact_local_tangent_uu_time": float(assembled.get("contact_local_tangent_uu_time", 0.0)),
                "contact_local_tangent_uphi_time": float(assembled.get("contact_local_tangent_uphi_time", 0.0)),
                "contact_geometry_eval_call_count": int(assembled.get("contact_geometry_eval_call_count", 0)),
                "contact_geometry_eval_avg_time": float(assembled.get("contact_geometry_eval_avg_time", 0.0)),
                "contact_query_avg_time": float(assembled.get("contact_query_avg_time", 0.0)),
                "contact_query_call_count": int(assembled.get("contact_query_call_count", 0)),
                "contact_query_cache_hit_count": int(assembled.get("contact_query_cache_hit_count", 0)),
                "contact_query_cache_miss_count": int(assembled.get("contact_query_cache_miss_count", 0)),
                "contact_sensitivity_call_count": int(assembled.get("contact_sensitivity_call_count", 0)),
                "contact_sensitivity_cache_hit_count": int(assembled.get("contact_sensitivity_cache_hit_count", 0)),
                "contact_sensitivity_cache_miss_count": int(assembled.get("contact_sensitivity_cache_miss_count", 0)),
                "contact_sensitivity_time": float(assembled.get("contact_sensitivity_time", 0.0)),
                "contact_sensitivity_avg_time": float(assembled.get("contact_sensitivity_avg_time", 0.0)),
                "contact_local_tangent_uphi_call_count": int(assembled.get("contact_local_tangent_uphi_call_count", 0)),
                "contact_local_tangent_uu_call_count": int(assembled.get("contact_local_tangent_uu_call_count", 0)),
                "cell_geometry_cache_hit_count": int(assembled.get("cell_geometry_cache_hit_count", 0)),
                "cell_geometry_cache_miss_count": int(assembled.get("cell_geometry_cache_miss_count", 0)),
                "function_cell_dof_cache_hit_count": int(assembled.get("function_cell_dof_cache_hit_count", 0)),
                "function_cell_dof_cache_miss_count": int(assembled.get("function_cell_dof_cache_miss_count", 0)),
                "vector_subfunction_cache_hit_count": int(assembled.get("vector_subfunction_cache_hit_count", 0)),
                "vector_subfunction_cache_miss_count": int(assembled.get("vector_subfunction_cache_miss_count", 0)),
                "contact_scatter_to_global_time": float(assembled.get("contact_scatter_to_global_time", 0.0)),
                "phi_form_assembly_time": float(assembled.get("phi_form_assembly_time", 0.0)),
                "phi_matrix_extract_time": float(assembled.get("phi_matrix_extract_time", 0.0)),
                "phi_matrix_convert_time": float(assembled.get("phi_matrix_convert_time", 0.0)),
                "phi_rhs_assembly_time": float(assembled.get("phi_rhs_assembly_time", 0.0)),
                "phi_rhs_extract_time": float(assembled.get("phi_rhs_extract_time", 0.0)),
                "phi_matrix_extract_or_convert_time": float(assembled.get("phi_matrix_extract_or_convert_time", 0.0)),
                "phi_rhs_extract_or_convert_time": float(assembled.get("phi_rhs_extract_or_convert_time", 0.0)),
                "phi_dof_lookup_time": float(assembled.get("phi_dof_lookup_time", 0.0)),
                "phi_geometry_helper_time": float(assembled.get("phi_geometry_helper_time", 0.0)),
                "phi_basis_tabulation_time": float(assembled.get("phi_basis_tabulation_time", 0.0)),
                "phi_form_call_count": int(assembled.get("phi_form_call_count", 0)),
                "phi_form_cache_hit_count": int(assembled.get("phi_form_cache_hit_count", 0)),
                "phi_form_cache_miss_count": int(assembled.get("phi_form_cache_miss_count", 0)),
                "phi_matrix_extract_call_count": int(assembled.get("phi_matrix_extract_call_count", 0)),
                "phi_matrix_convert_call_count": int(assembled.get("phi_matrix_convert_call_count", 0)),
                "phi_rhs_extract_call_count": int(assembled.get("phi_rhs_extract_call_count", 0)),
                "phi_dof_lookup_call_count": int(assembled.get("phi_dof_lookup_call_count", 0)),
                "phi_basis_tabulation_call_count": int(assembled.get("phi_basis_tabulation_call_count", 0)),
                "phi_residual_form_assembly_time": float(assembled.get("phi_residual_form_assembly_time", 0.0)),
                "phi_residual_extract_time": float(assembled.get("phi_residual_extract_time", 0.0)),
                "phi_kphiu_form_assembly_time": float(assembled.get("phi_kphiu_form_assembly_time", 0.0)),
                "phi_kphiu_extract_time": float(assembled.get("phi_kphiu_extract_time", 0.0)),
                "phi_kphiu_convert_time": float(assembled.get("phi_kphiu_convert_time", 0.0)),
                "phi_kphiphi_form_assembly_time": float(assembled.get("phi_kphiphi_form_assembly_time", 0.0)),
                "phi_kphiphi_extract_time": float(assembled.get("phi_kphiphi_extract_time", 0.0)),
                "phi_kphiphi_convert_time": float(assembled.get("phi_kphiphi_convert_time", 0.0)),
                "dofmap_lookup_time": float(assembled.get("dofmap_lookup_time", 0.0)),
                "surface_entity_iteration_time": float(assembled.get("surface_entity_iteration_time", 0.0)),
                "basis_eval_or_tabulation_time": float(assembled.get("basis_eval_or_tabulation_time", 0.0)),
                "numpy_temp_allocation_time": float(assembled.get("numpy_temp_allocation_time", 0.0)),
                "assembly_time": float(assembled.get("assembly_time", 0.0)),
                "block_build_time": float(assembled.get("block_build_time", 0.0)),
                "bc_elimination_time": float(assembled.get("bc_elimination_time", 0.0)),
                "global_matrix_allocation_time": float(assembled.get("global_matrix_allocation_time", 0.0)),
                "global_matrix_fill_time": float(assembled.get("global_matrix_fill_time", 0.0)),
                "global_rhs_build_time": float(assembled.get("global_rhs_build_time", 0.0)),
                "petsc_object_setup_time": float(assembled.get("petsc_object_setup_time", 0.0)),
                "ksp_setup_time": 0.0,
                "ksp_solve_time": 0.0,
                "linear_solve_time": 0.0,
                "state_update_time": 0.0,
                "newton_step_walltime": 0.0,
                "reused_global_matrix": bool(assembled.get("reused_global_matrix", False)),
                "reused_global_rhs_vec": bool(assembled.get("reused_global_rhs_vec", False)),
                "reused_ksp": False,
                "reused_fieldsplit_is": bool(assembled.get("reused_fieldsplit_is", False)),
                "reused_subksp": False,
                "reuse_attempted": False,
                "reuse_failed_stage": "",
                "reuse_failure_message": "",
                "nnz_global": int(assembled.get("nnz_global", 0)),
                "nnz_Juu": int(assembled.get("nnz_Juu", 0)),
                "nnz_Juphi": int(assembled.get("nnz_Juphi", 0)),
                "nnz_Jphiu": int(assembled.get("nnz_Jphiu", 0)),
                "nnz_Jphiphi": int(assembled.get("nnz_Jphiphi", 0)),
                **dof_layout,
                "history": history,
            }
            if profile_phi_detail:
                info.update(
                    {
                        "phi_form_time_R_phi": float(assembled.get("phi_form_time_R_phi", 0.0)),
                        "phi_form_time_K_phi_u": float(assembled.get("phi_form_time_K_phi_u", 0.0)),
                        "phi_form_time_K_phi_phi": float(assembled.get("phi_form_time_K_phi_phi", 0.0)),
                        "phi_extract_time_K_phi_u": float(assembled.get("phi_extract_time_K_phi_u", 0.0)),
                        "phi_convert_time_K_phi_u": float(assembled.get("phi_convert_time_K_phi_u", 0.0)),
                        "phi_extract_time_K_phi_phi": float(assembled.get("phi_extract_time_K_phi_phi", 0.0)),
                        "phi_convert_time_K_phi_phi": float(assembled.get("phi_convert_time_K_phi_phi", 0.0)),
                        "phi_form_call_count_R_phi": int(assembled.get("phi_form_call_count_R_phi", 0)),
                        "phi_form_call_count_K_phi_u": int(assembled.get("phi_form_call_count_K_phi_u", 0)),
                        "phi_form_call_count_K_phi_phi": int(assembled.get("phi_form_call_count_K_phi_phi", 0)),
                        "phi_matrix_extract_call_count_K_phi_u": int(
                            assembled.get("phi_matrix_extract_call_count_K_phi_u", 0)
                        ),
                        "phi_matrix_convert_call_count_K_phi_u": int(
                            assembled.get("phi_matrix_convert_call_count_K_phi_u", 0)
                        ),
                        "phi_matrix_extract_call_count_K_phi_phi": int(
                            assembled.get("phi_matrix_extract_call_count_K_phi_phi", 0)
                        ),
                        "phi_matrix_convert_call_count_K_phi_phi": int(
                            assembled.get("phi_matrix_convert_call_count_K_phi_phi", 0)
                        ),
                    }
                )
            return state, info

        try:
            linear_info = {
                "linear_solver_mode": linear_solver_mode,
                "ksp_type": "",
                "pc_type": "",
                "block_pc_name": "",
                "linear_converged": True,
                "linear_iterations": 0,
                "ksp_reason": 0,
            }
            linear_solve_t0 = time.perf_counter()
            if backend == "dense":
                delta = np.linalg.solve(assembled["global_jacobian_dense"], -residual)
            elif backend == "petsc_block":
                if assembled.get("global_residual_vec") is not None and reuse_matrix_pattern:
                    rhs = assembled["global_residual_vec"].duplicate()
                    _fill_petsc_vec_from_array(rhs, -residual)
                else:
                    rhs = _create_petsc_vec_from_array(-residual, state["domain"].mpi_comm())
                ndof_u = state["u"].vector.getLocalSize()
                ndof_phi = state["phi"].vector.getLocalSize()
                delta, _, linear_info = _solve_petsc_linear_system(
                    assembled["global_jacobian_mat"],
                    rhs,
                    ndof_u=ndof_u,
                    ndof_phi=ndof_phi,
                    linear_solver_mode=linear_solver_mode,
                    ksp_type=ksp_type,
                    pc_type=pc_type,
                    block_pc_name=block_pc_name,
                    reuse_context=(
                        assembled.get("reuse_context")
                        if (reuse_ksp or reuse_fieldsplit_is)
                        else None
                    ),
                    reuse_ksp=reuse_ksp,
                    reuse_fieldsplit_is=reuse_fieldsplit_is,
                    ksp_rtol=ksp_rtol,
                    ksp_atol=ksp_atol,
                    ksp_max_it=ksp_max_it,
                )
                if not linear_info["linear_converged"]:
                    failure_reason = (
                        "monolithic linear solve failed: "
                        f"ksp_reason={linear_info['ksp_reason']}"
                    )
                    break
            else:
                raise ValueError(f"Unsupported monolithic backend: {backend}")
            linear_solve_time = time.perf_counter() - linear_solve_t0
        except (np.linalg.LinAlgError, RuntimeError) as exc:
            failure_reason = f"monolithic solve failed: {exc}"
            break

        ndof_u = state["u"].vector.getLocalSize()
        delta_u = delta[:ndof_u]
        delta_phi = delta[ndof_u:]
        increment_norm = float(np.linalg.norm(delta))

        used_fallback = False
        state_update_t0 = time.perf_counter()
        if line_search:
            line_search_out = _backtracking_line_search(
                state,
                cfg,
                delta_u,
                delta_phi,
                residual_norm,
                backend=backend,
                initial_damping=initial_damping,
                max_backtracks=max_backtracks,
                backtrack_factor=backtrack_factor,
            )
            if not line_search_out["accepted"]:
                failure_reason = "line search failed to produce a trial step"
                break
            step_length = float(line_search_out["step_length"])
            residual_after = float(line_search_out["residual_after"])
            assembled_after = line_search_out["assembled_after"]
            used_fallback = bool(line_search_out["used_fallback"])
        else:
            _apply_state_increment(state, delta_u, delta_phi, scale=1.0)
            step_length = 1.0
            residual_after, assembled_after = _evaluate_total_residual_norm(state, cfg, backend=backend)
        state_update_time = time.perf_counter() - state_update_t0
        newton_step_walltime = time.perf_counter() - newton_step_t0

        row = {
            "newton_iteration": newton_it,
            "mesh_resolution": dof_layout["mesh_resolution"],
            "ndof_u": dof_layout["ndof_u"],
            "ndof_phi": dof_layout["ndof_phi"],
            "build_path": build_path,
            "residual_norm_before": residual_norm,
            "residual_norm_after": residual_after,
            "residual_norm": residual_after,
            "outer_residual_norm_before_linear": residual_norm,
            "outer_residual_norm_after_linear": residual_after,
            "relative_linear_reduction": (
                0.0 if residual_norm <= 0.0 else float(residual_after / residual_norm)
            ),
            "assembly_time": float(assembled.get("assembly_time", 0.0)),
            "struct_block_assembly_time": float(assembled.get("struct_block_assembly_time", 0.0)),
            "contact_block_assembly_time": float(assembled.get("contact_block_assembly_time", 0.0)),
            "phi_block_assembly_time": float(assembled.get("phi_block_assembly_time", 0.0)),
            "contact_quadrature_loop_time": float(assembled.get("contact_quadrature_loop_time", 0.0)),
            "contact_geometry_eval_time": float(assembled.get("contact_geometry_eval_time", 0.0)),
            "contact_query_time": float(assembled.get("contact_query_time", 0.0)),
            "contact_gap_normal_eval_time": float(assembled.get("contact_gap_normal_eval_time", 0.0)),
            "contact_active_filter_time": float(assembled.get("contact_active_filter_time", 0.0)),
            "contact_local_residual_time": float(assembled.get("contact_local_residual_time", 0.0)),
            "contact_local_tangent_uu_time": float(assembled.get("contact_local_tangent_uu_time", 0.0)),
            "contact_local_tangent_uphi_time": float(assembled.get("contact_local_tangent_uphi_time", 0.0)),
            "contact_geometry_eval_call_count": int(assembled.get("contact_geometry_eval_call_count", 0)),
            "contact_geometry_eval_avg_time": float(assembled.get("contact_geometry_eval_avg_time", 0.0)),
            "contact_query_avg_time": float(assembled.get("contact_query_avg_time", 0.0)),
            "contact_query_call_count": int(assembled.get("contact_query_call_count", 0)),
            "contact_query_cache_hit_count": int(assembled.get("contact_query_cache_hit_count", 0)),
            "contact_query_cache_miss_count": int(assembled.get("contact_query_cache_miss_count", 0)),
            "contact_sensitivity_call_count": int(assembled.get("contact_sensitivity_call_count", 0)),
            "contact_sensitivity_cache_hit_count": int(assembled.get("contact_sensitivity_cache_hit_count", 0)),
            "contact_sensitivity_cache_miss_count": int(assembled.get("contact_sensitivity_cache_miss_count", 0)),
            "contact_sensitivity_time": float(assembled.get("contact_sensitivity_time", 0.0)),
            "contact_sensitivity_avg_time": float(assembled.get("contact_sensitivity_avg_time", 0.0)),
            "contact_local_tangent_uphi_call_count": int(assembled.get("contact_local_tangent_uphi_call_count", 0)),
            "contact_local_tangent_uu_call_count": int(assembled.get("contact_local_tangent_uu_call_count", 0)),
            "cell_geometry_cache_hit_count": int(assembled.get("cell_geometry_cache_hit_count", 0)),
            "cell_geometry_cache_miss_count": int(assembled.get("cell_geometry_cache_miss_count", 0)),
            "function_cell_dof_cache_hit_count": int(assembled.get("function_cell_dof_cache_hit_count", 0)),
            "function_cell_dof_cache_miss_count": int(assembled.get("function_cell_dof_cache_miss_count", 0)),
            "vector_subfunction_cache_hit_count": int(assembled.get("vector_subfunction_cache_hit_count", 0)),
            "vector_subfunction_cache_miss_count": int(assembled.get("vector_subfunction_cache_miss_count", 0)),
            "contact_scatter_to_global_time": float(assembled.get("contact_scatter_to_global_time", 0.0)),
            "phi_form_assembly_time": float(assembled.get("phi_form_assembly_time", 0.0)),
            "phi_matrix_extract_time": float(assembled.get("phi_matrix_extract_time", 0.0)),
            "phi_matrix_convert_time": float(assembled.get("phi_matrix_convert_time", 0.0)),
            "phi_rhs_assembly_time": float(assembled.get("phi_rhs_assembly_time", 0.0)),
            "phi_rhs_extract_time": float(assembled.get("phi_rhs_extract_time", 0.0)),
            "phi_matrix_extract_or_convert_time": float(assembled.get("phi_matrix_extract_or_convert_time", 0.0)),
            "phi_rhs_extract_or_convert_time": float(assembled.get("phi_rhs_extract_or_convert_time", 0.0)),
            "phi_dof_lookup_time": float(assembled.get("phi_dof_lookup_time", 0.0)),
            "phi_geometry_helper_time": float(assembled.get("phi_geometry_helper_time", 0.0)),
            "phi_basis_tabulation_time": float(assembled.get("phi_basis_tabulation_time", 0.0)),
            "phi_form_call_count": int(assembled.get("phi_form_call_count", 0)),
            "phi_form_cache_hit_count": int(assembled.get("phi_form_cache_hit_count", 0)),
            "phi_form_cache_miss_count": int(assembled.get("phi_form_cache_miss_count", 0)),
            "phi_matrix_extract_call_count": int(assembled.get("phi_matrix_extract_call_count", 0)),
            "phi_matrix_convert_call_count": int(assembled.get("phi_matrix_convert_call_count", 0)),
            "phi_rhs_extract_call_count": int(assembled.get("phi_rhs_extract_call_count", 0)),
            "phi_dof_lookup_call_count": int(assembled.get("phi_dof_lookup_call_count", 0)),
            "phi_basis_tabulation_call_count": int(assembled.get("phi_basis_tabulation_call_count", 0)),
            "phi_residual_form_assembly_time": float(assembled.get("phi_residual_form_assembly_time", 0.0)),
            "phi_residual_extract_time": float(assembled.get("phi_residual_extract_time", 0.0)),
            "phi_kphiu_form_assembly_time": float(assembled.get("phi_kphiu_form_assembly_time", 0.0)),
            "phi_kphiu_extract_time": float(assembled.get("phi_kphiu_extract_time", 0.0)),
            "phi_kphiu_convert_time": float(assembled.get("phi_kphiu_convert_time", 0.0)),
            "phi_kphiphi_form_assembly_time": float(assembled.get("phi_kphiphi_form_assembly_time", 0.0)),
            "phi_kphiphi_extract_time": float(assembled.get("phi_kphiphi_extract_time", 0.0)),
            "phi_kphiphi_convert_time": float(assembled.get("phi_kphiphi_convert_time", 0.0)),
            "dofmap_lookup_time": float(assembled.get("dofmap_lookup_time", 0.0)),
            "surface_entity_iteration_time": float(assembled.get("surface_entity_iteration_time", 0.0)),
            "basis_eval_or_tabulation_time": float(assembled.get("basis_eval_or_tabulation_time", 0.0)),
            "numpy_temp_allocation_time": float(assembled.get("numpy_temp_allocation_time", 0.0)),
            "block_build_time": float(assembled.get("block_build_time", 0.0)),
            "bc_elimination_time": float(assembled.get("bc_elimination_time", 0.0)),
            "global_matrix_allocation_time": float(assembled.get("global_matrix_allocation_time", 0.0)),
            "global_matrix_fill_time": float(assembled.get("global_matrix_fill_time", 0.0)),
            "global_rhs_build_time": float(assembled.get("global_rhs_build_time", 0.0)),
            "petsc_object_setup_time": float(assembled.get("petsc_object_setup_time", 0.0)),
            "ksp_setup_time": float(linear_info.get("ksp_setup_time", 0.0)),
            "ksp_solve_time": float(linear_info.get("ksp_solve_time", linear_solve_time)),
            "linear_solve_time": float(linear_solve_time),
            "state_update_time": float(state_update_time),
            "newton_step_walltime": float(newton_step_walltime),
            "reused_global_matrix": bool(assembled.get("reused_global_matrix", False)),
            "reused_global_rhs_vec": bool(assembled.get("reused_global_rhs_vec", False)),
            "reused_ksp": bool(linear_info.get("reused_ksp", False)),
            "reused_fieldsplit_is": bool(
                assembled.get("reused_fieldsplit_is", False) or linear_info.get("reused_fieldsplit_is", False)
            ),
            "reused_subksp": bool(linear_info.get("reused_subksp", False)),
            "reuse_attempted": bool(linear_info.get("reuse_attempted", False)),
            "reuse_failed_stage": linear_info.get("reuse_failed_stage", ""),
            "reuse_failure_message": linear_info.get("reuse_failure_message", ""),
            "nnz_global": int(assembled.get("nnz_global", 0)),
            "nnz_Juu": int(assembled.get("nnz_Juu", 0)),
            "nnz_Juphi": int(assembled.get("nnz_Juphi", 0)),
            "nnz_Jphiu": int(assembled.get("nnz_Jphiu", 0)),
            "nnz_Jphiphi": int(assembled.get("nnz_Jphiphi", 0)),
            "increment_norm": increment_norm,
            "active_contact_points": assembled["diagnostics"]["active_contact_points"],
            "negative_gap_sum": assembled["diagnostics"]["negative_gap_sum"],
            "reaction_norm": assembled["diagnostics"]["reaction_norm"],
            "max_penetration": assembled["diagnostics"]["max_penetration"],
            "active_contact_points_after": assembled_after["diagnostics"]["active_contact_points"],
            "negative_gap_sum_after": assembled_after["diagnostics"]["negative_gap_sum"],
            "reaction_norm_after": assembled_after["diagnostics"]["reaction_norm"],
            "max_penetration_after": assembled_after["diagnostics"]["max_penetration"],
            "step_length": step_length,
            "used_fallback": used_fallback,
            "backend": backend,
            "linear_solver_mode": linear_info["linear_solver_mode"],
            "ksp_type": linear_info["ksp_type"],
            "pc_type": linear_info["pc_type"],
            "block_pc_name": linear_info["block_pc_name"],
            "linear_converged": linear_info["linear_converged"],
            "linear_iterations": linear_info["linear_iterations"],
            "ksp_reason": linear_info["ksp_reason"],
        }
        if profile_phi_detail:
            row.update(
                {
                    "phi_form_time_R_phi": float(assembled.get("phi_form_time_R_phi", 0.0)),
                    "phi_form_time_K_phi_u": float(assembled.get("phi_form_time_K_phi_u", 0.0)),
                    "phi_form_time_K_phi_phi": float(assembled.get("phi_form_time_K_phi_phi", 0.0)),
                    "phi_extract_time_K_phi_u": float(assembled.get("phi_extract_time_K_phi_u", 0.0)),
                    "phi_convert_time_K_phi_u": float(assembled.get("phi_convert_time_K_phi_u", 0.0)),
                    "phi_extract_time_K_phi_phi": float(assembled.get("phi_extract_time_K_phi_phi", 0.0)),
                    "phi_convert_time_K_phi_phi": float(assembled.get("phi_convert_time_K_phi_phi", 0.0)),
                    "phi_form_call_count_R_phi": int(assembled.get("phi_form_call_count_R_phi", 0)),
                    "phi_form_call_count_K_phi_u": int(assembled.get("phi_form_call_count_K_phi_u", 0)),
                    "phi_form_call_count_K_phi_phi": int(assembled.get("phi_form_call_count_K_phi_phi", 0)),
                    "phi_matrix_extract_call_count_K_phi_u": int(
                        assembled.get("phi_matrix_extract_call_count_K_phi_u", 0)
                    ),
                    "phi_matrix_convert_call_count_K_phi_u": int(
                        assembled.get("phi_matrix_convert_call_count_K_phi_u", 0)
                    ),
                    "phi_matrix_extract_call_count_K_phi_phi": int(
                        assembled.get("phi_matrix_extract_call_count_K_phi_phi", 0)
                    ),
                    "phi_matrix_convert_call_count_K_phi_phi": int(
                        assembled.get("phi_matrix_convert_call_count_K_phi_phi", 0)
                    ),
                }
            )
        history.append(row)

        if verbose and rank0:
            print(
                f"Newton iter {newton_it}: "
                f"||R||={residual_norm:.6e} -> {residual_after:.6e}, "
                f"||d||={increment_norm:.6e}, alpha={step_length:.3f}, "
                f"lin_it={row['linear_iterations']}, "
                f"asm={row['assembly_time']:.3e}s, "
                f"blk={row['block_build_time']:.3e}s, "
                f"lin={row['linear_solve_time']:.3e}s, "
                f"active={row['active_contact_points_after']}, "
                f"gap_sum={row['negative_gap_sum_after']:.6e}"
            )

        if residual_after < tol_res or increment_norm * step_length < tol_inc:
            total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
            info = {
                "newton_iterations": newton_it,
                "converged": True,
                "failure_reason": "",
                "backend": backend,
                "linear_solver_mode": row["linear_solver_mode"],
                "ksp_type": row["ksp_type"],
                "pc_type": row["pc_type"],
                "block_pc_name": row["block_pc_name"],
                "linear_converged": row["linear_converged"],
                "linear_iterations": row["linear_iterations"],
                "total_linear_iterations": total_linear_iterations,
                "ksp_reason": row["ksp_reason"],
                "residual_norm": residual_after,
                "increment_norm": increment_norm * step_length,
                "active_contact_points": row["active_contact_points_after"],
                "negative_gap_sum": row["negative_gap_sum_after"],
                "reaction_norm": row["reaction_norm_after"],
                "max_penetration": row["max_penetration_after"],
                "step_length": step_length,
                "outer_residual_norm_before_linear": row["outer_residual_norm_before_linear"],
                "outer_residual_norm_after_linear": row["outer_residual_norm_after_linear"],
                "relative_linear_reduction": row["relative_linear_reduction"],
                "assembly_time": row["assembly_time"],
                "block_build_time": row["block_build_time"],
                "linear_solve_time": row["linear_solve_time"],
                "state_update_time": row["state_update_time"],
                "newton_step_walltime": row["newton_step_walltime"],
                **dof_layout,
                "history": history,
            }
            if profile_phi_detail:
                info.update(
                    {
                        "phi_form_time_R_phi": row.get("phi_form_time_R_phi", 0.0),
                        "phi_form_time_K_phi_u": row.get("phi_form_time_K_phi_u", 0.0),
                        "phi_form_time_K_phi_phi": row.get("phi_form_time_K_phi_phi", 0.0),
                        "phi_extract_time_K_phi_u": row.get("phi_extract_time_K_phi_u", 0.0),
                        "phi_convert_time_K_phi_u": row.get("phi_convert_time_K_phi_u", 0.0),
                        "phi_extract_time_K_phi_phi": row.get("phi_extract_time_K_phi_phi", 0.0),
                        "phi_convert_time_K_phi_phi": row.get("phi_convert_time_K_phi_phi", 0.0),
                    }
                )
            return state, info

    if not failure_reason:
        failure_reason = "Reached max_newton_iter without satisfying convergence tolerances"

    last = history[-1] if history else {
        "increment_norm": np.inf,
        "active_contact_points_after": 0,
        "negative_gap_sum_after": 0.0,
        "reaction_norm_after": 0.0,
        "max_penetration_after": 0.0,
        "residual_norm": np.inf,
        "step_length": 0.0,
        "linear_solver_mode": linear_solver_mode,
        "ksp_type": "preonly" if linear_solver_mode == "lu" else ksp_type,
        "pc_type": "lu" if linear_solver_mode == "lu" else pc_type,
        "block_pc_name": "global_lu" if linear_solver_mode == "lu" else block_pc_name,
        "linear_converged": False,
        "linear_iterations": 0,
        "ksp_reason": 0,
    }
    total_linear_iterations = int(sum(item["linear_iterations"] for item in history))
    info = {
        "newton_iterations": len(history),
        "converged": False,
        "failure_reason": failure_reason,
        "backend": backend,
        "linear_solver_mode": last["linear_solver_mode"],
        "ksp_type": last["ksp_type"],
        "pc_type": last["pc_type"],
        "block_pc_name": last["block_pc_name"],
        "linear_converged": last["linear_converged"],
        "linear_iterations": last["linear_iterations"],
        "total_linear_iterations": total_linear_iterations,
        "ksp_reason": last["ksp_reason"],
        "residual_norm": last["residual_norm"],
        "increment_norm": last["increment_norm"],
        "active_contact_points": last["active_contact_points_after"],
        "negative_gap_sum": last["negative_gap_sum_after"],
        "reaction_norm": last["reaction_norm_after"],
        "max_penetration": last["max_penetration_after"],
        "step_length": last["step_length"],
        "outer_residual_norm_before_linear": last.get("outer_residual_norm_before_linear", np.inf),
        "outer_residual_norm_after_linear": last.get("outer_residual_norm_after_linear", np.inf),
        "relative_linear_reduction": last.get("relative_linear_reduction", np.inf),
        "struct_block_assembly_time": last.get("struct_block_assembly_time", 0.0),
        "contact_block_assembly_time": last.get("contact_block_assembly_time", 0.0),
        "phi_block_assembly_time": last.get("phi_block_assembly_time", 0.0),
        "contact_quadrature_loop_time": last.get("contact_quadrature_loop_time", 0.0),
        "contact_geometry_eval_time": last.get("contact_geometry_eval_time", 0.0),
        "contact_gap_normal_eval_time": last.get("contact_gap_normal_eval_time", 0.0),
        "contact_active_filter_time": last.get("contact_active_filter_time", 0.0),
        "contact_local_residual_time": last.get("contact_local_residual_time", 0.0),
        "contact_local_tangent_uu_time": last.get("contact_local_tangent_uu_time", 0.0),
        "contact_local_tangent_uphi_time": last.get("contact_local_tangent_uphi_time", 0.0),
        "contact_scatter_to_global_time": last.get("contact_scatter_to_global_time", 0.0),
        "phi_form_assembly_time": last.get("phi_form_assembly_time", 0.0),
        "phi_matrix_extract_or_convert_time": last.get("phi_matrix_extract_or_convert_time", 0.0),
        "phi_rhs_extract_or_convert_time": last.get("phi_rhs_extract_or_convert_time", 0.0),
        "dofmap_lookup_time": last.get("dofmap_lookup_time", 0.0),
        "surface_entity_iteration_time": last.get("surface_entity_iteration_time", 0.0),
        "basis_eval_or_tabulation_time": last.get("basis_eval_or_tabulation_time", 0.0),
        "numpy_temp_allocation_time": last.get("numpy_temp_allocation_time", 0.0),
        "assembly_time": last.get("assembly_time", 0.0),
        "block_build_time": last.get("block_build_time", 0.0),
        "bc_elimination_time": last.get("bc_elimination_time", 0.0),
        "global_matrix_allocation_time": last.get("global_matrix_allocation_time", 0.0),
        "global_matrix_fill_time": last.get("global_matrix_fill_time", 0.0),
        "global_rhs_build_time": last.get("global_rhs_build_time", 0.0),
        "petsc_object_setup_time": last.get("petsc_object_setup_time", 0.0),
        "ksp_setup_time": last.get("ksp_setup_time", 0.0),
        "ksp_solve_time": last.get("ksp_solve_time", 0.0),
        "linear_solve_time": last.get("linear_solve_time", 0.0),
        "state_update_time": last.get("state_update_time", 0.0),
        "newton_step_walltime": last.get("newton_step_walltime", 0.0),
        "reused_global_matrix": last.get("reused_global_matrix", False),
        "reused_global_rhs_vec": last.get("reused_global_rhs_vec", False),
        "reused_ksp": last.get("reused_ksp", False),
        "reused_fieldsplit_is": last.get("reused_fieldsplit_is", False),
        "reused_subksp": last.get("reused_subksp", False),
        "reuse_attempted": last.get("reuse_attempted", False),
        "reuse_failed_stage": last.get("reuse_failed_stage", ""),
        "reuse_failure_message": last.get("reuse_failure_message", ""),
        "nnz_global": last.get("nnz_global", 0),
        "nnz_Juu": last.get("nnz_Juu", 0),
        "nnz_Juphi": last.get("nnz_Juphi", 0),
        "nnz_Jphiu": last.get("nnz_Jphiu", 0),
        "nnz_Jphiphi": last.get("nnz_Jphiphi", 0),
        **dof_layout,
        "history": history,
    }
    return state, info


def solve_monolithic_contact_loadpath(
    state0,
    cfg,
    load_schedule,
    *,
    backend="dense",
    build_path="current",
    linear_solver_mode="lu",
    ksp_type="gmres",
    pc_type="fieldsplit",
    block_pc_name="fieldsplit_additive_ilu",
    reuse_ksp=False,
    reuse_matrix_pattern=False,
    reuse_fieldsplit_is=False,
    ksp_rtol=1e-10,
    ksp_atol=1e-12,
    ksp_max_it=400,
    max_newton_iter=15,
    max_cutbacks=6,
    tol_res=1e-8,
    tol_inc=1e-8,
    line_search=False,
    initial_damping=1.0,
    max_backtracks=8,
    backtrack_factor=0.5,
    write_outputs=True,
    history_path=None,
    verbose=True,
    min_cutback_increment=1e-8,
    max_load_steps=None,
    max_newton_steps=None,
    max_walltime_seconds=None,
    stop_after_first_nonzero_accepted_step=False,
    profile_assembly_detail=False,
    phi_cache_prime=True,
    phi_scatter_reuse=True,
    profile_phi_detail=True,
    phi_matrix_assembly_backend="python",
):
    """Advance a load path with the monolithic Newton step."""
    rank0 = state0["domain"].mpi_comm().rank == 0
    dof_layout = _state_dof_layout(state0)
    steps = []
    for idx, entry in enumerate(load_schedule, start=1):
        if isinstance(entry, dict):
            step_data = dict(entry)
            step_data.setdefault("step", idx)
        else:
            step_data = {"step": idx, "load_value": float(entry)}
        target_load = float(step_data["load_value"])
        step_data.setdefault("target_load", target_load)
        step_data.setdefault("attempt_load", target_load)
        step_data.setdefault("cutback_level", 0)
        step_data.setdefault("label", f"step_{step_data['step']}")
        steps.append(step_data)

    attempt_history = []
    accepted_history = []
    accepted_load = float(state0.get("current_load_value", 0.0))
    requested_final_target_load = float(steps[-1]["target_load"]) if steps else accepted_load
    output_index = 0
    attempt_state_index = 0
    accepted_state_index = 0
    index = 0
    terminated_early = False
    termination_reason = "completed_schedule"
    termination_category = "completed_full_run"
    start_walltime = time.perf_counter()
    total_newton_steps_used = 0
    phi_cache_prime_time = 0.0

    if build_path == "optimized" and phi_cache_prime:
        phi_cache_prime_time = _prime_phi_cache(state0)

    while index < len(steps):
        elapsed_before_attempt = time.perf_counter() - start_walltime
        if max_walltime_seconds is not None and elapsed_before_attempt >= float(max_walltime_seconds):
            terminated_early = True
            termination_reason = "Reached max_walltime_seconds before next load step"
            termination_category = "terminated_by_walltime"
            break
        step_data = dict(steps[index])
        attempt_load = float(step_data.get("attempt_load", step_data["load_value"]))
        target_load = float(step_data.get("target_load", attempt_load))
        accepted_load_before_step = float(accepted_load)
        load_increment = attempt_load - accepted_load_before_step
        attempt_state_index += 1
        snapshot = _snapshot_state_fields(state0)
        _apply_step_data(state0, step_data)
        load_step_t0 = time.perf_counter()

        if verbose and rank0:
            print(
                f"monolithic step {step_data['step']} attempt load = {attempt_load} "
                f"(target = {target_load}, increment = {load_increment}, "
                f"cutback level = {step_data.get('cutback_level', 0)})"
            )

        state0, step_info = solve_monolithic_contact(
            state0,
            cfg,
            backend=backend,
            build_path=build_path,
            linear_solver_mode=linear_solver_mode,
            ksp_type=ksp_type,
            pc_type=pc_type,
            block_pc_name=block_pc_name,
            reuse_ksp=reuse_ksp,
            reuse_matrix_pattern=reuse_matrix_pattern,
            reuse_fieldsplit_is=reuse_fieldsplit_is,
            ksp_rtol=ksp_rtol,
            ksp_atol=ksp_atol,
            ksp_max_it=ksp_max_it,
            max_newton_iter=max_newton_iter,
            tol_res=tol_res,
            tol_inc=tol_inc,
            line_search=line_search,
            initial_damping=initial_damping,
            max_backtracks=max_backtracks,
            backtrack_factor=backtrack_factor,
            profile_assembly_detail=profile_assembly_detail,
            phi_scatter_reuse=phi_scatter_reuse,
            profile_phi_detail=profile_phi_detail,
            phi_matrix_assembly_backend=phi_matrix_assembly_backend,
            verbose=verbose,
        )

        history_item = {
            "attempt_state_index": attempt_state_index,
            "accepted_state_index": accepted_state_index,
            "accepted": False,
            "step": step_data["step"],
            "load_value": attempt_load,
            "target_load": target_load,
            "attempt_load": attempt_load,
            "accepted_load_before_step": accepted_load_before_step,
            "load_increment": load_increment,
            "converged": bool(step_info["converged"]),
            "backend": backend,
            "build_path": build_path,
            "linear_solver_mode": step_info["linear_solver_mode"],
            "ksp_type": step_info["ksp_type"],
            "pc_type": step_info["pc_type"],
            "block_pc_name": step_info["block_pc_name"],
            "linear_converged": bool(step_info["linear_converged"]),
            "linear_iterations": int(step_info["total_linear_iterations"]),
            "ksp_reason": int(step_info["ksp_reason"]),
            "outer_residual_norm_before_linear": float(step_info["outer_residual_norm_before_linear"]),
            "outer_residual_norm_after_linear": float(step_info["outer_residual_norm_after_linear"]),
            "relative_linear_reduction": float(step_info["relative_linear_reduction"]),
            "newton_iterations": int(step_info["newton_iterations"]),
            "assembly_time": float(step_info.get("assembly_time", 0.0)),
            "struct_block_assembly_time": float(step_info.get("struct_block_assembly_time", 0.0)),
            "contact_block_assembly_time": float(step_info.get("contact_block_assembly_time", 0.0)),
            "phi_block_assembly_time": float(step_info.get("phi_block_assembly_time", 0.0)),
            "contact_quadrature_loop_time": float(step_info.get("contact_quadrature_loop_time", 0.0)),
            "contact_geometry_eval_time": float(step_info.get("contact_geometry_eval_time", 0.0)),
            "contact_query_time": float(step_info.get("contact_query_time", 0.0)),
            "contact_gap_normal_eval_time": float(step_info.get("contact_gap_normal_eval_time", 0.0)),
            "contact_active_filter_time": float(step_info.get("contact_active_filter_time", 0.0)),
            "contact_local_residual_time": float(step_info.get("contact_local_residual_time", 0.0)),
            "contact_local_tangent_uu_time": float(step_info.get("contact_local_tangent_uu_time", 0.0)),
            "contact_local_tangent_uphi_time": float(step_info.get("contact_local_tangent_uphi_time", 0.0)),
            "contact_geometry_eval_call_count": int(step_info.get("contact_geometry_eval_call_count", 0)),
            "contact_geometry_eval_avg_time": float(step_info.get("contact_geometry_eval_avg_time", 0.0)),
            "contact_query_avg_time": float(step_info.get("contact_query_avg_time", 0.0)),
            "contact_query_call_count": int(step_info.get("contact_query_call_count", 0)),
            "contact_query_cache_hit_count": int(step_info.get("contact_query_cache_hit_count", 0)),
            "contact_query_cache_miss_count": int(step_info.get("contact_query_cache_miss_count", 0)),
            "contact_sensitivity_call_count": int(step_info.get("contact_sensitivity_call_count", 0)),
            "contact_sensitivity_cache_hit_count": int(step_info.get("contact_sensitivity_cache_hit_count", 0)),
            "contact_sensitivity_cache_miss_count": int(step_info.get("contact_sensitivity_cache_miss_count", 0)),
            "contact_sensitivity_time": float(step_info.get("contact_sensitivity_time", 0.0)),
            "contact_sensitivity_avg_time": float(step_info.get("contact_sensitivity_avg_time", 0.0)),
            "contact_local_tangent_uphi_call_count": int(step_info.get("contact_local_tangent_uphi_call_count", 0)),
            "contact_local_tangent_uu_call_count": int(step_info.get("contact_local_tangent_uu_call_count", 0)),
            "cell_geometry_cache_hit_count": int(step_info.get("cell_geometry_cache_hit_count", 0)),
            "cell_geometry_cache_miss_count": int(step_info.get("cell_geometry_cache_miss_count", 0)),
            "function_cell_dof_cache_hit_count": int(step_info.get("function_cell_dof_cache_hit_count", 0)),
            "function_cell_dof_cache_miss_count": int(step_info.get("function_cell_dof_cache_miss_count", 0)),
            "vector_subfunction_cache_hit_count": int(step_info.get("vector_subfunction_cache_hit_count", 0)),
            "vector_subfunction_cache_miss_count": int(step_info.get("vector_subfunction_cache_miss_count", 0)),
            "contact_scatter_to_global_time": float(step_info.get("contact_scatter_to_global_time", 0.0)),
            "phi_form_assembly_time": float(step_info.get("phi_form_assembly_time", 0.0)),
            "phi_matrix_extract_time": float(step_info.get("phi_matrix_extract_time", 0.0)),
            "phi_matrix_convert_time": float(step_info.get("phi_matrix_convert_time", 0.0)),
            "phi_rhs_assembly_time": float(step_info.get("phi_rhs_assembly_time", 0.0)),
            "phi_rhs_extract_time": float(step_info.get("phi_rhs_extract_time", 0.0)),
            "phi_matrix_extract_or_convert_time": float(
                step_info.get("phi_matrix_extract_or_convert_time", 0.0)
            ),
            "phi_rhs_extract_or_convert_time": float(
                step_info.get("phi_rhs_extract_or_convert_time", 0.0)
            ),
            "phi_dof_lookup_time": float(step_info.get("phi_dof_lookup_time", 0.0)),
            "phi_geometry_helper_time": float(step_info.get("phi_geometry_helper_time", 0.0)),
            "phi_basis_tabulation_time": float(step_info.get("phi_basis_tabulation_time", 0.0)),
            "phi_form_call_count": int(step_info.get("phi_form_call_count", 0)),
            "phi_form_cache_hit_count": int(step_info.get("phi_form_cache_hit_count", 0)),
            "phi_form_cache_miss_count": int(step_info.get("phi_form_cache_miss_count", 0)),
            "phi_matrix_extract_call_count": int(step_info.get("phi_matrix_extract_call_count", 0)),
            "phi_matrix_convert_call_count": int(step_info.get("phi_matrix_convert_call_count", 0)),
            "phi_rhs_extract_call_count": int(step_info.get("phi_rhs_extract_call_count", 0)),
            "phi_dof_lookup_call_count": int(step_info.get("phi_dof_lookup_call_count", 0)),
            "phi_basis_tabulation_call_count": int(step_info.get("phi_basis_tabulation_call_count", 0)),
            "phi_residual_form_assembly_time": float(step_info.get("phi_residual_form_assembly_time", 0.0)),
            "phi_residual_extract_time": float(step_info.get("phi_residual_extract_time", 0.0)),
            "phi_kphiu_form_assembly_time": float(step_info.get("phi_kphiu_form_assembly_time", 0.0)),
            "phi_kphiu_extract_time": float(step_info.get("phi_kphiu_extract_time", 0.0)),
            "phi_kphiu_convert_time": float(step_info.get("phi_kphiu_convert_time", 0.0)),
            "phi_kphiphi_form_assembly_time": float(step_info.get("phi_kphiphi_form_assembly_time", 0.0)),
            "phi_kphiphi_extract_time": float(step_info.get("phi_kphiphi_extract_time", 0.0)),
            "phi_kphiphi_convert_time": float(step_info.get("phi_kphiphi_convert_time", 0.0)),
            "phi_form_time_R_phi": float(step_info.get("phi_form_time_R_phi", 0.0)),
            "phi_form_time_K_phi_u": float(step_info.get("phi_form_time_K_phi_u", 0.0)),
            "phi_form_time_K_phi_phi": float(step_info.get("phi_form_time_K_phi_phi", 0.0)),
            "phi_extract_time_K_phi_u": float(step_info.get("phi_extract_time_K_phi_u", 0.0)),
            "phi_convert_time_K_phi_u": float(step_info.get("phi_convert_time_K_phi_u", 0.0)),
            "phi_extract_time_K_phi_phi": float(step_info.get("phi_extract_time_K_phi_phi", 0.0)),
            "phi_convert_time_K_phi_phi": float(step_info.get("phi_convert_time_K_phi_phi", 0.0)),
            "phi_form_call_count_R_phi": int(step_info.get("phi_form_call_count_R_phi", 0)),
            "phi_form_call_count_K_phi_u": int(step_info.get("phi_form_call_count_K_phi_u", 0)),
            "phi_form_call_count_K_phi_phi": int(step_info.get("phi_form_call_count_K_phi_phi", 0)),
            "phi_matrix_extract_call_count_K_phi_u": int(
                step_info.get("phi_matrix_extract_call_count_K_phi_u", 0)
            ),
            "phi_matrix_convert_call_count_K_phi_u": int(
                step_info.get("phi_matrix_convert_call_count_K_phi_u", 0)
            ),
            "phi_matrix_extract_call_count_K_phi_phi": int(
                step_info.get("phi_matrix_extract_call_count_K_phi_phi", 0)
            ),
            "phi_matrix_convert_call_count_K_phi_phi": int(
                step_info.get("phi_matrix_convert_call_count_K_phi_phi", 0)
            ),
            "dofmap_lookup_time": float(step_info.get("dofmap_lookup_time", 0.0)),
            "surface_entity_iteration_time": float(step_info.get("surface_entity_iteration_time", 0.0)),
            "basis_eval_or_tabulation_time": float(step_info.get("basis_eval_or_tabulation_time", 0.0)),
            "numpy_temp_allocation_time": float(step_info.get("numpy_temp_allocation_time", 0.0)),
            "block_build_time": float(step_info.get("block_build_time", 0.0)),
            "bc_elimination_time": float(step_info.get("bc_elimination_time", 0.0)),
            "global_matrix_allocation_time": float(step_info.get("global_matrix_allocation_time", 0.0)),
            "global_matrix_fill_time": float(step_info.get("global_matrix_fill_time", 0.0)),
            "global_rhs_build_time": float(step_info.get("global_rhs_build_time", 0.0)),
            "petsc_object_setup_time": float(step_info.get("petsc_object_setup_time", 0.0)),
            "ksp_setup_time": float(step_info.get("ksp_setup_time", 0.0)),
            "ksp_solve_time": float(step_info.get("ksp_solve_time", 0.0)),
            "linear_solve_time": float(step_info.get("linear_solve_time", 0.0)),
            "state_update_time": float(step_info.get("state_update_time", 0.0)),
            "newton_step_walltime": float(step_info.get("newton_step_walltime", 0.0)),
            "residual_norm": float(step_info["residual_norm"]),
            "increment_norm": float(step_info["increment_norm"]),
            "active_contact_points": int(step_info["active_contact_points"]),
            "negative_gap_sum": float(step_info["negative_gap_sum"]),
            "reaction_norm": float(step_info["reaction_norm"]),
            "max_penetration": float(step_info["max_penetration"]),
            "mesh_resolution": dof_layout["mesh_resolution"],
            "ndof_u": dof_layout["ndof_u"],
            "ndof_phi": dof_layout["ndof_phi"],
            "nnz_global": int(step_info.get("nnz_global", 0)),
            "nnz_Juu": int(step_info.get("nnz_Juu", 0)),
            "nnz_Juphi": int(step_info.get("nnz_Juphi", 0)),
            "nnz_Jphiu": int(step_info.get("nnz_Jphiu", 0)),
            "nnz_Jphiphi": int(step_info.get("nnz_Jphiphi", 0)),
            "reused_global_matrix": bool(step_info.get("reused_global_matrix", False)),
            "reused_global_rhs_vec": bool(step_info.get("reused_global_rhs_vec", False)),
            "reused_ksp": bool(step_info.get("reused_ksp", False)),
            "reused_fieldsplit_is": bool(step_info.get("reused_fieldsplit_is", False)),
            "reused_subksp": bool(step_info.get("reused_subksp", False)),
            "reuse_attempted": bool(step_info.get("reuse_attempted", False)),
            "reuse_failed_stage": step_info.get("reuse_failed_stage", ""),
            "reuse_failure_message": step_info.get("reuse_failure_message", ""),
            "linear_iterations_list": [
                int(item.get("linear_iterations", 0)) for item in step_info.get("history", [])
            ],
            "ksp_reason_list": [
                int(item.get("ksp_reason", 0)) for item in step_info.get("history", [])
            ],
            "outer_residual_norm_before_linear_list": [
                float(item.get("outer_residual_norm_before_linear", 0.0))
                for item in step_info.get("history", [])
            ],
            "outer_residual_norm_after_linear_list": [
                float(item.get("outer_residual_norm_after_linear", 0.0))
                for item in step_info.get("history", [])
            ],
            "relative_linear_reduction_list": [
                float(item.get("relative_linear_reduction", 0.0))
                for item in step_info.get("history", [])
            ],
            "assembly_time_list": [
                float(item.get("assembly_time", 0.0)) for item in step_info.get("history", [])
            ],
            "struct_block_assembly_time_list": [
                float(item.get("struct_block_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_block_assembly_time_list": [
                float(item.get("contact_block_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_block_assembly_time_list": [
                float(item.get("phi_block_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_quadrature_loop_time_list": [
                float(item.get("contact_quadrature_loop_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_geometry_eval_time_list": [
                float(item.get("contact_geometry_eval_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_query_time_list": [
                float(item.get("contact_query_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_gap_normal_eval_time_list": [
                float(item.get("contact_gap_normal_eval_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_active_filter_time_list": [
                float(item.get("contact_active_filter_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_local_residual_time_list": [
                float(item.get("contact_local_residual_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_local_tangent_uu_time_list": [
                float(item.get("contact_local_tangent_uu_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_local_tangent_uphi_time_list": [
                float(item.get("contact_local_tangent_uphi_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_geometry_eval_call_count_list": [
                int(item.get("contact_geometry_eval_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_geometry_eval_avg_time_list": [
                float(item.get("contact_geometry_eval_avg_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_query_avg_time_list": [
                float(item.get("contact_query_avg_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_query_call_count_list": [
                int(item.get("contact_query_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_query_cache_hit_count_list": [
                int(item.get("contact_query_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_query_cache_miss_count_list": [
                int(item.get("contact_query_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_sensitivity_call_count_list": [
                int(item.get("contact_sensitivity_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_sensitivity_cache_hit_count_list": [
                int(item.get("contact_sensitivity_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_sensitivity_cache_miss_count_list": [
                int(item.get("contact_sensitivity_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_sensitivity_time_list": [
                float(item.get("contact_sensitivity_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_sensitivity_avg_time_list": [
                float(item.get("contact_sensitivity_avg_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "contact_local_tangent_uphi_call_count_list": [
                int(item.get("contact_local_tangent_uphi_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_local_tangent_uu_call_count_list": [
                int(item.get("contact_local_tangent_uu_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "cell_geometry_cache_hit_count_list": [
                int(item.get("cell_geometry_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "cell_geometry_cache_miss_count_list": [
                int(item.get("cell_geometry_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "function_cell_dof_cache_hit_count_list": [
                int(item.get("function_cell_dof_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "function_cell_dof_cache_miss_count_list": [
                int(item.get("function_cell_dof_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "vector_subfunction_cache_hit_count_list": [
                int(item.get("vector_subfunction_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "vector_subfunction_cache_miss_count_list": [
                int(item.get("vector_subfunction_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "contact_scatter_to_global_time_list": [
                float(item.get("contact_scatter_to_global_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_assembly_time_list": [
                float(item.get("phi_form_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_extract_time_list": [
                float(item.get("phi_matrix_extract_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_convert_time_list": [
                float(item.get("phi_matrix_convert_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_rhs_assembly_time_list": [
                float(item.get("phi_rhs_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_rhs_extract_time_list": [
                float(item.get("phi_rhs_extract_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_extract_or_convert_time_list": [
                float(item.get("phi_matrix_extract_or_convert_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_rhs_extract_or_convert_time_list": [
                float(item.get("phi_rhs_extract_or_convert_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_dof_lookup_time_list": [
                float(item.get("phi_dof_lookup_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_geometry_helper_time_list": [
                float(item.get("phi_geometry_helper_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_basis_tabulation_time_list": [
                float(item.get("phi_basis_tabulation_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_call_count_list": [
                int(item.get("phi_form_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_form_cache_hit_count_list": [
                int(item.get("phi_form_cache_hit_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_form_cache_miss_count_list": [
                int(item.get("phi_form_cache_miss_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_extract_call_count_list": [
                int(item.get("phi_matrix_extract_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_convert_call_count_list": [
                int(item.get("phi_matrix_convert_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_rhs_extract_call_count_list": [
                int(item.get("phi_rhs_extract_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_dof_lookup_call_count_list": [
                int(item.get("phi_dof_lookup_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_basis_tabulation_call_count_list": [
                int(item.get("phi_basis_tabulation_call_count", 0))
                for item in step_info.get("history", [])
            ],
            "phi_residual_form_assembly_time_list": [
                float(item.get("phi_residual_form_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_residual_extract_time_list": [
                float(item.get("phi_residual_extract_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiu_form_assembly_time_list": [
                float(item.get("phi_kphiu_form_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiu_extract_time_list": [
                float(item.get("phi_kphiu_extract_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiu_convert_time_list": [
                float(item.get("phi_kphiu_convert_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiphi_form_assembly_time_list": [
                float(item.get("phi_kphiphi_form_assembly_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiphi_extract_time_list": [
                float(item.get("phi_kphiphi_extract_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_kphiphi_convert_time_list": [
                float(item.get("phi_kphiphi_convert_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_time_R_phi_list": [
                float(item.get("phi_form_time_R_phi", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_time_K_phi_u_list": [
                float(item.get("phi_form_time_K_phi_u", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_time_K_phi_phi_list": [
                float(item.get("phi_form_time_K_phi_phi", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_extract_time_K_phi_u_list": [
                float(item.get("phi_extract_time_K_phi_u", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_convert_time_K_phi_u_list": [
                float(item.get("phi_convert_time_K_phi_u", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_extract_time_K_phi_phi_list": [
                float(item.get("phi_extract_time_K_phi_phi", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_convert_time_K_phi_phi_list": [
                float(item.get("phi_convert_time_K_phi_phi", 0.0))
                for item in step_info.get("history", [])
            ],
            "phi_form_call_count_R_phi_list": [
                int(item.get("phi_form_call_count_R_phi", 0))
                for item in step_info.get("history", [])
            ],
            "phi_form_call_count_K_phi_u_list": [
                int(item.get("phi_form_call_count_K_phi_u", 0))
                for item in step_info.get("history", [])
            ],
            "phi_form_call_count_K_phi_phi_list": [
                int(item.get("phi_form_call_count_K_phi_phi", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_extract_call_count_K_phi_u_list": [
                int(item.get("phi_matrix_extract_call_count_K_phi_u", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_convert_call_count_K_phi_u_list": [
                int(item.get("phi_matrix_convert_call_count_K_phi_u", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_extract_call_count_K_phi_phi_list": [
                int(item.get("phi_matrix_extract_call_count_K_phi_phi", 0))
                for item in step_info.get("history", [])
            ],
            "phi_matrix_convert_call_count_K_phi_phi_list": [
                int(item.get("phi_matrix_convert_call_count_K_phi_phi", 0))
                for item in step_info.get("history", [])
            ],
            "dofmap_lookup_time_list": [
                float(item.get("dofmap_lookup_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "surface_entity_iteration_time_list": [
                float(item.get("surface_entity_iteration_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "basis_eval_or_tabulation_time_list": [
                float(item.get("basis_eval_or_tabulation_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "numpy_temp_allocation_time_list": [
                float(item.get("numpy_temp_allocation_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "block_build_time_list": [
                float(item.get("block_build_time", 0.0)) for item in step_info.get("history", [])
            ],
            "bc_elimination_time_list": [
                float(item.get("bc_elimination_time", 0.0)) for item in step_info.get("history", [])
            ],
            "global_matrix_allocation_time_list": [
                float(item.get("global_matrix_allocation_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "global_matrix_fill_time_list": [
                float(item.get("global_matrix_fill_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "global_rhs_build_time_list": [
                float(item.get("global_rhs_build_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "petsc_object_setup_time_list": [
                float(item.get("petsc_object_setup_time", 0.0))
                for item in step_info.get("history", [])
            ],
            "ksp_setup_time_list": [
                float(item.get("ksp_setup_time", 0.0)) for item in step_info.get("history", [])
            ],
            "ksp_solve_time_list": [
                float(item.get("ksp_solve_time", 0.0)) for item in step_info.get("history", [])
            ],
            "linear_solve_time_list": [
                float(item.get("linear_solve_time", 0.0)) for item in step_info.get("history", [])
            ],
            "state_update_time_list": [
                float(item.get("state_update_time", 0.0)) for item in step_info.get("history", [])
            ],
            "newton_step_walltime_list": [
                float(item.get("newton_step_walltime", 0.0)) for item in step_info.get("history", [])
            ],
            "cutback_level": int(step_data.get("cutback_level", 0)),
            "step_length": float(step_info.get("step_length", 0.0)),
            "cutback_triggered": False,
            "cutback_reason": "",
            "terminated_early": False,
            "termination_reason": "",
            "requested_final_target_load": requested_final_target_load,
            "final_accepted_load": accepted_load_before_step,
            "reached_final_target": False,
            "output_index": output_index,
            "load_step_walltime": float(time.perf_counter() - load_step_t0),
        }
        total_newton_steps_used += int(step_info["newton_iterations"])

        if step_info["converged"]:
            accepted_state_index += 1
            history_item["accepted"] = True
            history_item["accepted_state_index"] = accepted_state_index
            history_item["final_accepted_load"] = attempt_load
            attempt_history.append(history_item)
            accepted_history.append(dict(history_item))
            accepted_load = attempt_load
            output_index += 1
            index += 1
            accepted_nonzero_step_count = sum(
                1 for item in accepted_history if abs(float(item["load_value"])) > 1e-14
            )
            if stop_after_first_nonzero_accepted_step and abs(attempt_load) > 1e-14:
                terminated_early = True
                termination_reason = "Stopped after first nonzero accepted step"
                termination_category = "terminated_by_step_limit"
                break
            if max_load_steps is not None and accepted_nonzero_step_count >= int(max_load_steps):
                terminated_early = True
                termination_reason = f"Reached max_load_steps={int(max_load_steps)}"
                termination_category = "terminated_by_step_limit"
                break
            if max_newton_steps is not None and total_newton_steps_used >= int(max_newton_steps):
                terminated_early = True
                termination_reason = f"Reached max_newton_steps={int(max_newton_steps)}"
                termination_category = "terminated_by_step_limit"
                break
            if max_walltime_seconds is not None and (time.perf_counter() - start_walltime) >= float(max_walltime_seconds):
                terminated_early = True
                termination_reason = "Reached max_walltime_seconds after accepted step"
                termination_category = "terminated_by_walltime"
                break
            continue

        _restore_state_fields(state0, snapshot)

        current_cutback = int(step_data.get("cutback_level", 0))
        load_gap = abs(target_load - accepted_load_before_step)
        same_increment = abs(attempt_load - accepted_load_before_step) < min_cutback_increment
        min_increment_reached = load_gap < min_cutback_increment or same_increment

        if current_cutback >= max_cutbacks or min_increment_reached:
            history_item["terminated_early"] = True
            history_item["termination_reason"] = step_info["failure_reason"]
            history_item["cutback_reason"] = (
                "min_increment_reached" if min_increment_reached else "max_cutbacks_reached"
            )
            history_item["final_accepted_load"] = accepted_load_before_step
            attempt_history.append(history_item)
            terminated_early = True
            termination_reason = step_info["failure_reason"]
            termination_category = "terminated_by_nonconvergence"
            break

        midpoint = accepted_load_before_step + 0.5 * (attempt_load - accepted_load_before_step)
        cutback_step = dict(step_data)
        cutback_step["attempt_load"] = midpoint
        cutback_step["target_load"] = target_load
        cutback_step["cutback_level"] = current_cutback + 1
        history_item["cutback_triggered"] = True
        history_item["cutback_reason"] = step_info["failure_reason"]
        history_item["next_attempt_load"] = midpoint
        attempt_history.append(history_item)
        steps.insert(index, cutback_step)

    result = _summarize_loadpath_result(
        attempt_history,
        accepted_history,
        requested_final_target_load,
        terminated_early,
        termination_reason,
    )
    result.update(dof_layout)
    result["completed_full_run"] = bool(
        not terminated_early and result["reached_final_target"]
    )
    result["terminated_by_walltime"] = termination_category == "terminated_by_walltime"
    result["terminated_by_step_limit"] = termination_category == "terminated_by_step_limit"
    result["terminated_by_nonconvergence"] = termination_category == "terminated_by_nonconvergence"
    result["termination_category"] = termination_category
    result["total_newton_steps_used"] = int(total_newton_steps_used)
    result["total_walltime"] = float(time.perf_counter() - start_walltime)
    result["phi_cache_prime_time"] = float(phi_cache_prime_time)
    for item in attempt_history:
        item["terminated_early"] = result["terminated_early"]
        item["termination_reason"] = result["termination_reason"]
        item["final_accepted_load"] = result["final_accepted_load"]
        item["reached_final_target"] = result["reached_final_target"]

    if write_outputs and rank0:
        _write_history_csv(attempt_history, history_path)

    return state0, result
