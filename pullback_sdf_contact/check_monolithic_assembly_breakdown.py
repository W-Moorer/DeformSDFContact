import argparse
import csv
import json
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_indenter_block_contact import build_indenter_state, build_load_schedule
from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import (
    _apply_step_data,
    _prime_phi_cache,
    get_petsc_runtime_info,
    monolithic_block_pc_names,
    recommended_monolithic_contact_options,
    solve_monolithic_contact,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _resolve_reuse_defaults(args, recommended, resolved_pc_type, resolved_block_pc_name):
    default_matrix = bool(args.build_path == "optimized")
    default_fieldsplit_is = bool(
        args.build_path == "optimized" and resolved_pc_type == "fieldsplit"
    )
    default_ksp = bool(
        args.build_path == "optimized"
        and not resolved_block_pc_name.startswith("fieldsplit_schur_")
    )
    reuse_matrix_pattern = default_matrix if args.reuse_matrix_pattern is None else args.reuse_matrix_pattern
    reuse_fieldsplit_is = (
        default_fieldsplit_is if args.reuse_fieldsplit_is is None else args.reuse_fieldsplit_is
    )
    reuse_ksp = default_ksp if args.reuse_ksp is None else args.reuse_ksp
    return reuse_ksp, reuse_matrix_pattern, reuse_fieldsplit_is


def _rows_from_result(result, summary):
    rows = []
    for accepted_idx, item in enumerate(result.get("accepted_history", []), start=1):
        nsteps = len(item.get("linear_iterations_list", []))
        for idx in range(nsteps):
            row = {
                "mode": summary["mode"],
                "mesh_resolution": summary["mesh_resolution"],
                "build_path": summary["build_path"],
                "linear_solver_mode": summary["linear_solver_mode"],
                "ksp_type": summary["ksp_type"],
                "pc_type": summary["pc_type"],
                "block_pc_name": summary["block_pc_name"],
                "ndof_u": summary["ndof_u"],
                "ndof_phi": summary["ndof_phi"],
                "total_dofs": summary["total_dofs"],
                "accepted_state_index": accepted_idx,
                "load_value": item.get("load_value", 0.0),
                "newton_iteration_within_load": idx + 1,
            }
            def _list_value(name, default=0.0):
                values = item.get(f"{name}_list", [])
                if idx < len(values):
                    return values[idx]
                return default
            for key in (
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
                "phi_matrix_extract_time",
                "phi_matrix_convert_time",
                "phi_rhs_assembly_time",
                "phi_rhs_extract_time",
                "phi_matrix_extract_or_convert_time",
                "phi_rhs_extract_or_convert_time",
                "phi_dof_lookup_time",
                "phi_geometry_helper_time",
                "phi_basis_tabulation_time",
                "phi_form_call_count",
                "phi_form_cache_hit_count",
                "phi_form_cache_miss_count",
                "phi_matrix_extract_call_count",
                "phi_matrix_convert_call_count",
                "phi_rhs_extract_call_count",
                "phi_dof_lookup_call_count",
                "phi_basis_tabulation_call_count",
                "phi_residual_form_assembly_time",
                "phi_residual_extract_time",
                "phi_kphiu_form_assembly_time",
                "phi_kphiu_extract_time",
                "phi_kphiu_convert_time",
                "phi_kphiphi_form_assembly_time",
                "phi_kphiphi_extract_time",
                "phi_kphiphi_convert_time",
                "phi_form_time_R_phi",
                "phi_form_time_K_phi_u",
                "phi_form_time_K_phi_phi",
                "phi_extract_time_K_phi_u",
                "phi_convert_time_K_phi_u",
                "phi_extract_time_K_phi_phi",
                "phi_convert_time_K_phi_phi",
                "phi_form_call_count_R_phi",
                "phi_form_call_count_K_phi_u",
                "phi_form_call_count_K_phi_phi",
                "phi_matrix_extract_call_count_K_phi_u",
                "phi_matrix_convert_call_count_K_phi_u",
                "phi_matrix_extract_call_count_K_phi_phi",
                "phi_matrix_convert_call_count_K_phi_phi",
                "phi_scatter_pattern_build_time",
                "phi_scatter_pattern_build_count",
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
            ):
                row[key] = _list_value(key)
            for key in (
                "reused_global_matrix",
                "reused_global_rhs_vec",
                "reused_ksp",
                "reused_fieldsplit_is",
                "reused_subksp",
                "reuse_attempted",
                "reuse_failed_stage",
                "reuse_failure_message",
                "nnz_global",
                "nnz_Juu",
                "nnz_Juphi",
                "nnz_Jphiu",
                "nnz_Jphiphi",
            ):
                row[key] = item.get(key, "")
            rows.append(row)
    return rows


def _summary_from_rows(summary, rows):
    if not rows:
        return summary
    totals = {}
    float_keys = (
        "assembly_time",
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
        "contact_geometry_eval_avg_time",
        "contact_query_avg_time",
        "contact_sensitivity_time",
        "contact_sensitivity_avg_time",
        "contact_scatter_to_global_time",
        "phi_form_assembly_time",
        "phi_matrix_extract_time",
        "phi_matrix_convert_time",
        "phi_rhs_assembly_time",
        "phi_rhs_extract_time",
        "phi_matrix_extract_or_convert_time",
        "phi_rhs_extract_or_convert_time",
        "phi_dof_lookup_time",
        "phi_geometry_helper_time",
        "phi_basis_tabulation_time",
        "phi_residual_form_assembly_time",
        "phi_residual_extract_time",
        "phi_kphiu_form_assembly_time",
        "phi_kphiu_extract_time",
        "phi_kphiu_convert_time",
        "phi_kphiphi_form_assembly_time",
        "phi_kphiphi_extract_time",
        "phi_kphiphi_convert_time",
        "phi_form_time_R_phi",
        "phi_form_time_K_phi_u",
        "phi_form_time_K_phi_phi",
        "phi_extract_time_K_phi_u",
        "phi_convert_time_K_phi_u",
        "phi_extract_time_K_phi_phi",
        "phi_convert_time_K_phi_phi",
        "phi_scatter_pattern_build_time",
        "dofmap_lookup_time",
        "surface_entity_iteration_time",
        "basis_eval_or_tabulation_time",
        "numpy_temp_allocation_time",
        "block_build_time",
        "ksp_setup_time",
        "ksp_solve_time",
        "linear_solve_time",
        "newton_step_walltime",
    )
    for key in float_keys:
        totals[f"total_{key}"] = float(sum(row.get(key, 0.0) for row in rows))
    int_keys = (
        "contact_geometry_eval_call_count",
        "contact_query_call_count",
        "contact_query_cache_hit_count",
        "contact_query_cache_miss_count",
        "contact_sensitivity_call_count",
        "contact_sensitivity_cache_hit_count",
        "contact_sensitivity_cache_miss_count",
        "contact_local_tangent_uphi_call_count",
        "contact_local_tangent_uu_call_count",
        "cell_geometry_cache_hit_count",
        "cell_geometry_cache_miss_count",
        "function_cell_dof_cache_hit_count",
        "function_cell_dof_cache_miss_count",
        "vector_subfunction_cache_hit_count",
        "vector_subfunction_cache_miss_count",
        "phi_form_call_count",
        "phi_form_cache_hit_count",
        "phi_form_cache_miss_count",
        "phi_matrix_extract_call_count",
        "phi_matrix_convert_call_count",
        "phi_rhs_extract_call_count",
        "phi_dof_lookup_call_count",
        "phi_basis_tabulation_call_count",
        "phi_form_call_count_R_phi",
        "phi_form_call_count_K_phi_u",
        "phi_form_call_count_K_phi_phi",
        "phi_matrix_extract_call_count_K_phi_u",
        "phi_matrix_convert_call_count_K_phi_u",
        "phi_matrix_extract_call_count_K_phi_phi",
        "phi_matrix_convert_call_count_K_phi_phi",
        "phi_scatter_pattern_build_count",
    )
    for key in int_keys:
        totals[f"total_{key}"] = int(sum(row.get(key, 0) for row in rows))
    summary.update(totals)
    geometry_calls = max(summary.get("total_contact_geometry_eval_call_count", 0), 1)
    summary["avg_contact_geometry_eval_time_per_call"] = (
        summary.get("total_contact_geometry_eval_time", 0.0) / geometry_calls
    )
    for prefix in ("contact_query", "contact_sensitivity"):
        call_count = summary.get(f"total_{prefix}_call_count", 0)
        hit_count = summary.get(f"total_{prefix}_cache_hit_count", 0)
        summary[f"{prefix}_cache_hit_rate"] = (
            0.0 if call_count == 0 else float(hit_count) / float(call_count)
        )
        summary[f"{prefix}_avg_time_per_call"] = (
            0.0 if call_count == 0 else summary.get(f"total_{prefix}_time", 0.0) / float(call_count)
        )
    phi_form_calls = summary.get("total_phi_form_call_count", 0)
    phi_form_hits = summary.get("total_phi_form_cache_hit_count", 0)
    summary["phi_form_cache_hit_rate"] = (
        0.0 if phi_form_calls == 0 else float(phi_form_hits) / float(phi_form_calls)
    )
    return summary


def _run_single_newton(args, resolved):
    state, solver_cfg, _ = build_indenter_state(
        args.mode,
        mesh_scale="small",
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )
    load_schedule = build_load_schedule(
        args.mode,
        mesh_scale="small",
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )
    nonzero_step = next(step for step in load_schedule if abs(float(step["load_value"])) > 1e-14)
    _apply_step_data(state, nonzero_step)
    phi_cache_prime_time = 0.0
    if args.build_path == "optimized" and args.phi_cache_prime:
        phi_cache_prime_time = _prime_phi_cache(state)
    _, info = solve_monolithic_contact(
        state,
        solver_cfg,
        backend="petsc_block",
        build_path=args.build_path,
        linear_solver_mode=args.linear_solver,
        ksp_type=resolved["ksp_type"],
        pc_type=resolved["pc_type"],
        block_pc_name=resolved["block_pc_name"],
        reuse_ksp=resolved["reuse_ksp"],
        reuse_matrix_pattern=resolved["reuse_matrix_pattern"],
        reuse_fieldsplit_is=resolved["reuse_fieldsplit_is"],
        ksp_rtol=resolved["ksp_rtol"],
        ksp_atol=resolved["ksp_atol"],
        ksp_max_it=resolved["ksp_max_it"],
        max_newton_iter=1,
        line_search=resolved["line_search"],
        initial_damping=resolved["damping"],
        profile_assembly_detail=args.profile_assembly_detail,
        phi_scatter_reuse=args.phi_scatter_reuse,
        profile_phi_detail=(args.phi_profile_mode == "full"),
        phi_matrix_assembly_backend=args.phi_matrix_assembly_backend,
        verbose=False,
    )
    rows = []
    for row in info.get("history", []):
        row_copy = dict(row)
        row_copy["mode"] = args.mode
        row_copy["load_value"] = float(nonzero_step["load_value"])
        for key in (
            "phi_form_time_R_phi",
            "phi_form_time_K_phi_u",
            "phi_form_time_K_phi_phi",
            "phi_convert_time_K_phi_u",
            "phi_convert_time_K_phi_phi",
        ):
            row_copy.setdefault(key, 0.0)
        rows.append(row_copy)
    summary = {
        "mode": args.mode,
        "mesh_resolution": state.get("mesh_resolution", ""),
        "build_path": args.build_path,
        "linear_solver_mode": args.linear_solver,
        "ksp_type": resolved["ksp_type"],
        "pc_type": resolved["pc_type"],
        "block_pc_name": resolved["block_pc_name"],
        "ndof_u": state["u"].vector.getLocalSize(),
        "ndof_phi": state["phi"].vector.getLocalSize(),
        "total_dofs": state["u"].vector.getLocalSize() + state["phi"].vector.getLocalSize(),
        "single_newton_benchmark": True,
        "phi_scatter_reuse": bool(args.phi_scatter_reuse),
        "phi_profile_mode": args.phi_profile_mode,
        "phi_matrix_assembly_backend": args.phi_matrix_assembly_backend,
        "phi_cache_prime_enabled": bool(args.phi_cache_prime),
        "phi_cache_prime_time": float(phi_cache_prime_time),
        "target_load": float(nonzero_step["load_value"]),
        "newton_iterations": int(info.get("newton_iterations", 0)),
        "total_linear_iterations": int(info.get("total_linear_iterations", 0)),
        "final_residual_norm": float(info.get("residual_norm", 0.0)),
        "final_reaction_norm": float(info.get("reaction_norm", 0.0)),
        "final_max_penetration": float(info.get("max_penetration", 0.0)),
    }
    return _summary_from_rows(summary, rows), rows


def _run_loadpath(args, resolved):
    _, result, summary = run_monolithic_case(
        args.mode,
        mesh_scale="small",
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        backend="petsc_block",
        build_path=args.build_path,
        linear_solver_mode=args.linear_solver,
        ksp_type=resolved["ksp_type"],
        pc_type=resolved["pc_type"],
        block_pc_name=resolved["block_pc_name"],
        reuse_ksp=resolved["reuse_ksp"],
        reuse_matrix_pattern=resolved["reuse_matrix_pattern"],
        reuse_fieldsplit_is=resolved["reuse_fieldsplit_is"],
        ksp_rtol=resolved["ksp_rtol"],
        ksp_atol=resolved["ksp_atol"],
        ksp_max_it=resolved["ksp_max_it"],
        max_newton_iter=resolved["max_newton_iter"],
        line_search=resolved["line_search"],
        damping=resolved["damping"],
        max_load_steps=args.max_load_steps,
        max_newton_steps=args.max_newton_steps,
        max_walltime_seconds=args.max_walltime_seconds,
        profile_assembly_detail=args.profile_assembly_detail,
        phi_cache_prime=args.phi_cache_prime,
        phi_scatter_reuse=args.phi_scatter_reuse,
        phi_profile_mode=args.phi_profile_mode,
        phi_matrix_assembly_backend=args.phi_matrix_assembly_backend,
        write_outputs=False,
        verbose=False,
    )
    rows = _rows_from_result(result, summary)
    summary = _summary_from_rows(summary, rows)
    summary["phi_cache_prime_enabled"] = bool(args.phi_cache_prime)
    summary["phi_scatter_reuse"] = bool(args.phi_scatter_reuse)
    summary["phi_profile_mode"] = args.phi_profile_mode
    summary["phi_matrix_assembly_backend"] = args.phi_matrix_assembly_backend
    return summary, rows


def _print_summary(summary, rows):
    print(f"mode = {summary['mode']}")
    print(f"mesh_resolution = {summary['mesh_resolution']}")
    print(f"ndof_u = {summary['ndof_u']}")
    print(f"ndof_phi = {summary['ndof_phi']}")
    print(f"total_dofs = {summary['total_dofs']}")
    print(f"build_path = {summary['build_path']}")
    if "phi_cache_prime_enabled" in summary:
        print(f"phi_cache_prime_enabled = {summary['phi_cache_prime_enabled']}")
    if "phi_scatter_reuse" in summary:
        print(f"phi_scatter_reuse = {summary['phi_scatter_reuse']}")
    if "phi_profile_mode" in summary:
        print(f"phi_profile_mode = {summary['phi_profile_mode']}")
    if "phi_matrix_assembly_backend" in summary:
        print(f"phi_matrix_assembly_backend = {summary['phi_matrix_assembly_backend']}")
    print(f"linear_solver_mode = {summary['linear_solver_mode']}")
    print(f"ksp_type = {summary['ksp_type']}")
    print(f"pc_type = {summary['pc_type']}")
    print(f"block_pc_name = {summary['block_pc_name']}")
    print(f"single_newton_benchmark = {summary.get('single_newton_benchmark', False)}")
    if "final_accepted_load" in summary:
        print(f"final_accepted_load = {summary['final_accepted_load']}")
        print(f"reached_final_target = {summary['reached_final_target']}")
        print(f"total_newton_iterations = {summary['total_newton_iterations_accepted']}")
    else:
        print(f"target_load = {summary['target_load']}")
        print(f"newton_iterations = {summary['newton_iterations']}")
    print(f"total_linear_iterations = {summary['total_linear_iterations'] if 'total_linear_iterations' in summary else summary['total_linear_iterations_accepted']}")
    print(f"final_residual_norm = {summary['final_residual_norm']}")
    print(f"final_reaction_norm = {summary['final_reaction_norm']}")
    print(f"final_max_penetration = {summary['final_max_penetration']}")
    for key in (
        "phi_cache_prime_time",
        "total_contact_block_assembly_time",
        "total_phi_block_assembly_time",
        "total_contact_quadrature_loop_time",
        "total_contact_geometry_eval_time",
        "total_contact_query_time",
        "total_contact_gap_normal_eval_time",
        "total_contact_local_tangent_uu_time",
        "total_contact_local_tangent_uphi_time",
        "total_contact_sensitivity_time",
        "total_phi_form_assembly_time",
        "total_phi_matrix_extract_time",
        "total_phi_matrix_convert_time",
        "total_phi_rhs_assembly_time",
        "total_phi_rhs_extract_time",
        "total_phi_form_time_R_phi",
        "total_phi_form_time_K_phi_u",
        "total_phi_form_time_K_phi_phi",
        "total_phi_extract_time_K_phi_u",
        "total_phi_convert_time_K_phi_u",
        "total_phi_extract_time_K_phi_phi",
        "total_phi_convert_time_K_phi_phi",
        "total_phi_scatter_pattern_build_time",
        "total_phi_matrix_extract_or_convert_time",
        "total_phi_rhs_extract_or_convert_time",
        "total_block_build_time",
        "total_ksp_setup_time",
        "total_ksp_solve_time",
        "total_newton_step_walltime",
    ):
        if key in summary:
            print(f"{key} = {summary[key]}")
    for key in (
        "total_contact_geometry_eval_call_count",
        "avg_contact_geometry_eval_time_per_call",
        "total_contact_query_call_count",
        "total_contact_query_cache_hit_count",
        "contact_query_cache_hit_rate",
        "contact_query_avg_time_per_call",
        "total_contact_sensitivity_call_count",
        "total_contact_sensitivity_cache_hit_count",
        "contact_sensitivity_cache_hit_rate",
        "contact_sensitivity_avg_time_per_call",
        "total_contact_local_tangent_uphi_call_count",
        "total_contact_local_tangent_uu_call_count",
        "total_phi_form_call_count",
        "total_phi_form_cache_hit_count",
        "total_phi_form_cache_miss_count",
        "phi_form_cache_hit_rate",
        "total_phi_matrix_extract_call_count",
        "total_phi_matrix_convert_call_count",
        "total_phi_rhs_extract_call_count",
        "total_phi_form_call_count_R_phi",
        "total_phi_form_call_count_K_phi_u",
        "total_phi_form_call_count_K_phi_phi",
        "total_phi_matrix_extract_call_count_K_phi_u",
        "total_phi_matrix_convert_call_count_K_phi_u",
        "total_phi_matrix_extract_call_count_K_phi_phi",
        "total_phi_matrix_convert_call_count_K_phi_phi",
        "total_phi_scatter_pattern_build_count",
    ):
        if key in summary:
            print(f"{key} = {summary[key]}")
    print("per_newton_rows:")
    for row in rows:
        row_display = dict(row)
        for key in (
            "phi_form_time_R_phi",
            "phi_form_time_K_phi_u",
            "phi_form_time_K_phi_phi",
            "phi_convert_time_K_phi_u",
            "phi_convert_time_K_phi_phi",
        ):
            row_display.setdefault(key, 0.0)
        print(
            "  load={load_value:.4f} newton={newton} "
            "lin_it={linear_iterations} ksp_reason={ksp_reason} "
            "asm={assembly_time:.4f}s contact={contact_block_assembly_time:.4f}s "
            "phi={phi_block_assembly_time:.4f}s phi_form={phi_form_assembly_time:.4f}s "
            "Rphi_form={phi_form_time_R_phi:.4f}s "
            "Kphiu_form={phi_form_time_K_phi_u:.4f}s Kphiu_conv={phi_convert_time_K_phi_u:.4f}s "
            "Kphiphi_form={phi_form_time_K_phi_phi:.4f}s Kphiphi_conv={phi_convert_time_K_phi_phi:.4f}s "
            "phi_mextract={phi_matrix_extract_time:.4f}s phi_mconvert={phi_matrix_convert_time:.4f}s "
            "build={block_build_time:.4f}s "
            "ksp_setup={ksp_setup_time:.4f}s ksp_solve={ksp_solve_time:.4f}s "
            "geom_calls={contact_geometry_eval_call_count} "
            "query_hits={contact_query_cache_hit_count}/{contact_query_call_count} "
            "sens_hits={contact_sensitivity_cache_hit_count}/{contact_sensitivity_call_count} "
            "res_before={outer_residual_norm_before_linear:.6e} "
            "res_after={outer_residual_norm_after_linear:.6e} "
            "red={relative_linear_reduction:.6e}".format(
                newton=row_display.get("newton_iteration", row_display.get("newton_iteration_within_load", 0)),
                **row_display,
            )
        )


def _write_outputs(base_name, summary, rows):
    csv_path = f"{base_name}.csv"
    json_path = f"{base_name}.json"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "newton_rows": rows}, handle, indent=2, sort_keys=True)
    print(f"csv_path = {csv_path}")
    print(f"json_path = {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive"], default="baseline")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--linear-solver", choices=["lu", "krylov"], required=True)
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--pc-type", default=None)
    parser.add_argument("--block-pc-name", default=None)
    parser.add_argument("--build-path", choices=["current", "optimized", "all"], default="all")
    parser.add_argument("--reuse-ksp", type=_parse_bool, default=None)
    parser.add_argument("--reuse-matrix-pattern", type=_parse_bool, default=None)
    parser.add_argument("--reuse-fieldsplit-is", type=_parse_bool, default=None)
    parser.add_argument("--phi-cache-prime", type=_parse_bool, default=None)
    parser.add_argument("--phi-scatter-reuse", type=_parse_bool, default=None)
    parser.add_argument("--phi-profile-mode", choices=["full", "light"], default=None)
    parser.add_argument(
        "--phi-matrix-assembly-backend",
        choices=["python", "cpp_petsc"],
        default=None,
    )
    parser.add_argument("--max-load-steps", type=int, default=None)
    parser.add_argument("--max-newton-steps", type=int, default=None)
    parser.add_argument("--max-walltime-seconds", type=float, default=None)
    parser.add_argument("--profile-assembly-detail", action="store_true")
    parser.add_argument("--profile-contact-geometry-detail", action="store_true")
    parser.add_argument("--profile-phi-detail", action="store_true")
    parser.add_argument("--report-cache-stats", action="store_true")
    parser.add_argument("--report-phi-cache-stats", action="store_true")
    parser.add_argument("--microbenchmark-contact-geometry", action="store_true")
    parser.add_argument("--microbenchmark-phi", action="store_true")
    parser.add_argument("--single-newton-benchmark", action="store_true")
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    runtime = get_petsc_runtime_info()
    print("Monolithic assembly breakdown")
    print("")
    print(f"petsc_version = {runtime['petsc_version']}")
    print(f"available_block_pc_names = {monolithic_block_pc_names()}")
    print("")

    build_paths = ["current", "optimized"] if args.build_path == "all" else [args.build_path]
    for build_path in build_paths:
        args.build_path = build_path
        resolved_ksp_type = recommended["ksp_type"] if args.ksp_type is None else args.ksp_type
        resolved_pc_type = recommended["pc_type"] if args.pc_type is None else args.pc_type
        resolved_block_pc_name = (
            recommended["block_pc_name"] if args.block_pc_name is None else args.block_pc_name
        )
        if args.linear_solver == "lu":
            resolved_ksp_type = "preonly"
            resolved_pc_type = "lu"
            resolved_block_pc_name = "global_lu"
        elif resolved_pc_type != "fieldsplit" and args.block_pc_name is None:
            resolved_block_pc_name = f"global_{resolved_pc_type}"
        reuse_ksp, reuse_matrix_pattern, reuse_fieldsplit_is = _resolve_reuse_defaults(
            args,
            recommended,
            resolved_pc_type,
            resolved_block_pc_name,
        )
        resolved = {
            "ksp_type": resolved_ksp_type,
            "pc_type": resolved_pc_type,
            "block_pc_name": resolved_block_pc_name,
            "reuse_ksp": reuse_ksp,
            "reuse_matrix_pattern": reuse_matrix_pattern,
            "reuse_fieldsplit_is": reuse_fieldsplit_is,
            "ksp_rtol": recommended["ksp_rtol"],
            "ksp_atol": recommended["ksp_atol"],
            "ksp_max_it": recommended["ksp_max_it"],
            "max_newton_iter": recommended["max_newton_iter"],
            "line_search": recommended["line_search"],
            "damping": recommended["initial_damping"],
        }
        if args.phi_cache_prime is None:
            args.phi_cache_prime = bool(recommended["phi_cache_prime"])
        if args.phi_scatter_reuse is None:
            args.phi_scatter_reuse = bool(recommended["phi_scatter_reuse"])
        if args.phi_profile_mode is None:
            args.phi_profile_mode = str(recommended["phi_profile_mode"])
        if args.phi_matrix_assembly_backend is None:
            args.phi_matrix_assembly_backend = str(
                recommended["phi_matrix_assembly_backend"]
            )

        print(f"=== build_path = {build_path} ===")
        print(f"reuse_ksp = {reuse_ksp}")
        print(f"reuse_matrix_pattern = {reuse_matrix_pattern}")
        print(f"reuse_fieldsplit_is = {reuse_fieldsplit_is}")
        print(f"phi_cache_prime = {args.phi_cache_prime}")
        print(f"phi_scatter_reuse = {args.phi_scatter_reuse}")
        print(f"phi_profile_mode = {args.phi_profile_mode}")
        print(f"phi_matrix_assembly_backend = {args.phi_matrix_assembly_backend}")
        args.profile_assembly_detail = bool(
            args.profile_assembly_detail
            or args.profile_contact_geometry_detail
            or args.report_cache_stats
            or args.microbenchmark_contact_geometry
        )
        if args.microbenchmark_contact_geometry or args.microbenchmark_phi:
            args.single_newton_benchmark = True
        if args.single_newton_benchmark:
            summary, rows = _run_single_newton(args, resolved)
        else:
            summary, rows = _run_loadpath(args, resolved)
        _print_summary(summary, rows)
        base_name = (
            f"monolithic_assembly_breakdown_{args.mode}_{summary['mesh_resolution']}_"
            f"{args.linear_solver}_{summary['block_pc_name']}_{build_path}"
        )
        base_name += (
            f"_phiPrime{int(bool(args.phi_cache_prime))}"
            f"_scatter{int(bool(args.phi_scatter_reuse))}"
            f"_phiProf{args.phi_profile_mode}"
        )
        base_name += f"_phiAsm{args.phi_matrix_assembly_backend}"
        if args.single_newton_benchmark:
            base_name += "_single_newton"
        _write_outputs(base_name, summary, rows)
        print("")


if __name__ == "__main__":
    main()
