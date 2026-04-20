import argparse
import csv
import json
import os

os.environ.setdefault("HWLOC_COMPONENTS", "-gl,-opencl,-cuda")

from check_monolithic_contact import run_monolithic_case
from coupled_solver.monolithic import (
    get_petsc_runtime_info,
    monolithic_block_pc_names,
    recommended_monolithic_contact_options,
)


def _newton_rows(result, summary):
    rows = []
    for accepted_idx, item in enumerate(result.get("accepted_history", []), start=1):
        nsteps = len(item.get("linear_iterations_list", []))
        for idx in range(nsteps):
            rows.append(
                {
                    "mode": summary["mode"],
                    "mesh_resolution": summary["mesh_resolution"],
                    "build_path": summary["build_path"],
                    "ndof_u": summary["ndof_u"],
                    "ndof_phi": summary["ndof_phi"],
                    "total_dofs": summary["total_dofs"],
                    "linear_solver_mode": summary["linear_solver_mode"],
                    "ksp_type": summary["ksp_type"],
                    "pc_type": summary["pc_type"],
                    "block_pc_name": summary["block_pc_name"],
                    "accepted_state_index": accepted_idx,
                    "load_value": item.get("load_value", 0.0),
                    "newton_iteration_within_load": idx + 1,
                    "linear_iterations": item.get("linear_iterations_list", [])[idx],
                    "ksp_reason": item.get("ksp_reason_list", [])[idx],
                    "outer_residual_norm_before_linear": item.get(
                        "outer_residual_norm_before_linear_list", []
                    )[idx],
                    "outer_residual_norm_after_linear": item.get(
                        "outer_residual_norm_after_linear_list", []
                    )[idx],
                    "relative_linear_reduction": item.get(
                        "relative_linear_reduction_list", []
                    )[idx],
                    "assembly_time": item.get("assembly_time_list", [])[idx],
                    "struct_block_assembly_time": item.get(
                        "struct_block_assembly_time_list", []
                    )[idx],
                    "contact_block_assembly_time": item.get(
                        "contact_block_assembly_time_list", []
                    )[idx],
                    "phi_block_assembly_time": item.get("phi_block_assembly_time_list", [])[idx],
                    "block_build_time": item.get("block_build_time_list", [])[idx],
                    "bc_elimination_time": item.get("bc_elimination_time_list", [])[idx],
                    "global_matrix_allocation_time": item.get(
                        "global_matrix_allocation_time_list", []
                    )[idx],
                    "global_matrix_fill_time": item.get("global_matrix_fill_time_list", [])[idx],
                    "global_rhs_build_time": item.get("global_rhs_build_time_list", [])[idx],
                    "petsc_object_setup_time": item.get("petsc_object_setup_time_list", [])[idx],
                    "ksp_setup_time": item.get("ksp_setup_time_list", [])[idx],
                    "ksp_solve_time": item.get("ksp_solve_time_list", [])[idx],
                    "linear_solve_time": item.get("linear_solve_time_list", [])[idx],
                    "state_update_time": item.get("state_update_time_list", [])[idx],
                    "newton_step_walltime": item.get("newton_step_walltime_list", [])[idx],
                    "reused_global_matrix": item.get("reused_global_matrix", False),
                    "reused_global_rhs_vec": item.get("reused_global_rhs_vec", False),
                    "reused_ksp": item.get("reused_ksp", False),
                    "reused_fieldsplit_is": item.get("reused_fieldsplit_is", False),
                    "reused_subksp": item.get("reused_subksp", False),
                    "nnz_global": item.get("nnz_global", 0),
                    "nnz_Juu": item.get("nnz_Juu", 0),
                    "nnz_Juphi": item.get("nnz_Juphi", 0),
                    "nnz_Jphiu": item.get("nnz_Jphiu", 0),
                    "nnz_Jphiphi": item.get("nnz_Jphiphi", 0),
                }
            )
    return rows


def _print_summary(summary, rows):
    print(f"mode = {summary['mode']}")
    print(f"mesh_resolution = {summary['mesh_resolution']}")
    print(f"ndof_u = {summary['ndof_u']}")
    print(f"ndof_phi = {summary['ndof_phi']}")
    print(f"total_dofs = {summary['total_dofs']}")
    print(f"build_path = {summary['build_path']}")
    print(f"linear_solver_mode = {summary['linear_solver_mode']}")
    print(f"ksp_type = {summary['ksp_type']}")
    print(f"pc_type = {summary['pc_type']}")
    print(f"block_pc_name = {summary['block_pc_name']}")
    print(f"completed_full_run = {summary['completed_full_run']}")
    print(f"terminated_by_step_limit = {summary['terminated_by_step_limit']}")
    print(f"terminated_by_walltime = {summary['terminated_by_walltime']}")
    print(f"termination_category = {summary['termination_category']}")
    print(f"accepted_nonzero_step_count = {summary['accepted_nonzero_step_count']}")
    print(f"final_accepted_load = {summary['final_accepted_load']}")
    print(f"total_newton_iterations = {summary['total_newton_iterations_accepted']}")
    print(f"total_linear_iterations = {summary['total_linear_iterations_accepted']}")
    print(f"avg_linear_iterations_per_newton = {summary['avg_linear_iterations_per_newton']}")
    print(f"final_residual_norm = {summary['final_residual_norm']}")
    print(f"final_reaction_norm = {summary['final_reaction_norm']}")
    print(f"final_max_penetration = {summary['final_max_penetration']}")
    if rows:
        print(f"nnz_global = {rows[-1]['nnz_global']}")
        print(f"nnz_Juu = {rows[-1]['nnz_Juu']}")
        print(f"nnz_Juphi = {rows[-1]['nnz_Juphi']}")
        print(f"nnz_Jphiu = {rows[-1]['nnz_Jphiu']}")
        print(f"nnz_Jphiphi = {rows[-1]['nnz_Jphiphi']}")
        print(f"reused_global_matrix = {rows[-1]['reused_global_matrix']}")
        print(f"reused_global_rhs_vec = {rows[-1]['reused_global_rhs_vec']}")
        print(f"reused_ksp = {rows[-1]['reused_ksp']}")
        print(f"reused_fieldsplit_is = {rows[-1]['reused_fieldsplit_is']}")
        print(f"reused_subksp = {rows[-1]['reused_subksp']}")
    print("per_newton_rows:")
    for row in rows:
        print(
            "  load={load_value:.4f} newton={newton_iteration_within_load} "
            "lin_it={linear_iterations} ksp_reason={ksp_reason} "
            "res_before={outer_residual_norm_before_linear:.6e} "
            "res_after={outer_residual_norm_after_linear:.6e} "
            "red={relative_linear_reduction:.6e} "
            "struct={struct_block_assembly_time:.4f}s contact={contact_block_assembly_time:.4f}s "
            "phi={phi_block_assembly_time:.4f}s bc={bc_elimination_time:.4f}s "
            "mat_alloc={global_matrix_allocation_time:.4f}s mat_fill={global_matrix_fill_time:.4f}s "
            "rhs={global_rhs_build_time:.4f}s petsc_setup={petsc_object_setup_time:.4f}s "
            "ksp_setup={ksp_setup_time:.4f}s ksp_solve={ksp_solve_time:.4f}s "
            "step_wall={newton_step_walltime:.4f}s".format(**row)
        )


def _write_outputs(base_name, summary, rows):
    csv_path = f"{base_name}.csv"
    json_path = f"{base_name}.json"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "newton_rows": rows}, handle, indent=2, sort_keys=True)
    print(f"csv_path = {csv_path}")
    print(f"json_path = {json_path}")


def _run_once(args, build_path):
    recommended = recommended_monolithic_contact_options()
    resolved_ksp_type = recommended["ksp_type"] if args.ksp_type is None else args.ksp_type
    resolved_pc_type = recommended["pc_type"] if args.pc_type is None else args.pc_type
    resolved_block_pc_name = (
        recommended["block_pc_name"] if args.block_pc_name is None else args.block_pc_name
    )
    _, result, summary = run_monolithic_case(
        args.mode,
        mesh_scale="small",
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        backend=recommended["backend"],
        build_path=build_path,
        linear_solver_mode=args.linear_solver,
        ksp_type=resolved_ksp_type,
        pc_type=resolved_pc_type,
        block_pc_name=resolved_block_pc_name,
        reuse_ksp=(build_path == "optimized"),
        reuse_matrix_pattern=(build_path == "optimized"),
        ksp_rtol=recommended["ksp_rtol"],
        ksp_atol=recommended["ksp_atol"],
        ksp_max_it=recommended["ksp_max_it"],
        max_newton_iter=recommended["max_newton_iter"],
        line_search=recommended["line_search"],
        damping=recommended["initial_damping"],
        max_load_steps=args.max_load_steps,
        max_newton_steps=args.max_newton_steps,
        max_walltime_seconds=args.max_walltime_seconds,
        write_outputs=False,
        verbose=False,
    )
    rows = _newton_rows(result, summary)
    _print_summary(summary, rows)
    base_name = (
        f"monolithic_build_breakdown_{args.mode}_{summary['mesh_resolution']}_"
        f"{args.linear_solver}_{summary['block_pc_name']}_{build_path}"
    )
    _write_outputs(base_name, summary, rows)


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
    parser.add_argument("--max-load-steps", type=int, default=None)
    parser.add_argument("--max-newton-steps", type=int, default=None)
    parser.add_argument("--max-walltime-seconds", type=float, default=None)
    parser.add_argument("--build-path", choices=["current", "optimized", "all"], default="all")
    args = parser.parse_args()

    runtime = get_petsc_runtime_info()
    print("Monolithic build breakdown")
    print("")
    print(f"petsc_version = {runtime['petsc_version']}")
    print(f"available_block_pc_names = {monolithic_block_pc_names()}")
    print("")

    build_paths = ["current", "optimized"] if args.build_path == "all" else [args.build_path]
    for build_path in build_paths:
        print(f"=== build_path = {build_path} ===")
        _run_once(args, build_path)
        print("")


if __name__ == "__main__":
    main()
