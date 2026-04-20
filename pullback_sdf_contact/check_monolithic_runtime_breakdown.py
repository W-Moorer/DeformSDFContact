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


def build_newton_rows(result, summary):
    rows = []
    accepted_state_index = 0
    accepted_history = result.get("accepted_history", [])
    for item in accepted_history:
        accepted_state_index += 1
        linear_iterations_list = item.get("linear_iterations_list", [])
        ksp_reason_list = item.get("ksp_reason_list", [])
        residual_before_list = item.get("outer_residual_norm_before_linear_list", [])
        residual_after_list = item.get("outer_residual_norm_after_linear_list", [])
        reduction_list = item.get("relative_linear_reduction_list", [])
        assembly_time_list = item.get("assembly_time_list", [])
        block_build_time_list = item.get("block_build_time_list", [])
        linear_solve_time_list = item.get("linear_solve_time_list", [])
        state_update_time_list = item.get("state_update_time_list", [])
        newton_step_walltime_list = item.get("newton_step_walltime_list", [])
        nsteps = max(
            len(linear_iterations_list),
            len(ksp_reason_list),
            len(residual_before_list),
            len(residual_after_list),
            len(reduction_list),
            len(assembly_time_list),
            len(block_build_time_list),
            len(linear_solve_time_list),
            len(state_update_time_list),
            len(newton_step_walltime_list),
        )
        for idx in range(nsteps):
            rows.append(
                {
                    "mode": summary["mode"],
                    "mesh_resolution": summary["mesh_resolution"],
                    "ndof_u": summary["ndof_u"],
                    "ndof_phi": summary["ndof_phi"],
                    "linear_solver_mode": summary["linear_solver_mode"],
                    "ksp_type": summary["ksp_type"],
                    "pc_type": summary["pc_type"],
                    "block_pc_name": summary["block_pc_name"],
                    "accepted_state_index": accepted_state_index,
                    "load_value": item.get("load_value", 0.0),
                    "newton_iteration_within_load": idx + 1,
                    "linear_iterations": (
                        int(linear_iterations_list[idx]) if idx < len(linear_iterations_list) else 0
                    ),
                    "ksp_reason": int(ksp_reason_list[idx]) if idx < len(ksp_reason_list) else 0,
                    "outer_residual_norm_before_linear": (
                        float(residual_before_list[idx]) if idx < len(residual_before_list) else 0.0
                    ),
                    "outer_residual_norm_after_linear": (
                        float(residual_after_list[idx]) if idx < len(residual_after_list) else 0.0
                    ),
                    "relative_linear_reduction": (
                        float(reduction_list[idx]) if idx < len(reduction_list) else 0.0
                    ),
                    "assembly_time": (
                        float(assembly_time_list[idx]) if idx < len(assembly_time_list) else 0.0
                    ),
                    "block_build_time": (
                        float(block_build_time_list[idx]) if idx < len(block_build_time_list) else 0.0
                    ),
                    "linear_solve_time": (
                        float(linear_solve_time_list[idx]) if idx < len(linear_solve_time_list) else 0.0
                    ),
                    "state_update_time": (
                        float(state_update_time_list[idx]) if idx < len(state_update_time_list) else 0.0
                    ),
                    "newton_step_walltime": (
                        float(newton_step_walltime_list[idx]) if idx < len(newton_step_walltime_list) else 0.0
                    ),
                }
            )
    return rows


def print_summary(summary, newton_rows):
    total_step_walltime = sum(row["newton_step_walltime"] for row in newton_rows)
    total_assembly_time = sum(row["assembly_time"] for row in newton_rows)
    total_block_build_time = sum(row["block_build_time"] for row in newton_rows)
    total_linear_solve_time = sum(row["linear_solve_time"] for row in newton_rows)
    total_state_update_time = sum(row["state_update_time"] for row in newton_rows)
    print(f"mode = {summary['mode']}")
    print(f"mesh_resolution = {summary['mesh_resolution']}")
    print(f"ndof_u = {summary['ndof_u']}")
    print(f"ndof_phi = {summary['ndof_phi']}")
    print(f"total_dofs = {summary['total_dofs']}")
    print(f"linear_solver_mode = {summary['linear_solver_mode']}")
    print(f"ksp_type = {summary['ksp_type']}")
    print(f"pc_type = {summary['pc_type']}")
    print(f"block_pc_name = {summary['block_pc_name']}")
    print(f"completed_full_run = {summary['completed_full_run']}")
    print(f"terminated_by_walltime = {summary['terminated_by_walltime']}")
    print(f"terminated_by_step_limit = {summary['terminated_by_step_limit']}")
    print(f"terminated_by_nonconvergence = {summary['terminated_by_nonconvergence']}")
    print(f"termination_category = {summary['termination_category']}")
    print(f"termination_reason = {summary['termination_reason']}")
    print(f"accepted_nonzero_step_count = {summary['accepted_nonzero_step_count']}")
    print(f"final_accepted_load = {summary['final_accepted_load']}")
    print(f"reached_final_target = {summary['reached_final_target']}")
    print(f"total_newton_iterations = {summary['total_newton_iterations_accepted']}")
    print(f"total_linear_iterations = {summary['total_linear_iterations_accepted']}")
    print(f"avg_linear_iterations_per_newton = {summary['avg_linear_iterations_per_newton']}")
    print(f"total_walltime = {summary['total_walltime']}")
    print(f"total_assembly_time = {total_assembly_time}")
    print(f"total_block_build_time = {total_block_build_time}")
    print(f"total_linear_solve_time = {total_linear_solve_time}")
    print(f"total_state_update_time = {total_state_update_time}")
    print(f"total_newton_step_walltime = {total_step_walltime}")
    print(f"final_residual_norm = {summary['final_residual_norm']}")
    print(f"final_reaction_norm = {summary['final_reaction_norm']}")
    print(f"final_max_penetration = {summary['final_max_penetration']}")
    print(f"u_norm = {summary['u_norm']}")
    print(f"phi_delta_norm = {summary['phi_delta_norm']}")
    print(f"ksp_reason_histogram = {summary['ksp_reason_histogram']}")
    print("per_newton_rows:")
    for row in newton_rows:
        print(
            "  load={load_value:.4f} newton={newton_iteration_within_load} "
            "linear_it={linear_iterations} ksp_reason={ksp_reason} "
            "res_before={outer_residual_norm_before_linear:.6e} "
            "res_after={outer_residual_norm_after_linear:.6e} "
            "reduction={relative_linear_reduction:.6e} "
            "assembly={assembly_time:.4f}s block_build={block_build_time:.4f}s "
            "linear={linear_solve_time:.4f}s update={state_update_time:.4f}s "
            "step_wall={newton_step_walltime:.4f}s".format(**row)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "aggressive"], default="baseline")
    parser.add_argument("--mesh-scale", choices=["small", "larger", "xlarger"], default="small")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--linear-solver", choices=["lu", "krylov"], required=True)
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--pc-type", default=None)
    parser.add_argument("--block-pc-name", default=None)
    parser.add_argument("--max-load-steps", type=int, default=None)
    parser.add_argument("--max-newton-steps", type=int, default=None)
    parser.add_argument("--max-walltime-seconds", type=float, default=None)
    parser.add_argument("--stop-after-first-nonzero-accepted-step", action="store_true")
    args = parser.parse_args()

    recommended = recommended_monolithic_contact_options()
    runtime = get_petsc_runtime_info()
    resolved_ksp_type = recommended["ksp_type"] if args.ksp_type is None else args.ksp_type
    resolved_pc_type = recommended["pc_type"] if args.pc_type is None else args.pc_type
    resolved_block_pc_name = (
        recommended["block_pc_name"] if args.block_pc_name is None else args.block_pc_name
    )
    print("Monolithic runtime breakdown")
    print("")
    print(f"petsc_version = {runtime['petsc_version']}")
    print(f"available_block_pc_names = {monolithic_block_pc_names()}")
    print("")
    _, result, summary = run_monolithic_case(
        args.mode,
        mesh_scale=args.mesh_scale,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        backend=recommended["backend"],
        linear_solver_mode=args.linear_solver,
        ksp_type=resolved_ksp_type,
        pc_type=resolved_pc_type,
        block_pc_name=resolved_block_pc_name,
        ksp_rtol=recommended["ksp_rtol"],
        ksp_atol=recommended["ksp_atol"],
        ksp_max_it=recommended["ksp_max_it"],
        max_newton_iter=recommended["max_newton_iter"],
        line_search=recommended["line_search"],
        damping=recommended["initial_damping"],
        max_load_steps=args.max_load_steps,
        max_newton_steps=args.max_newton_steps,
        max_walltime_seconds=args.max_walltime_seconds,
        stop_after_first_nonzero_accepted_step=args.stop_after_first_nonzero_accepted_step,
        write_outputs=False,
        verbose=False,
    )
    newton_rows = build_newton_rows(result, summary)
    print_summary(summary, newton_rows)

    base_name = (
        f"monolithic_runtime_breakdown_{args.mode}_{summary['mesh_resolution']}_"
        f"{summary['linear_solver_mode']}_{summary['block_pc_name']}"
    )
    csv_path = f"{base_name}.csv"
    json_path = f"{base_name}.json"
    if newton_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(newton_rows[0].keys()))
            writer.writeheader()
            for row in newton_rows:
                writer.writerow(row)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "runtime": runtime,
                "summary": summary,
                "newton_rows": newton_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(f"csv_path = {csv_path}")
    print(f"json_path = {json_path}")


if __name__ == "__main__":
    main()
